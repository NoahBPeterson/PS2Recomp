#ifndef PS2_VU1_ISSUE_H
#define PS2_VU1_ISSUE_H
// The per-instruction-pair step of the VU interpreter and the helpers it inlines, plus the lazy
// commit step of translated VU1 blocks (recCommitTo). Included by ps2_vu1_core.cpp (the interpreter
// loop) and by the translated blocks (T-0121, src/vu1_aot), which mirror issuePair() pair by pair.

#include "ps2_vu1_detail.h"
#include "ps2_vu1_decode.h"

#include <array>
#include <cstring>

inline void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

inline void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    if (commitDue())
        commitReadyPipelines();
    if (m_xgkick.active)
        progressXgkick();
}

inline void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
        advanceOneCycle();
}

inline uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    uint64_t ready = m_cycle;
    for (uint32_t index = 0; index < decoded.vfReadLaneCount; ++index)
    {
        const uint8_t lane = decoded.vfReadLanes[index];
        ready = std::max(ready, m_vfReady[lane >> 2][lane & 3u]);
    }
    for (uint32_t index = 0; index < decoded.viReadCount; ++index)
        ready = std::max(ready, m_viReady[decoded.viReads[index]]);
    for (uint32_t index = 0; index < decoded.accReadCount; ++index)
        ready = std::max(ready, m_accReady[decoded.accReads[index]]);

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

inline void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    for (uint32_t index = 0; index < decoded.lowerVfWriteCount; ++index)
    {
        const uint8_t lane = decoded.lowerVfWriteLanes[index];
        m_vfReady[lane >> 2][lane & 3u] = m_cycle + decoded.lowerVfWriteLatency;
    }
    for (uint32_t index = 0; index < decoded.upperVfWriteCount; ++index)
    {
        const uint8_t lane = decoded.upperVfWriteLanes[index];
        m_vfReady[lane >> 2][lane & 3u] = m_cycle + decoded.upperVfWriteLatency;
    }
    for (uint32_t index = 0; index < decoded.viWriteCount; ++index)
        m_viReady[decoded.viWrites[index]] = m_cycle + decoded.viWriteLatency;
    for (uint32_t index = 0; index < decoded.accWriteCount; ++index)
        m_accReady[decoded.accWrites[index]] = m_cycle + kAccForwardLatency;
}

// One VU instruction pair at m_state.pc: wait out its stalls, execute it, account its pipeline
// effects and advance one cycle (the interpreter loop, run). Translated blocks emit the same steps
// with the pair known (tools/vu1_rec/vu1_rec_gen.cpp); keep the two in step.
[[gnu::always_inline]] inline VU1Interpreter::IssueResult VU1Interpreter::issuePair(
    const DecodedInstructionPair &decoded, uint64_t budgetEnd, uint8_t *vuData, uint32_t dataSize, GS &gs,
    PS2Memory *memory, uint32_t codeSize)
{
    bool ended = false;
    if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
    {
        reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
        return IssueStopped;
    }

    uint64_t readyCycle = calculatePairReadyCycle(decoded);
    while (readyCycle > m_cycle)
    {
        if (readyCycle >= budgetEnd)
        {
            advanceTo(budgetEnd);
            break;
        }
        advanceTo(readyCycle);
        readyCycle = calculatePairReadyCycle(decoded);
    }
    if (m_cycle >= budgetEnd)
        return IssueStopped;

    // A VI write in the delay slot of a taken branch is followed by the branch target, which
    // starts a new block: a branch there reads the new value. Only a write by the instruction
    // directly before a branch in program order is hidden from it.
    const bool inTakenDelaySlot = m_state.branchPending && m_state.branchDelay == 0u;
    const uint8_t writtenVi = decoded.viWriteCount != 0u ? decoded.viWrites[0] : 0u; // lowest written VI
    const int32_t oldVi = writtenVi != 0u ? m_state.vi[writtenVi] : 0;

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    const bool hasUpperWrite = upperWrite.reg != 0u;
    const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
    const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
    float oldUpperVf[4]{};
    float newUpperVf[4]{};
    float oldLowerVf[4]{};
    float newLowerVf[4]{};
    float oldAcc[4]{};
    float newAcc[4]{};
    // On VU1's immediate-write path, the specialized FMAC handlers write exactly their
    // dest lanes (= the usage lanes), so there is nothing to restore for them.
    const bool immediateWrites = m_unit == Unit::VU1 && m_cycle + kImmediateWriteMargin < budgetEnd;
    const bool upperExactLanes = immediateWrites && decoded.upperHandler != nullptr;
    if (hasUpperWrite && !upperExactLanes)
        std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
    if (hasDistinctLowerWrite)
        std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
    if (decoded.upperUsage.accWrite != 0u && !upperExactLanes)
        std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));

    if (decoded.iBit)
    {
        if (decoded.upperHandler)
            (this->*decoded.upperHandler)(decoded.upper);
        else
            execUpper(decoded.upper);
        float immediate = 0.0f;
        std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
        m_state.i = normalizeOperand(immediate);
    }
    else if (decoded.upperVfShadowReg != 0u)
    {
        float oldVf[4]{};
        float upperVf[4]{};
        std::memcpy(oldVf,
                    m_state.vf[decoded.upperVfShadowReg],
                    sizeof(oldVf));
        if (decoded.upperHandler)
            (this->*decoded.upperHandler)(decoded.upper);
        else
            execUpper(decoded.upper);
        std::memcpy(upperVf,
                    m_state.vf[decoded.upperVfShadowReg],
                    sizeof(upperVf));
        std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                    oldVf,
                    sizeof(oldVf));
        execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                    upperVf,
                    sizeof(upperVf));
    }
    else
    {
        if (decoded.upperHandler)
            (this->*decoded.upperHandler)(decoded.upper);
        else
            execUpper(decoded.upper);
        execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
    }

    m_viBranchBackupValid = false;

    // VU1 writes VF/ACC results into the registers at issue instead of queueing them. A queued
    // write is invisible until its ready cycle, but every instruction reading one of its lanes
    // stalls until then (calculatePairReadyCycle on m_vfReady/m_accReady), and among
    // overlapping writes the last issued wins either way, so instructions cannot tell the
    // difference. What can: the register state when a run stops on its cycle budget with
    // writes in flight (they stay invisible, and a following MSCAL drops them). So the last
    // kImmediateWriteMargin cycles before the budget queue as before; every write issued
    // earlier has landed by the end. Immediate writes still take a sequence number, so a
    // queued write left over from such a run cannot land on top of them after a resume.
    // Lanes outside the write mask keep their old values, as a queued write would leave them.
    // VU0 always queues: the EE can read its registers while a microprogram runs.
    if (immediateWrites)
    {
        auto land = [this](float *reg, const float *old, uint8_t lanes, std::array<uint64_t, 4> &latest)
        {
            const uint64_t sequence = ++m_nextWriteSequence;
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((lanes & laneForComponent(component)) == 0u)
                    reg[component] = old[component];
                else
                    latest[component] = sequence;
            }
        };
        if (hasUpperWrite && upperWrite.lanes != 0u)
        {
            if (upperExactLanes)
            {
                const uint64_t sequence = ++m_nextWriteSequence;
                for (uint32_t component = 0; component < 4u; ++component)
                    if ((upperWrite.lanes & laneForComponent(component)) != 0u)
                        m_vfLatestWrite[upperWrite.reg][component] = sequence;
            }
            else
                land(m_state.vf[upperWrite.reg], oldUpperVf, upperWrite.lanes, m_vfLatestWrite[upperWrite.reg]);
            m_vfPendingUntil = std::max(m_vfPendingUntil, m_cycle + decoded.upperVfWriteLatency);
        }
        else if (hasUpperWrite && !upperExactLanes)
            std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
        if (hasDistinctLowerWrite && lowerWrite.lanes != 0u)
        {
            land(m_state.vf[lowerWrite.reg], oldLowerVf, lowerWrite.lanes, m_vfLatestWrite[lowerWrite.reg]);
            m_vfPendingUntil = std::max(m_vfPendingUntil, m_cycle + decoded.lowerVfWriteLatency);
        }
        else if (hasDistinctLowerWrite)
            std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
        if (decoded.upperUsage.accWrite != 0u)
        {
            if (upperExactLanes)
            {
                const uint64_t sequence = ++m_nextWriteSequence;
                for (uint32_t component = 0; component < 4u; ++component)
                    if ((decoded.upperUsage.accWrite & laneForComponent(component)) != 0u)
                        m_accLatestWrite[component] = sequence;
            }
            else
                land(m_state.acc, oldAcc, decoded.upperUsage.accWrite, m_accLatestWrite);
            m_accPendingUntil = std::max(m_accPendingUntil, m_cycle + kAccForwardLatency);
        }
    }
    else
    {
        if (hasUpperWrite)
        {
            std::memcpy(newUpperVf, m_state.vf[upperWrite.reg], sizeof(newUpperVf));
            std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
            const uint32_t latency =
                decoded.upperUsage.vfLatency != 0u
                    ? decoded.upperUsage.vfLatency
                    : decoded.upperUsage.latency;
            queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
        }
        if (hasDistinctLowerWrite)
        {
            std::memcpy(newLowerVf, m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
            std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
            const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                         ? decoded.lowerUsage.vfLatency
                                         : decoded.lowerUsage.latency;
            queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
        }
        if (decoded.upperUsage.accWrite != 0u)
        {
            std::memcpy(newAcc, m_state.acc, sizeof(newAcc));
            std::memcpy(m_state.acc, oldAcc, sizeof(oldAcc));
            // ACC is forwarded to the next upper instruction. Its arithmetic
            // flags still use the normal four-cycle FMAC timeline.
            queueAccWrite(decoded.upperUsage.accWrite, newAcc,
                          kAccForwardLatency);
        }
    }
    if (writtenVi != 0u)
    {
        const int32_t newVi = m_state.vi[writtenVi];
        m_state.vi[writtenVi] = oldVi;
        const uint32_t latency =
            decoded.lowerUsage.viLatency != 0u
                ? decoded.lowerUsage.viLatency
                : decoded.lowerUsage.latency;
        queueViWrite(writtenVi, newVi, latency);
    }

    markPairWrites(decoded);
    if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead && !inTakenDelaySlot)
        recordViWriteForBranch(writtenVi, oldVi);

    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    m_state.vi[0] = 0;

    uint32_t nextPc = m_state.pc + 8u;
    if (nextPc >= codeSize)
        nextPc = 0u;
    m_state.pc = nextPc;

    if (m_state.branchPending)
    {
        if (m_state.branchDelay == 0u)
        {
            m_state.pc = m_state.branchTarget & microAddressMask();
            m_state.branchPending = false;
        }
        else
        {
            --m_state.branchDelay;
        }
    }

    const bool dHalt = decoded.dBit && m_state.dBitEnabled;
    const bool tHalt = decoded.tBit && m_state.tBitEnabled;
    const bool haltBit = dHalt || tHalt;
    const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

    if (m_state.haltAfterDelaySlot)
    {
        m_state.stoppedByD = m_pendingHaltD;
        m_state.stoppedByT = m_pendingHaltT;
        ended = true;
    }
    else if (m_state.ebit)
        ended = true;
    else if (haltBit && !haltBranch)
    {
        m_state.stoppedByD = dHalt;
        m_state.stoppedByT = tHalt;
        ended = true;
    }
    else if (decoded.eBit)
        m_state.ebit = true;
    else if (haltBranch)
    {
        m_state.haltAfterDelaySlot = true;
        m_pendingHaltD = dHalt;
        m_pendingHaltT = tHalt;
    }

    advanceOneCycle();
    return ended ? IssueEnded : IssueContinue;
}

// Process the cycle boundaries (m_recCommitted, cycle] the way advanceOneCycle() would have: the
// commits due, then PATH1 progress. Translated blocks call it before anything that depends on them.
// With writes queued (the interpreter's, left from before a block), the general per-boundary step.
inline void VU1Interpreter::recCommitQueuedTo(uint64_t cycle)
{
    const uint64_t current = m_cycle;
    for (uint64_t boundary = m_recCommitted + 1u; boundary <= cycle; ++boundary)
    {
        m_cycle = boundary;
        if (commitDue())
            commitReadyPipelines();
        if (m_xgkick.active)
            progressXgkick();
    }
    m_recCommitted = cycle;
    m_cycle = current;
}

[[gnu::always_inline]] inline void VU1Interpreter::recCommitTo(uint64_t cycle)
{
    if (cycle <= m_recCommitted)
        return;
    if ((m_storeMask | m_vfWriteMask | m_viWriteMask | m_accWriteMask) != 0u)
    {
        recCommitQueuedTo(cycle); // queued writes (only from the interpreter)
        return;
    }
    const uint64_t current = m_cycle;
    {
        // commitReadyPipelines with an empty wheel commits, per boundary, the flag updates due,
        // then Q, then P. Those touch disjoint state (MAC / STATUS Z S U O and their sticky bits /
        // CLIP; Q and STATUS D I; P), so each kind can go on its own, in its own order. PATH1 moves
        // VU memory to the GIF, independent of all of them.
        const uint64_t lastFlags = std::min(cycle, m_flagPendingUntil);
        for (uint64_t boundary = std::max<uint64_t>(m_recCommitted + 1u, kFmacLatency); boundary <= lastFlags; ++boundary)
        {
            FlagSlot &slot = m_flagRing[(boundary - kFmacLatency) & (kFlagRingSize - 1u)];
            if (slot.issueCycle == boundary - kFmacLatency && slot.count != 0u)
                commitFlagSlot(slot);
        }
        if (m_fdiv.valid || m_efu[0].valid || m_efu[1].valid)
        {
            m_cycle = cycle;
            // P results due in this span commit in ready order (one per boundary, entry 0 first)
            if (m_efu[0].valid && m_efu[1].valid && m_efu[1].readyCycle < m_efu[0].readyCycle && m_efu[1].readyCycle <= cycle)
            {
                m_state.p = m_efu[1].value;
                m_efu[1] = {};
            }
            commitScalars();
        }
        if (m_xgkick.active)
            progressXgkick(static_cast<uint32_t>(cycle - m_recCommitted));
    }
    m_recCommitted = cycle;
    m_cycle = current;
}

#endif
