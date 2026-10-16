#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <ps2_log.h>

void ps2xDumpRecentVif1Unpacks(); // ps2_vif1_interpreter.cpp



void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    reset();
}

void VU1Interpreter::resetScheduler()
{
    m_flagRing = {};
    m_flagPendingUntil = 0u;
    m_fdiv = {};
    m_efu = {};
    m_storePipeline = {};
    m_vfWritePipeline = {};
    m_viWritePipeline = {};
    m_accWritePipeline = {};
    m_storeMask = m_vfWriteMask = m_viWriteMask = m_accWriteMask = 0u;
    m_vfPendingUntil = m_accPendingUntil = 0u;
    for (WheelBucket &bucket : m_wheel)
        bucket.count = 0u;
    m_xgkick.resetState();
    m_vfReady = {};
    m_viReady = {};
    m_accReady = {};
    m_vfLatestWrite = {};
    m_viLatestWrite = {};
    m_accLatestWrite = {};
    m_nextWriteSequence = 0;
    m_efuResourceReady = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
}

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    if (FlagSlot *slot = currentFlagSlot())
        for (uint32_t index = 0; index < slot->count; ++index)
            slot->entries[index].writesStatus = false;

    if (FlagPipelineEntry *entry = allocFlagEntry(false, 0xFFFFFFFEu))
    {
        entry->status = static_cast<uint32_t>(immediate) & 0xFC0u;
        entry->writesSticky = true;
    }
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (FlagPipelineEntry *entry = allocFlagEntry(true, 0xFFFFFFFDu))
    {
        entry->clip = m_workingClip;
        entry->writesClip = true;
    }
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    if (FlagSlot *slot = currentFlagSlot())
        for (uint32_t index = 0; index < slot->count; ++index)
            slot->entries[index].writesClip = false;
    if (FlagPipelineEntry *entry = allocFlagEntry(false, 0xFFFFFFFAu))
    {
        entry->clip = m_workingClip;
        entry->writesClip = true;
    }
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    if (const int slot = firstFreeSlot(m_storeMask, kMaxPendingStores); slot >= 0)
    {
        PendingStore &store = m_storePipeline[slot];
        m_storeMask |= 1u << slot;
        {
            store.valid = true;
            store.readyCycle = m_cycle + 1u;
            wheelPush(WheelStore, static_cast<uint32_t>(slot), store.readyCycle);
            store.address = address;
            store.laneMask = laneMask;
            std::copy(words, words + 4, store.words.begin());
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFCu);
}

void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    if (const int slot = firstFreeSlot(m_vfWriteMask, kMaxPendingVfWrites); slot >= 0)
    {
        PendingVfWrite &write = m_vfWritePipeline[slot];
        m_vfWriteMask |= 1u << slot;
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            wheelPush(WheelVf, static_cast<uint32_t>(slot), write.readyCycle);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_vfLatestWrite[reg][component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF7u);
}

void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    if (const int slot = firstFreeSlot(m_viWriteMask, kMaxPendingViWrites); slot >= 0)
    {
        PendingViWrite &write = m_viWritePipeline[slot];
        m_viWriteMask |= 1u << slot;
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            wheelPush(WheelVi, static_cast<uint32_t>(slot), write.readyCycle);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.value = value;
            m_viLatestWrite[reg] = write.sequence;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF6u);
}

void VU1Interpreter::queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency)
{
    if (laneMask == 0u)
        return;
    if (const int slot = firstFreeSlot(m_accWriteMask, kMaxPendingAccWrites); slot >= 0)
    {
        PendingAccWrite &write = m_accWritePipeline[slot];
        m_accWriteMask |= 1u << slot;
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            wheelPush(WheelAcc, static_cast<uint32_t>(slot), write.readyCycle);
            write.sequence = ++m_nextWriteSequence;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_accLatestWrite[component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFF5u);
}

void VU1Interpreter::commitReadyPipelines()
{
    // Called every simulated cycle. Flags commit before Q and P (all three write STATUS), as with a
    // scan of every live entry; flag entries due together were issued in the same cycle and commit
    // in issue order (fixed latency). Stores, VF, VI and ACC touch disjoint state, and VF/VI/ACC
    // writes resolve by sequence number, so their order does not matter. Committed entries are only
    // marked invalid: queueing rewrites every field.
    WheelBucket &bucket = m_wheel[m_cycle & (kWheelSize - 1u)];
    const bool scalarsPending = m_fdiv.valid || m_efu[0].valid || m_efu[1].valid;
    FlagSlot *flags = nullptr;
    if (m_cycle >= kFmacLatency)
    {
        FlagSlot &slot = m_flagRing[(m_cycle - kFmacLatency) & (kFlagRingSize - 1u)];
        if (slot.issueCycle == m_cycle - kFmacLatency && slot.count != 0u)
            flags = &slot;
    }
    if (bucket.count == 0u && !scalarsPending && !flags)
        return;

    if (flags)
    {
        for (uint32_t index = 0; index < flags->count; ++index)
        {
            FlagPipelineEntry &entry = flags->entries[index];
            if (entry.writesMac)
                m_state.mac = entry.mac;
            if (entry.writesStatus)
            {
                const uint32_t current = entry.status & 0xFu;
                m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
            }
            if (entry.writesSticky)
            {
                m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
            }
            if (entry.writesClip)
                m_state.clip = entry.clip;
            entry.valid = false;
        }
        flags->count = 0u;
    }

    if (scalarsPending)
    {
        if (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle)
        {
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv = {};
        }

        for (ScalarPipelineEntry &entry : m_efu)
        {
            if (entry.valid && entry.readyCycle <= m_cycle)
            {
                m_state.p = entry.value;
                entry = {};
            }
        }
    }

    if (bucket.count != 0u)
    {
        for (uint32_t index = 0; index < bucket.count; ++index)
        {
            const uint32_t slot = bucket.items[index] & 31u;
            switch (bucket.items[index] >> 5)
            {
            case WheelStore:
            {
                PendingStore &store = m_storePipeline[slot];
                if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
                {
                    uint32_t oldWords[4]{};
                    std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
                    for (uint32_t component = 0; component < 4u; ++component)
                    {
                        if ((store.laneMask & laneForComponent(component)) != 0u)
                            oldWords[component] = store.words[component];
                    }
                    std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
                }
                store.valid = false;
                m_storeMask &= ~(1u << slot);
                break;
            }
            case WheelVf:
            {
                PendingVfWrite &write = m_vfWritePipeline[slot];
                for (uint32_t component = 0; component < 4u; ++component)
                {
                    if ((write.laneMask & laneForComponent(component)) != 0u &&
                        m_vfLatestWrite[write.reg][component] == write.sequence)
                    {
                        m_state.vf[write.reg][component] = write.value[component];
                    }
                }
                write.valid = false;
                m_vfWriteMask &= ~(1u << slot);
                break;
            }
            case WheelVi:
            {
                PendingViWrite &write = m_viWritePipeline[slot];
                if (m_viLatestWrite[write.reg] == write.sequence)
                    m_state.vi[write.reg] = static_cast<int16_t>(write.value);
                write.valid = false;
                m_viWriteMask &= ~(1u << slot);
                break;
            }
            case WheelAcc:
            {
                PendingAccWrite &write = m_accWritePipeline[slot];
                for (uint32_t component = 0; component < 4u; ++component)
                {
                    if ((write.laneMask & laneForComponent(component)) != 0u &&
                        m_accLatestWrite[component] == write.sequence)
                    {
                        m_state.acc[component] = write.value[component];
                    }
                }
                write.valid = false;
                m_accWriteMask &= ~(1u << slot);
                break;
            }
            default:
                break;
            }
        }
    }
    bucket.count = 0u;
}

void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            static const bool dump = std::getenv("PS2X_VU_RESERVED_DUMP") != nullptr;
            if (dump)
                std::fprintf(stderr, "[xgkick-overflow] src=0x%x copied=0x%x (no EOP within buffer)\n",
                             m_xgkick.sourceAddress, m_xgkick.copiedBytes);
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        // One qword from VU memory, wrapping at its end (a straight copy unless it straddles it).
        const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes) % m_activeVuDataSize;
        const uint32_t firstPart = std::min(16u, m_activeVuDataSize - source);
        std::memcpy(m_xgkick.packet.data() + qwordOffset, m_activeVuData + source, firstPart);
        for (uint32_t i = firstPart; i < 16u; ++i)
            m_xgkick.packet[qwordOffset + i] = m_activeVuData[(source + i) % m_activeVuDataSize];
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else // 2 = IMAGE; 3 ("disable") behaves as IMAGE on hardware
                tagBytes += static_cast<uint64_t>(nloop) * 16u;

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                static const bool dump = std::getenv("PS2X_VU_RESERVED_DUMP") != nullptr;
                if (dump)
                {
                    uint64_t tagHi = 0;
                    std::memcpy(&tagHi, m_xgkick.packet.data() + qwordOffset + 8u, sizeof(tagHi));
                    std::fprintf(stderr,
                                 "[xgkick-overflow] src=0x%x offset=0x%x tag=%016llx:%016llx nloop=%u flg=%u nreg=%u bytes=0x%llx\n",
                                 m_xgkick.sourceAddress, qwordOffset, static_cast<unsigned long long>(tagHi),
                                 static_cast<unsigned long long>(tagLo), nloop, format, nreg,
                                 static_cast<unsigned long long>(tagBytes));
                    std::fprintf(stderr, "  top=0x%x itop=0x%x vi:", m_state.top, m_state.itop);
                    for (uint32_t r = 0; r < 16u; ++r)
                        std::fprintf(stderr, " %u=0x%x", r, static_cast<uint16_t>(m_state.vi[r]));
                    std::fprintf(stderr, "\n  recent VIF1 unpacks:\n");
                    ps2xDumpRecentVif1Unpacks();
                }
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes, 1u);
    m_xgkick.active = false;
}

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    m_xgkick.resetState();
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}

void VU1Interpreter::advanceOneCycle()
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

void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
        advanceOneCycle();
}

bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    return (m_storeMask | m_vfWriteMask | m_viWriteMask | m_accWriteMask) != 0u || m_cycle < m_flagPendingUntil ||
           m_cycle < m_vfPendingUntil || m_cycle < m_accPendingUntil;
}

void VU1Interpreter::flushPipelines()
{
    while (pipelinesPending())
        advanceOneCycle();
}

uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
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

void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
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

VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper) const
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower) const
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
    case 0x7E:
        // EFU (PCSX2 LowerOP_T3_xx): ESADD 70, ERSADD 71, ELENG 72, ERLENG 73, EATANxy 74, EATANxz 75, ESUM 76, ESQRT 78,
        // ERSQRT 79, ERCPR 7A, ESIN 7C, EATAN 7D, EEXP 7E (77 is invalid)
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x79:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7D:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x7C:
            usage.latency = 29u;
            break;
        case 0x7E:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    decoded.upperHandler = decoded.upperUsage.reserved ? nullptr : fmacHandlerFor(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }

    for (const InstructionUsage *usage : {&decoded.upperUsage, &decoded.lowerUsage})
    {
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            for (uint32_t component = 0; component < 4u; ++component)
                if ((access.lanes & laneForComponent(component)) != 0u && decoded.vfReadLaneCount < 16u)
                    decoded.vfReadLanes[decoded.vfReadLaneCount++] = static_cast<uint8_t>(access.reg * 4u + component);
        }
        for (uint32_t reg = 1; reg < 16u; ++reg)
            if ((usage->viRead & (1u << reg)) != 0u && decoded.viReadCount < 15u)
                decoded.viReads[decoded.viReadCount++] = static_cast<uint8_t>(reg);
        for (uint32_t component = 0; component < 4u; ++component)
            if ((usage->accRead & laneForComponent(component)) != 0u && decoded.accReadCount < 4u)
                decoded.accReads[decoded.accReadCount++] = static_cast<uint8_t>(component);
    }

    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg)
    {
        decoded.lowerVfWriteLatency = decoded.lowerUsage.vfLatency != 0u ? decoded.lowerUsage.vfLatency
                                                                         : decoded.lowerUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
            if ((lowerWrite.lanes & laneForComponent(component)) != 0u)
                decoded.lowerVfWriteLanes[decoded.lowerVfWriteCount++] = static_cast<uint8_t>(lowerWrite.reg * 4u + component);
    }
    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        decoded.upperVfWriteLatency = decoded.upperUsage.vfLatency != 0u ? decoded.upperUsage.vfLatency
                                                                         : decoded.upperUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
            if ((upperWrite.lanes & laneForComponent(component)) != 0u)
                decoded.upperVfWriteLanes[decoded.upperVfWriteCount++] = static_cast<uint8_t>(upperWrite.reg * 4u + component);
    }
    decoded.viWriteLatency = decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency;
    for (uint32_t reg = 1; reg < 16u; ++reg)
        if ((decoded.lowerUsage.viWrite & (1u << reg)) != 0u)
            decoded.viWrites[decoded.viWriteCount++] = static_cast<uint8_t>(reg);
    for (uint32_t component = 0; component < 4u; ++component)
        if ((decoded.upperUsage.accWrite & laneForComponent(component)) != 0u)
            decoded.accWrites[decoded.accWriteCount++] = static_cast<uint8_t>(component);
    return decoded;
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
    for (uint32_t i = 0; i < pairCount; ++i)
        m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
}

const VU1Interpreter::DecodedInstructionPair &VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    if ((pc & 7u) != 0u)
        return m_uncachedDecoded = decodeInstructionPair(vuCode, pc);

    const bool trackedVu1Code = memory != nullptr &&
                                ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                                 (m_unit == Unit::VU0 && vuCode == memory->getVU0Code()));
    if (!trackedVu1Code)
        return m_uncachedDecoded = decodeInstructionPair(vuCode, pc);

    const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration();
    if (!m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }
    const uint32_t pairIndex = pc / 8u;
    if (pairIndex >= kMaxDecodedPairs)
        return m_uncachedDecoded = decodeInstructionPair(vuCode, pc);
    return m_decodedCodeCache[pairIndex];
}

namespace
{
    // Context for reserved-instruction reports: the last MSCAL/MSCALF start PC and the
    // code memory being run (per unit: 0 = VU0, 1 = VU1).
    uint32_t s_lastStartPc[2] = {0u, 0u};
    const uint8_t *s_lastCode[2] = {nullptr, nullptr};
    uint32_t s_lastCodeSize[2] = {0u, 0u};
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    // PS2X_VU_RESERVED_DUMP=1 adds the program start PC and the code words around the fault.
    static const bool dump = std::getenv("PS2X_VU_RESERVED_DUMP") != nullptr;
    if (dump)
    {
        const int unit = m_unit == Unit::VU1 ? 1 : 0;
        std::fprintf(stderr, "[VU%d reserved-ctx] start=0x%x pc=0x%x code:", unit, s_lastStartPc[unit], m_state.pc);
        if (const uint8_t *code = s_lastCode[unit])
        {
            const uint32_t from = m_state.pc >= 0x20u ? m_state.pc - 0x20u : 0u;
            for (uint32_t a = from; a < m_state.pc + 0x20u && a + 8u <= s_lastCodeSize[unit]; a += 8u)
            {
                uint32_t lo = 0u, up = 0u;
                std::memcpy(&lo, code + a, 4);
                std::memcpy(&up, code + a + 4, 4);
                std::fprintf(stderr, " %s%x:%08x/%08x", a == m_state.pc ? "*" : "", a, up, lo);
            }
        }
        std::fprintf(stderr, "\n");
    }

    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    resetScheduler();
    m_state.pc = startPC & microAddressMask();
    s_lastStartPc[m_unit == Unit::VU1 ? 1 : 0] = m_state.pc;
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    s_lastCode[m_unit == Unit::VU1 ? 1 : 0] = vuCode;
    s_lastCodeSize[m_unit == Unit::VU1 ? 1 : 0] = codeSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const int previousRoundingMode = std::fegetround();
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    const uint64_t budgetEnd = m_cycle + maxCycles;
    bool programEnded = false;
    // Each later iteration starts right after advanceOneCycle(), which already committed the
    // current cycle, so only the first one needs this.
    commitReadyPipelines();
    // Micro memory cannot change while the program runs: validate the decode cache once.
    (void)getDecodedInstructionPairForPc(vuCode, codeSize, memory, 0u);
    const bool trackedCode = memory != nullptr && m_decodedCodeCacheValid && m_cachedVuCode == vuCode &&
                             m_cachedMemory == memory && m_cachedCodeSize == codeSize &&
                             vuCode == (m_unit == Unit::VU1 ? memory->getVU1Code() : memory->getVU0Code());
    const uint32_t cachedPairs = trackedCode ? std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs) : 0u;
    while (m_cycle < budgetEnd && !m_stopRequested)
    {
        if (m_state.pc + 8u > codeSize)
            break;

        const uint32_t pairIndex = m_state.pc >> 3;
        const DecodedInstructionPair &decoded =
            ((m_state.pc & 7u) == 0u && pairIndex < cachedPairs)
                ? m_decodedCodeCache[pairIndex]
                : getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
        if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
        {
            reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
            break;
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
            break;

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
            programEnded = true;
        }
        else if (m_state.ebit)
            programEnded = true;
        else if (haltBit && !haltBranch)
        {
            m_state.stoppedByD = dHalt;
            m_state.stoppedByT = tHalt;
            programEnded = true;
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
        if (programEnded)
            break;
    }

    if (programEnded)
    {
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    m_state.cycles = m_cycle;
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
}
