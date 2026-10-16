#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"
#include "ps2_vu1_decode.h"
#include "ps2_vu1_issue.h"

#include <algorithm>
#include <vector>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <ps2_log.h>

void ps2xDumpRecentVif1Unpacks(); // ps2_vif1_interpreter.cpp



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
    m_vfPendingUntil = m_accPendingUntil = m_viPendingUntil = 0u;
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
    if (m_storesImmediate)
    {
        // Translated block: a queued store would land at the next cycle boundary, before anything
        // else reads VU memory (the lower slot is the only memory op, and the block lets an XGKICK
        // in flight read up to this cycle first), so land it now.
        if (m_activeVuData && address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + address, sizeof(oldWords));
            for (uint32_t component = 0; component < 4u; ++component)
                if ((laneMask & laneForComponent(component)) != 0u)
                    oldWords[component] = words[component];
            std::memcpy(m_activeVuData + address, oldWords, sizeof(oldWords));
        }
        return;
    }
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
        commitFlagSlot(*flags);

    if (scalarsPending)
        commitScalars();

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

void VU1Interpreter::progressXgkick(uint32_t cycles)
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    // Credit left over when the transfer ends is dropped (startXgkick resets it), so several
    // boundaries at once move it exactly as one at a time.
    m_xgkick.cycleCredit += cycles;
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

bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    return (m_storeMask | m_vfWriteMask | m_viWriteMask | m_accWriteMask) != 0u || m_cycle < m_flagPendingUntil ||
           m_cycle < m_vfPendingUntil || m_cycle < m_accPendingUntil || m_cycle < m_viPendingUntil;
}

void VU1Interpreter::flushPipelines()
{
    while (pipelinesPending())
        advanceOneCycle();
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    uint32_t lower = 0u, upper = 0u;
    std::memcpy(&lower, vuCode + pc, sizeof(lower));
    std::memcpy(&upper, vuCode + pc + sizeof(lower), sizeof(upper));
    return decodePair(lower, upper, m_unit);
}

namespace
{
    // Registered blocks, all of them and per pc / 8 (longest first: where blocks of different
    // lengths match, the longest runs).
    struct CompiledBlockRegistry
    {
        std::vector<VU1Interpreter::CompiledBlockInfo> all;
        std::array<std::vector<VU1Interpreter::CompiledBlockInfo>, 0x4000u / 8u> byPc;
    };
    CompiledBlockRegistry &compiledRegistry()
    {
        static CompiledBlockRegistry registry;
        return registry;
    }
}

void ps2xVu1AotRegister(); // generated (ps2xRuntime/src/vu1_aot) or the empty stub

namespace
{
    void ps2xVu1AotRegisterOnce()
    {
        static const bool registered = []
        {
            ps2xVu1AotRegister();
            return true;
        }();
        (void)registered;
    }
}

void (*VU1Interpreter::s_pcHook)(const uint8_t *, uint32_t) = nullptr;

void VU1Interpreter::registerCompiledBlocks(const CompiledBlockInfo *blocks, uint32_t count)
{
    CompiledBlockRegistry &registry = compiledRegistry();
    for (uint32_t i = 0; i < count; ++i)
    {
        const CompiledBlockInfo &info = blocks[i];
        if (info.pc >= 0x4000u || (info.pc & 7u) != 0u || info.size == 0u || info.pc + info.size > 0x4000u)
            continue;
        registry.all.push_back(info);
        auto &bucket = registry.byPc[info.pc >> 3];
        bucket.push_back(info);
        std::stable_sort(bucket.begin(), bucket.end(), [](const CompiledBlockInfo &a, const CompiledBlockInfo &b)
                         { return a.size > b.size; });
    }
}

const std::vector<VU1Interpreter::CompiledBlockInfo> &VU1Interpreter::compiledBlockList()
{
    ps2xVu1AotRegisterOnce();
    return compiledRegistry().all;
}

void VU1Interpreter::resolveCompiledBlock(const uint8_t *vuCode, uint32_t index)
{
    m_compiledResolved[index] = true;
    m_compiledBlocks[index] = nullptr;
    const uint32_t pc = index << 3;
    for (const CompiledBlockInfo &info : compiledRegistry().byPc[index])
        if (std::memcmp(vuCode + pc, info.code, info.size) == 0)
        {
            m_compiledBlocks[index] = info.block;
            return;
        }
}

const VU1Interpreter::CompiledBlock *VU1Interpreter::compiledBlocksFor(const uint8_t *vuCode, uint32_t codeSize,
                                                                       const PS2Memory *memory)
{
    ps2xVu1AotRegisterOnce();
    static const bool disabled = std::getenv("PS2X_VU1_NO_AOT") != nullptr;
    if (disabled || !m_compiledEnabled || m_unit != Unit::VU1 || !memory || codeSize != 0x4000u ||
        compiledRegistry().all.empty())
        return nullptr;
    const uint64_t generation = memory->getVU1CodeGeneration();
    if (generation != m_compiledGeneration || vuCode != m_compiledCode)
    {
        // Micro memory changed (MPG upload): every pc is looked up again when reached.
        m_compiledGeneration = generation;
        m_compiledCode = vuCode;
        m_compiledResolved.fill(false);
    }
    return m_compiledBlocks.data();
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
    const CompiledBlock *compiled = trackedCode ? compiledBlocksFor(vuCode, codeSize, memory) : nullptr;
    const BlockArgs blockArgs{budgetEnd, vuData, dataSize, &gs, memory, codeSize};
    while (m_cycle < budgetEnd && !m_stopRequested)
    {
        if (m_state.pc + 8u > codeSize)
            break;

        if (compiled && (m_state.pc & 7u) == 0u)
        {
            const uint32_t index = m_state.pc >> 3;
            if (!m_compiledResolved[index])
                resolveCompiledBlock(vuCode, index);
            if (const CompiledBlock block = compiled[index])
            {
                const IssueResult issued = (this->*block)(blockArgs);
                if (issued == IssueStopped)
                    break;
                if (issued == IssueEnded)
                {
                    programEnded = true;
                    break;
                }
                if (issued != IssueDeclined)
                    continue;
                // Declined (its entry conditions do not hold): interpret one pair.
            }
        }

        if (s_pcHook)
            s_pcHook(vuCode, m_state.pc);
        ++m_interpretedPairs;
        const uint32_t pairIndex = m_state.pc >> 3;
        const DecodedInstructionPair &decoded =
            ((m_state.pc & 7u) == 0u && pairIndex < cachedPairs)
                ? m_decodedCodeCache[pairIndex]
                : getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
        const IssueResult issued = issuePair(decoded, budgetEnd, vuData, dataSize, gs, memory, codeSize);
        if (issued == IssueStopped)
            break;
        if (issued == IssueEnded)
        {
            programEnded = true;
            break;
        }
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
