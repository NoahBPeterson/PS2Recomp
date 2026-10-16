#ifndef PS2_VU1_DETAIL_H
#define PS2_VU1_DETAIL_H

#include <cstdint>

// Instruction field extraction helpers
static constexpr inline uint8_t DEST(uint32_t i) { return (uint8_t)((i >> 21) & 0xF); }
static constexpr inline uint8_t FT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static constexpr inline uint8_t FS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static constexpr inline uint8_t FD(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static constexpr inline uint8_t BC(uint32_t i) { return (uint8_t)(i & 0x3); }

// Lower instruction field helpers
static constexpr inline uint8_t LIT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static constexpr inline uint8_t LIS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static constexpr inline uint8_t LID(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static constexpr inline uint8_t VIT(uint32_t i) { return (uint8_t)((i >> 16) & 0xF); }
static constexpr inline uint8_t VIS(uint32_t i) { return (uint8_t)((i >> 11) & 0xF); }
static constexpr inline uint8_t VID(uint32_t i) { return (uint8_t)((i >> 6) & 0xF); }
static constexpr inline int16_t IMM11(uint32_t i) { return (int16_t)(int32_t)((int32_t)(i << 21) >> 21); }
static constexpr inline int16_t IMM15(uint32_t i)
{
    uint32_t lo11 = i & 0x7FF;
    uint32_t hi4 = (i >> 21) & 0xF;
    uint32_t raw = (hi4 << 11) | lo11;
    return (int16_t)(int32_t)((int32_t)(raw << 17) >> 17);
}

// Hot VU1Interpreter helpers shared by ps2_vu1_core/upper/lower.cpp, defined here so every VU source
// can inline them.
#include "runtime/ps2_vu1.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

static constexpr uint8_t laneForComponent(uint32_t component)
{
    return static_cast<uint8_t>(1u << (3u - component));
}

// Lowest free slot of a pipeline array of `size` entries, or -1 when full.
static inline int firstFreeSlot(uint32_t mask, uint32_t size)
{
    const uint32_t freeSlots = ~mask & ((size >= 32u) ? 0xFFFFFFFFu : ((1u << size) - 1u));
    return freeSlots ? __builtin_ctz(freeSlots) : -1;
}

inline float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

inline uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

inline void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

inline void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

inline uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  double exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const double magnitude = std::fabs(exactResult);
    const double maximum = static_cast<double>(std::numeric_limits<float>::max());
    const double minimum = static_cast<double>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

inline void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    FlagPipelineEntry *entry = allocFlagEntry(true, 0xFFFFFFFFu);
    if (!entry)
        return;
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

inline void VU1Interpreter::wheelPush(WheelKind kind, uint32_t slot, uint64_t readyCycle)
{
    // An entry due now or earlier is first seen by the commit of the next cycle (the one after
    // this instruction), exactly as with a scan of every live entry.
    const uint64_t commitCycle = std::max(readyCycle, m_cycle + 1u);
    if (commitCycle - m_cycle >= kWheelSize)
    {
        reportReservedInstruction(false, 0xFFFFFFF4u);
        return;
    }
    WheelBucket &bucket = m_wheel[commitCycle & (kWheelSize - 1u)];
    bucket.items[bucket.count++] = static_cast<uint8_t>((static_cast<uint32_t>(kind) << 5) | slot);
}

inline VU1Interpreter::FlagPipelineEntry *VU1Interpreter::allocFlagEntry(bool upper, uint32_t overflowCode)
{
    FlagSlot &slot = m_flagRing[m_cycle & (kFlagRingSize - 1u)];
    if (slot.issueCycle != m_cycle)
    {
        slot.issueCycle = m_cycle; // its previous entries committed kFlagRingSize - kFmacLatency cycles ago
        slot.count = 0u;
    }
    if (slot.count >= slot.entries.size())
    {
        reportReservedInstruction(upper, overflowCode);
        return nullptr;
    }
    FlagPipelineEntry *entry = &slot.entries[slot.count++];
    *entry = {};
    entry->valid = true;
    entry->issueCycle = m_cycle;
    entry->readyCycle = m_cycle + kFmacLatency;
    m_flagPendingUntil = std::max(m_flagPendingUntil, entry->readyCycle);
    return entry;
}

inline void VU1Interpreter::commitFlagSlot(FlagSlot &slot)
{
    for (uint32_t index = 0; index < slot.count; ++index)
    {
        FlagPipelineEntry &entry = slot.entries[index];
        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry.valid = false;
    }
    slot.count = 0u;
}

inline void VU1Interpreter::commitScalars()
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

inline bool VU1Interpreter::commitDue() const
{
    // Q and P commit at their ready cycle; until then there is nothing to do for them.
    if (m_wheel[m_cycle & (kWheelSize - 1u)].count != 0u || (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle) ||
        (m_efu[0].valid && m_efu[0].readyCycle <= m_cycle) || (m_efu[1].valid && m_efu[1].readyCycle <= m_cycle))
        return true;
    if (m_cycle < kFmacLatency)
        return false;
    const FlagSlot &slot = m_flagRing[(m_cycle - kFmacLatency) & (kFlagRingSize - 1u)];
    return slot.issueCycle == m_cycle - kFmacLatency && slot.count != 0u;
}

inline VU1Interpreter::FlagSlot *VU1Interpreter::currentFlagSlot()
{
    FlagSlot &slot = m_flagRing[m_cycle & (kFlagRingSize - 1u)];
    return slot.issueCycle == m_cycle ? &slot : nullptr;
}

#endif
