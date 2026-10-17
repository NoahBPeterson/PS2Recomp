#pragma once
// Lazy reading of the part of a VIF1 DMA list that sits behind a VIF1 i-bit stall.
//
// The runtime walks a DMA chain and copies its payload when the channel is started, but on
// hardware the DMAC reads each tag and qword only when the VIF consumes it. A VIF1 list that
// stalls on an i-bit lets the EE run its VIF1 interrupt handler before the rest of the list
// is read, and True Crime: New York City's handler (one i-bit NOP per streamed texture group)
// rewrites the not-yet-transferred part of the list (TEX0/CLUT/FRAME registers of the next
// batches). Without a second read, the rest of the list ran with what that double-buffered
// list held two frames earlier and sampled unrelated VRAM (the "texture noise").
//
// So for VIF1 the runtime records where every byte of a walked chain came from and the walker
// state after each tag. On FBRST.STC it re-reads the rest of the current tag's payload and
// walks the chain again from the next tag (resumeVIF1AfterStall).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

struct Vif1SourceSegment
{
    uint32_t offset = 0u; // byte offset in the buffer
    uint32_t length = 0u;
    uint32_t phys = 0u;   // physical offset in RDRAM or scratchpad
    bool scratchpad = false;
};

// Source-chain walker state (TADR, ASR0/1, CHCR.ASP, last tag's upper half, tags walked).
struct DmaChainCursor
{
    uint32_t tagAddr = 0u;
    uint32_t asr0 = 0u;
    uint32_t asr1 = 0u;
    uint32_t asp = 0u;
    uint32_t lastTagUpper = 0u;
    int tags = 0;
    bool ended = false;
    // MADR as the DMAC leaves it: the end of the last tag's payload (PCSX2 hwDmacSrcChain).
    uint32_t madr = 0u;
    bool madrValid = false;
};

// Walker state after a tag: the bytes of that tag end at endOffset in the buffer.
struct Vif1TagMark
{
    uint32_t endOffset = 0u;
    uint32_t tagAddr = 0u; // next tag
    uint32_t asr0 = 0u;
    uint32_t asr1 = 0u;
    uint32_t asp = 0u;
    bool ended = false;    // the chain ended with this tag
};

struct Vif1ChainSources
{
    uint32_t chcr = 0u;                     // CHCR of the transfer (TTE, TIE)
    std::vector<Vif1SourceSegment> segments; // where each byte came from
    std::vector<Vif1TagMark> marks;          // empty: a normal-mode transfer (no chain to re-walk)
};

// The held-back bytes of a stalled list (PS2Memory::m_vif1StalledData) and how to read them again.
struct Vif1StalledSources
{
    bool valid = false;
    std::vector<Vif1SourceSegment> segments; // rest of the tag that was being consumed
    bool rewalk = false;                     // re-walk the chain from `resume` ...
    Vif1TagMark resume;
    uint32_t chcr = 0u;
    uint32_t tailBegin = 0u; // ... replacing m_vif1StalledData[tailBegin, tailEnd)
    uint32_t tailEnd = 0u;

    void clear() { *this = Vif1StalledSources{}; }
};

struct Vif1SourceTracking
{
    const uint8_t *buffer = nullptr; // buffer processVIF1Data is consuming, described by `current`
    Vif1ChainSources current;
    Vif1StalledSources stalled;
    std::vector<std::pair<const uint8_t *, Vif1ChainSources>> pending; // walked chains not yet consumed

    void reset()
    {
        buffer = nullptr;
        current = Vif1ChainSources{};
        stalled.clear();
        pending.clear();
    }

    // Called when VIF1 stalls at byte `pos` of `data`; the held-back bytes start at heldBase in
    // m_vif1StalledData.
    void noteStall(const uint8_t *data, uint32_t pos, uint32_t size, uint32_t heldBase)
    {
        if (buffer != data || stalled.valid)
            return;
        uint32_t prefixEnd = size;
        const Vif1TagMark *mark = nullptr;
        for (const auto &m : current.marks)
        {
            if (m.endOffset > pos)
            {
                mark = &m;
                prefixEnd = m.endOffset;
                break;
            }
        }
        if (!current.marks.empty() && !mark)
            return; // stalled after the last tag: nothing left to read
        stalled.valid = true;
        for (const auto &s : current.segments)
        {
            const uint32_t begin = s.offset > pos ? s.offset : pos;
            const uint32_t end = (s.offset + s.length) < prefixEnd ? (s.offset + s.length) : prefixEnd;
            if (begin >= end)
                continue;
            stalled.segments.push_back({begin - pos + heldBase, end - begin, s.phys + (begin - s.offset), s.scratchpad});
        }
        if (mark && !mark->ended)
        {
            stalled.rewalk = true;
            stalled.resume = *mark;
            stalled.chcr = current.chcr;
            stalled.tailBegin = prefixEnd - pos + heldBase;
            stalled.tailEnd = size - pos + heldBase;
        }
    }

    // Re-reads the mapped bytes of buf from guest memory; returns how many bytes changed.
    static size_t refresh(std::vector<uint8_t> &buf, const std::vector<Vif1SourceSegment> &segments,
                          const uint8_t *rdram, uint32_t rdramSize, const uint8_t *spr, uint32_t sprSize)
    {
        size_t changed = 0u;
        for (const auto &s : segments)
        {
            const uint8_t *base = s.scratchpad ? spr : rdram;
            const uint32_t limit = s.scratchpad ? sprSize : rdramSize;
            if (!base || s.phys > limit || s.length > limit - s.phys)
                continue;
            if (static_cast<size_t>(s.offset) + s.length > buf.size())
                continue;
            uint8_t *dst = buf.data() + s.offset;
            const uint8_t *src = base + s.phys;
            if (std::memcmp(dst, src, s.length) != 0)
            {
                for (uint32_t i = 0; i < s.length; ++i)
                    changed += dst[i] != src[i];
                std::memcpy(dst, src, s.length);
            }
        }
        return changed;
    }
};

