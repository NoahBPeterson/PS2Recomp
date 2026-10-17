#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace GSMem
{
    class TexturePageCache
    {
    public:
        static constexpr uint32_t kPageSize = 8192u;

        void Invalidate() noexcept
        {
            m_pageBase = UINT32_MAX;
        }

        // Also forgets the spare copy: for when VRAM changes without its writes being stamped.
        void InvalidateAll() noexcept
        {
            m_pageBase = UINT32_MAX;
            for (Slot &slot : m_slots)
                slot.base = UINT32_MAX;
        }

        // Write tracking, optional. pageStamps[page] is the stamp of the last write to that 8 KiB
        // page and *writeStamp the latest stamp handed out. With it, a miss on a page held in the
        // spare slot reuses that copy if the page has not been written since the copy was made:
        // the copy then equals VRAM, so the result is the same as copying again.
        void SetWriteTracking(const uint64_t *pageStamps, const uint64_t *writeStamp) noexcept
        {
            m_pageStamps = pageStamps;
            m_writeStamp = writeStamp;
        }

        // Pages written but not stamped yet (a 512-bit set, e.g. the primitive being drawn), or null.
        void SetPendingWrites(const uint64_t *pages) noexcept
        {
            m_pending = pages;
        }

        // byteAddress is the wrapped, swizzled VRAM address. The returned
        // pointer is valid only until the next miss or invalidation.
        const uint8_t* Resolve(const uint8_t* vram, uint32_t byteAddress) noexcept
        {
            const uint32_t pageBase = byteAddress & ~(kPageSize - 1u);
            if (m_pageBase != pageBase)
                Load(vram, pageBase);
            return m_slots[m_current].bytes.data() + (byteAddress & (kPageSize - 1u));
        }

        // Byte address of the page currently held, or UINT32_MAX when empty.
        uint32_t PageBase() const noexcept
        {
            return m_pageBase;
        }

    private:
        struct Slot
        {
            alignas(64) std::array<uint8_t, kPageSize> bytes{};
            uint32_t base = UINT32_MAX;
            uint64_t stamp = 0u; // *m_writeStamp when copied
        };

        void Load(const uint8_t *vram, uint32_t pageBase) noexcept
        {
            m_pageBase = pageBase;
            for (uint32_t i : {m_current, m_current ^ 1u}) // the current slot matches after an Invalidate
                if (Unchanged(m_slots[i], pageBase))
                {
                    m_current = i;
                    return;
                }
            m_current ^= 1u;
            Slot &slot = m_slots[m_current];
            std::memcpy(slot.bytes.data(), vram + pageBase, kPageSize);
            slot.base = pageBase;
            slot.stamp = m_writeStamp ? *m_writeStamp : 0u;
        }

        bool Unchanged(const Slot &slot, uint32_t pageBase) const noexcept
        {
            const uint32_t page = pageBase / kPageSize;
            return m_pageStamps && slot.base == pageBase && m_pageStamps[page] <= slot.stamp &&
                   !(m_pending && ((m_pending[page >> 6u] >> (page & 63u)) & 1u));
        }

        std::array<Slot, 2> m_slots{};
        uint32_t m_current = 0u;
        uint32_t m_pageBase = UINT32_MAX;
        const uint64_t *m_pageStamps = nullptr;
        const uint64_t *m_writeStamp = nullptr;
        const uint64_t *m_pending = nullptr;
    };
}
