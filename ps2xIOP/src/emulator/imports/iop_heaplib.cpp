// HEAPLIB: the BIOS heap allocator, with its heaps, chunks and fragment lists kept in guest memory
// exactly as the firmware lays them out (heap blocks come from sysmem, like AllocSysMemory).
// Port of the SCE algorithm as documented by ps2sdk iop/system/heaplib (a reconstruction of the
// SDK 1.3.4 module), checked byte-for-byte against the BIOS HEAPLIB module by ps2_iop_ground_truth.
//
// Layouts (32-bit words, 8-byte blocks):
//   heap   +0 key (heap+1)   +4 size-to-grow | high-memory flag   +8 list next   +12 list prev
//          +16 first chunk
//   chunk  +0 key (chunk-1)  +4 byte size   +8 used blocks   +12 rover fragment   +16 fragment
//   fragment +0 next  +4 size in blocks  +8 payload
//   extra sysmem block: +0 list next  +4 list prev  +8 chunk
#include "iop_heaplib.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"

namespace ps2x::iop::detail
{
    IopHeaplib::IopHeaplib(IopMemory &memory) noexcept
        : m_memory(memory)
    {
    }

    namespace
    {
        struct Heap
        {
            IopMemory &m;
            uint32_t r(uint32_t a) const { return m.read32(a); }
            void w(uint32_t a, uint32_t v) { m.write32(a, v); }

            // sysmem as the HLE serves it (iop_sysmem.cpp)
            uint32_t allocSys(uint32_t size) { return m.allocate(size, 16u); }
            void freeSys(uint32_t a) { (void)m.freeAllocation(a); }
            uint32_t blockSize(uint32_t a)
            {
                const auto block = m.allocationContaining(a);
                return block ? block->size : 0xFFFFFFFFu;
            }

            void prepare(uint32_t mem, uint32_t size)
            {
                if (mem == 0u || size < 0x29u)
                    return;
                w(mem, mem - 1u);
                const uint32_t calc = (size - 16u) >> 3;
                const uint32_t item = mem + 8u + 8u * calc;
                w(mem + 4u, size);
                w(mem + 12u, mem + 16u);
                w(mem + 8u, 0u);
                w(mem + 20u, calc - 1u);
                w(mem + 16u, item);
                w(item + 4u, 0u);
                w(r(mem + 16u), mem + 16u);
            }

            bool chunkKeyOk(uint32_t chunk) const { return chunk != 0u && r(chunk) == chunk - 1u; }

            uint32_t chunkAllocate(uint32_t chunk, uint32_t nbytes)
            {
                if (!chunkKeyOk(chunk))
                    return 0u;
                const uint32_t calc1 = nbytes < 8u ? 15u : nbytes + 7u;
                const uint32_t want = (calc1 >> 3) + 1u;
                const uint32_t rover = r(chunk + 12u);
                uint32_t prev = rover;
                uint32_t frag = r(prev);
                uint32_t blocks = 0;
                for (;; frag = r(frag))
                {
                    blocks = r(frag + 4u);
                    if (static_cast<int32_t>(blocks) >= static_cast<int32_t>(want))
                        break;
                    prev = frag;
                    if (frag == rover)
                        return 0u;
                }
                if (blocks == want)
                    w(prev, r(frag));
                else
                {
                    const uint32_t rest = blocks - want;
                    w(frag + 4u, rest);
                    frag += 8u * rest;
                    w(frag + 4u, want);
                }
                w(chunk + 12u, prev);
                w(frag, chunk);
                w(chunk + 8u, r(chunk + 8u) + r(frag + 4u));
                return frag + 8u;
            }

            int32_t chunkFree(uint32_t chunk, uint32_t ptr)
            {
                if (!chunkKeyOk(chunk))
                    return -4;
                if (ptr == 0u)
                    return -1;
                const uint32_t hdr = ptr - 8u;
                if (r(ptr - 8u) != chunk || static_cast<int32_t>(r(hdr + 4u)) <= 0)
                    return -1;
                if (hdr < chunk + 16u || hdr >= chunk + r(chunk + 4u))
                    return -2;
                const auto end = [&](uint32_t f) { return f + 8u * r(f + 4u); };
                uint32_t f1 = r(chunk + 12u);
                if (f1 >= hdr || hdr >= r(f1))
                {
                    for (;;)
                    {
                        if (hdr == f1)
                            return -3;
                        const uint32_t f2 = r(f1);
                        if (f1 < r(f1))
                            goto lbl18;
                        if (hdr < r(f1))
                            goto lbl20;
                        if (f1 >= hdr)
                        {
                        lbl18:
                            f1 = r(f1);
                            if (f2 >= hdr || hdr >= r(f2))
                                continue;
                        }
                        if (hdr >= r(f1))
                            goto lbl21;
                        goto lbl20;
                    }
                }
            lbl20:
                if (r(f1) < end(hdr))
                    return -3;
            lbl21:
                if (f1 < hdr && hdr < end(f1))
                    return -3;
                w(chunk + 8u, r(chunk + 8u) - r(hdr + 4u));
                {
                    const uint32_t next = end(hdr);
                    uint32_t nextBlocks = 0;
                    if (next != r(f1) || (nextBlocks = r(next + 4u), static_cast<int32_t>(nextBlocks) <= 0))
                        w(hdr, r(f1));
                    else
                    {
                        w(hdr + 4u, r(hdr + 4u) + nextBlocks);
                        w(hdr, r(r(f1)));
                    }
                }
                if (hdr == end(f1))
                {
                    w(f1 + 4u, r(f1 + 4u) + r(hdr + 4u));
                    w(f1, r(hdr));
                }
                else
                    w(f1, hdr);
                w(chunk + 12u, f1);
                return 0;
            }

            uint32_t chunkIsUnused(uint32_t chunk) const { return chunkKeyOk(chunk) && r(chunk + 8u) == 0u ? 1u : 0u; }

            uint32_t chunkFreeBytes(uint32_t chunk) const
            {
                return 8u * (((r(chunk + 4u) - 16u) >> 3) - r(chunk + 8u) - 1u);
            }

            uint32_t create(int32_t size, int32_t flag)
            {
                const int32_t calc = 4 * ((size + 3) >> 2);
                const uint32_t heap = allocSys(static_cast<uint32_t>(calc));
                if (heap == 0u)
                    return 0u;
                w(heap, heap + 1u);
                uint32_t grow = (flag & 1) ? static_cast<uint32_t>(calc) : 0u;
                grow |= (static_cast<uint32_t>(flag) >> 1) & 1u;
                w(heap + 4u, grow);
                w(heap + 8u, heap + 8u);
                w(heap + 12u, heap + 8u);
                prepare(heap + 16u, static_cast<uint32_t>(calc - 16));
                return heap;
            }

            void destroy(uint32_t heap)
            {
                if (r(heap) != heap + 1u)
                    return;
                uint32_t item = r(heap + 8u);
                w(heap, 0u);
                if (item != heap + 8u)
                {
                    uint32_t next;
                    do
                    {
                        next = r(item);
                        w(item + 8u, 0u);
                        freeSys(item);
                        item = next;
                    } while (next != heap + 8u);
                }
                w(heap + 16u, 0u);
                freeSys(heap);
            }

            uint32_t alloc(uint32_t heap, uint32_t nbytes)
            {
                if (r(heap) != heap + 1u)
                    return 0u;
                const uint32_t head = heap + 8u;
                for (uint32_t item = r(heap + 8u);; item = r(item))
                {
                    if (const uint32_t result = chunkAllocate(item + 8u, nbytes))
                        return result;
                    if (item != head)
                        continue;
                    const int32_t grow = static_cast<int32_t>(r(heap + 4u));
                    if (grow < 4)
                        return 0u;
                    uint32_t size = 2u * (static_cast<uint32_t>(grow) >> 1);
                    if (static_cast<int32_t>(size - 40u) < static_cast<int32_t>(nbytes))
                        size = nbytes + 40u;
                    const uint32_t block = allocSys(size);
                    if (block == 0u)
                        return 0u;
                    const uint32_t bytes = blockSize(block);
                    const uint32_t after = r(heap + 8u); // linked_list_add_after(heap->l.next, block)
                    w(block, after);
                    w(block + 4u, r(after + 4u));
                    w(after + 4u, block);
                    w(r(block + 4u), block);
                    prepare(block + 8u, bytes - 8u);
                    return chunkAllocate(block + 8u, nbytes);
                }
            }

            int32_t release(uint32_t heap, uint32_t ptr)
            {
                if (heap == 0u || r(heap) != heap + 1u)
                    return -4;
                const uint32_t head = heap + 8u;
                uint32_t item = r(heap + 8u);
                while (chunkFree(item + 8u, ptr) != 0)
                {
                    if (item == head)
                        return -1;
                    item = r(item);
                }
                if (item != head && chunkIsUnused(item + 8u))
                {
                    w(r(item) + 4u, r(item + 4u)); // linked_list_remove(item)
                    w(r(item + 4u), r(item));
                    w(item + 8u, 0u);
                    freeSys(item);
                }
                return 0;
            }

            int32_t totalFree(uint32_t heap)
            {
                if (heap == 0u || r(heap) != heap + 1u)
                    return -4;
                uint32_t total = 0;
                for (uint32_t item = r(heap + 8u);; item = r(item))
                {
                    total += chunkFreeBytes(item + 8u);
                    if (item == heap + 8u)
                        break;
                }
                return static_cast<int32_t>(total);
            }
        };
    }

    bool IopHeaplib::dispatchImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        Heap h{m_memory};
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };

        switch (ordinal)
        {
        case 4: // CreateHeap(size, flag): flag&1 grow on demand, flag&2 high memory
            setV0(h.create(static_cast<int32_t>(a0), static_cast<int32_t>(a1)));
            return true;
        case 5: // DeleteHeap
            h.destroy(a0);
            return true;
        case 6: // AllocHeapMemory(heap, n)
            setV0(h.alloc(a0, a1));
            return true;
        case 7: // FreeHeapMemory(heap, ptr)
            setV0(static_cast<uint32_t>(h.release(a0, a1)));
            return true;
        case 8: // HeapTotalFreeSize
            setV0(static_cast<uint32_t>(h.totalFree(a0)));
            return true;
        case 9:
        case 10:
        case 16:
        case 17: // _retonly
            return true;
        case 11: // HeapPrepare(mem, size)
            h.prepare(a0, a1);
            return true;
        case 12: // chunk is valid and unused
            setV0(h.chunkIsUnused(a0));
            return true;
        case 13: // chunk allocate
            setV0(h.chunkAllocate(a0, a1));
            return true;
        case 14: // chunk free
            setV0(static_cast<uint32_t>(h.chunkFree(a0, a1)));
            return true;
        case 15: // HeapChunkSize
            setV0(h.chunkFreeBytes(a0));
            return true;
        default:
            return false;
        }
    }
}
