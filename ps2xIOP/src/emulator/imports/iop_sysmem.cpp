#include "iop_sysmem.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <string>

namespace ps2x::iop::detail
{
    IopSysmem::IopSysmem(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    bool IopSysmem::dispatchImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const uint32_t a2 = cpu.gpr[6];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };

        switch (ordinal)
        {
        case 4: // AllocSysMemory(mode, size, address): ALLOC_FIRST 0, ALLOC_LAST 1, ALLOC_ADDRESS 2
        {
            using Placement = IopMemory::BlockPlacement;
            const Placement placement = a0 == 2u ? Placement::At : (a0 == 1u ? Placement::High : Placement::Low);
            setV0(m_memory.allocateSysmemBlock(a1, placement, a2));
            return true;
        }
        case 5: // FreeSysMemory
            setV0(m_memory.freeAllocation(a0) ? 0u : 0xFFFFFFFFu);
            return true;
        case 6: // QueryMemSize
            setV0(IopMemory::RamSize);
            return true;
        case 7: // QueryMaxFreeMemSize: the largest free block
            setV0(m_memory.freeBlockStats().largest);
            return true;
        case 8: // QueryTotalFreeMemSize: all free blocks together
            setV0(m_memory.freeBlockStats().total);
            return true;
        case 9:  // QueryBlockTopAddress
        case 10: // QueryBlockSize
        {
            // The block holding the address: a used one, or the free gap around it with bit 31 (FREE)
            // set, as SYSMEM's block list covers all memory; -1 outside it.
            if (const auto block = m_memory.allocationContaining(a0))
                setV0(ordinal == 9u ? block->address : block->size);
            else if (const auto gap = m_memory.freeGapContaining(a0))
                setV0((ordinal == 9u ? gap->address : gap->size) | 0x80000000u);
            else
                setV0(0xFFFFFFFFu);
            return true;
        }
        case 14: // Kprintf
        {
            const std::string format = m_memory.readString(a0, 512u);
            m_host.log(LogLevel::Info, std::string("[IOP Kprintf] ") + format);
            // SYSMEM's Kprintf returns what the handler installed with KprintfSet returns; a retail
            // boot installs none, and then it returns 0.
            setV0(0u);
            return true;
        }
        case 15:
            setV0(0u);
            return true;
        default:
            return false;
        }
    }
}
