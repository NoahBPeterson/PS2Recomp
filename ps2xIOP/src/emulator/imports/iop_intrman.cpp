#include "iop_intrman.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "../services/iop_rpc.h"

namespace ps2x::iop::detail
{
    namespace
    {
        // Kernel error codes (ps2sdk kerr.h).
        constexpr int32_t KE_ILLEGAL_INTRCODE = -101;
        constexpr int32_t KE_CPUDI = -102;
        constexpr int32_t KE_INTRDISABLE = -103;
        constexpr int32_t KE_FOUND_HANDLER = -104;
        constexpr int32_t KE_NOTFOUND_HANDLER = -105;

        // Handler slots: the interrupt lines 0-31, the DMA lines 32-45 and the two software
        // interrupts 62/63 (Sony INTRMAN, checked by the IOP ground truth).
        bool validHandlerIrq(int irq) { return (irq >= 0 && irq <= 45) || irq == 62 || irq == 63; }
        // DMA completion lines: DICR channels 0-6 (32-38) and DICR2 channels 7-12 (40-45).
        bool dmaLine(uint32_t line) { return (line >= 32u && line <= 38u) || (line >= 40u && line <= 45u); }
    }

    IopIntrman::IopIntrman(IopMemory &memory) noexcept
        : m_memory(memory)
    {
    }

    void IopIntrman::reset()
    {
        m_handlers.clear();
        m_enabled.clear();
        m_dmaEnabled = m_dmaMode1 = m_dmaMode2 = 0u;
    }

    bool IopIntrman::dispatchImport(uint16_t ordinal, IopCpuState &cpu, IopGuestExecutor &executor)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const uint32_t a2 = cpu.gpr[6];
        const uint32_t a3 = cpu.gpr[7];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };

        switch (ordinal)
        {
        case 3:
            setV0(0u);
            return true;
        case 4: // RegisterIntrHandler(irq, mode, handler, arg)
        {
            const int irq = static_cast<int>(a0);
            if (!validHandlerIrq(irq))
                setV0(static_cast<uint32_t>(KE_ILLEGAL_INTRCODE));
            else if (m_handlers.count(irq) != 0u)
                setV0(static_cast<uint32_t>(KE_FOUND_HANDLER));
            else
            {
                m_handlers[irq] = {a2, a3, cpu.gpr[28]};
                setV0(0u);
            }
            return true;
        }
        case 5: // ReleaseIntrHandler(irq)
        {
            const int irq = static_cast<int>(a0);
            if (!validHandlerIrq(irq))
                setV0(static_cast<uint32_t>(KE_ILLEGAL_INTRCODE));
            else if (m_handlers.erase(irq) == 0u)
                setV0(static_cast<uint32_t>(KE_NOTFOUND_HANDLER));
            else
                setV0(0u);
            return true;
        }
        case 6: // EnableIntr(irq): the low byte is the line; for DMA lines bits 8/9 select DICR/DICR2 completion bits
        {
            const uint32_t line = a0 & 0xFFu;
            int32_t result = 0;
            if (line < 32u)
                m_memory.setInterruptMask(m_memory.interruptMask() | (1u << line));
            else if (dmaLine(line))
            {
                const uint32_t bit = 1u << (line - 32u);
                m_dmaEnabled |= bit;
                m_dmaMode1 = (m_dmaMode1 & ~bit) | ((a0 & 0x100u) != 0u ? bit : 0u);
                m_dmaMode2 = (m_dmaMode2 & ~bit) | ((a0 & 0x200u) != 0u ? bit : 0u);
                m_memory.setInterruptMask(m_memory.interruptMask() | (1u << 3)); // the DMA line
            }
            else
                result = KE_ILLEGAL_INTRCODE;
            if (result == 0)
                m_enabled[static_cast<int>(line)] = true;
            setV0(static_cast<uint32_t>(result));
            return true;
        }
        case 7: // DisableIntr(irq, int *res): *res = the line (plus its DMA mode bits) if it was enabled
        {
            const uint32_t line = a0 & 0xFFu;
            int32_t result = 0;
            int32_t res = KE_INTRDISABLE;
            if (line < 32u)
            {
                const uint32_t mask = m_memory.interruptMask();
                m_memory.setInterruptMask(mask & ~(1u << line));
                if ((mask & (1u << line)) != 0u)
                    res = static_cast<int32_t>(line);
                else
                    result = KE_INTRDISABLE;
            }
            else if (dmaLine(line))
            {
                const uint32_t bit = 1u << (line - 32u);
                if ((m_dmaEnabled & bit) != 0u)
                {
                    // DICR's mode bits exist only for channels 0-6; DICR2 holds 0x200 for every channel.
                    res = static_cast<int32_t>(line | ((line <= 38u && (m_dmaMode1 & bit) != 0u) ? 0x100u : 0u) |
                                               ((m_dmaMode2 & bit) != 0u ? 0x200u : 0u));
                    m_dmaEnabled &= ~bit;
                }
                else
                    result = KE_INTRDISABLE;
            }
            else
                result = KE_ILLEGAL_INTRCODE;
            if (a1 != 0u)
                m_memory.write32(a1, static_cast<uint32_t>(res));
            if (result != KE_ILLEGAL_INTRCODE)
                m_enabled[static_cast<int>(line)] = false;
            setV0(static_cast<uint32_t>(result));
            return true;
        }
        case 8: // CpuDisableIntr: reading I_CTRL clears it
        {
            const uint32_t previous = m_memory.interruptControl();
            m_memory.setInterruptControl(0u);
            setV0(previous != 0u ? 0u : static_cast<uint32_t>(KE_CPUDI));
            return true;
        }
        case 9: // CpuEnableIntr
            m_memory.setInterruptControl(1u);
            setV0(0u);
            return true;
        case 14:
            setV0(a0 != 0u
                      ? executor.executeGuestFunctionWithBudget(a0, a1, a2, a3, 0u, cpu.gpr[28], 100000u)
                      : 0u);
            return true;
        case 15:
        case 16:
        case 23:
        case 24:
        case 25:
        case 28:
        case 30:
            setV0(0u);
            return true;
        case 17: // CpuSuspendIntr(int *state): *state = I_CTRL (the read clears it)
        {
            const uint32_t previous = m_memory.interruptControl();
            if (a0 != 0u)
                m_memory.write32(a0, previous);
            m_memory.setInterruptControl(0u);
            setV0(previous != 0u ? 0u : static_cast<uint32_t>(KE_CPUDI));
            return true;
        }
        case 18: // CpuResumeIntr(state): I_CTRL = state
            m_memory.setInterruptControl(a0);
            setV0(0u);
            return true;
        default:
            return false;
        }
    }

    bool IopIntrman::dispatchInterrupt(int irq, IopGuestExecutor &executor) const
    {
        const auto enabled = m_enabled.find(irq);
        if (enabled == m_enabled.end() || !enabled->second)
            return false;
        const auto handler = m_handlers.find(irq);
        if (handler == m_handlers.end() || handler->second.function == 0u)
            return false;

        (void)executor.executeGuestFunctionWithBudget(handler->second.function,
                                                      handler->second.argument,
                                                      0u,
                                                      0u,
                                                      0u,
                                                      handler->second.gp,
                                                      100000u);
        return true;
    }
}
