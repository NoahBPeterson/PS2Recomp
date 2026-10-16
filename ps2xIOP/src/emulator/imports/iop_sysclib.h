#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopMemory;
    class IopGuestExecutor;

    // Successive 32-bit variadic arguments: first the ones passed in registers, then the words
    // in guest memory from `memory` on (o32: the caller's outgoing area at sp+16, or a va_list).
    class IopVarArgs
    {
    public:
        IopVarArgs(const IopMemory &memory, std::vector<uint32_t> registers, uint32_t next) noexcept
            : m_memory(memory), m_registers(std::move(registers)), m_next(next)
        {
        }
        uint32_t next();

    private:
        const IopMemory &m_memory;
        std::vector<uint32_t> m_registers;
        size_t m_used = 0;
        uint32_t m_next;
    };

    // SYSCLIB prnt() as the IOPRP300 firmware implements it (checked against the original module by
    // ps2_iop_ground_truth): emit(0x200) first, every output character (sign-extended), emit(0x201)
    // last. Returns the character count. %n stores through guest memory.
    int32_t iopPrnt(IopMemory &memory, uint32_t format, IopVarArgs &args, const std::function<void(int32_t)> &emit);

    // The firmware's ctype flags for c in [-1, 127] (0 elsewhere): 1 upper, 2 lower, 4 digit,
    // 8 space, 0x10 punct, 0x20 control, 0x40 hex letter.
    uint8_t iopCtype(int32_t c) noexcept;

    class IopSysclib
    {
    public:
        explicit IopSysclib(IopMemory &memory) noexcept;

        void reset() noexcept;
        // prnt (18) calls a guest callback per character; without an executor it returns 0.
        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu, IopGuestExecutor *executor = nullptr);

    private:
        IopMemory &m_memory;
        uint32_t m_strtokSave = 0;   // strtok()'s static "lasts" pointer
        uint32_t m_ctypeTable = 0;   // guest copy of the ctype table (get_ctype_table)
    };
}
