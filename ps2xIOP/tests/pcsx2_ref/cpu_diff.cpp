// Differential test of the IOP CPU (IopCpuCore) against PCSX2 2.8.2's R3000A interpreter
// (R3000AInterpreter.cpp + R3000AOpcodeTables.cpp, compiled from PS2X_PCSX2_SRC with shim_cpu/).
// Random instruction blocks run on both until a halt loop; GPRs, HI/LO, COP0 Status/Cause/EPC,
// the data area and the final pc must match.
// Not comparable, so never generated or skipped when our core takes them: PCSX2 has no load delay
// slots (every load is followed by a NOP; delay slots hold ALU instructions only), no overflow,
// address-error or reserved-instruction exceptions (sequences where ours raises one are skipped
// and counted).
//   ps2_iop_cpu_diff [sequences=20000] [instructions=48] [seed=1]   exit 0 = all match
#include "emulator/core/iop_cpu.h"
#include "emulator/core/iop_memory.h"

#include "R3000A.h"
#include "IopHw.h"
#include "IopMem.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

alignas(16) psxRegisters psxRegs;
u32 g_refIopHw[0x1000];
s32 psxNextDeltaCounter;
u64 psxNextStartCounter;
bool iopEventAction = false;
bool iopEventTestIsActive = false;
static std::vector<u8> g_refRam(2u << 20);

u8 iopMemRead8(u32 a) { return g_refRam[a & 0x1FFFFF]; }
u16 iopMemRead16(u32 a) { u16 v; std::memcpy(&v, &g_refRam[a & 0x1FFFFE], 2); return v; }
u32 iopMemRead32(u32 a) { u32 v; std::memcpy(&v, &g_refRam[a & 0x1FFFFC], 4); return v; }
void iopMemWrite8(u32 a, u8 v) { g_refRam[a & 0x1FFFFF] = v; }
void iopMemWrite16(u32 a, u16 v) { std::memcpy(&g_refRam[a & 0x1FFFFE], &v, 2); }
void iopMemWrite32(u32 a, u32 v) { std::memcpy(&g_refRam[a & 0x1FFFFC], &v, 4); }
void iopEventTest() {}
bool psxBiosCall() { return false; }

// PCSX2's psxException (R3000A.cpp), which pulls in the whole IOP otherwise.
void psxException(u32 code, u32 bd)
{
    psxRegs.CP0.n.Cause &= ~0x7f;
    psxRegs.CP0.n.Cause |= code;
    if (bd)
    {
        psxRegs.CP0.n.Cause |= 0x80000000;
        psxRegs.CP0.n.EPC = (psxRegs.pc - 4);
    }
    else
        psxRegs.CP0.n.EPC = (psxRegs.pc);
    if (psxRegs.CP0.n.Status & 0x400000)
        psxRegs.pc = 0xbfc00180;
    else
        psxRegs.pc = 0x80000080;
    psxRegs.CP0.n.Status = (psxRegs.CP0.n.Status & ~0x3f) | ((psxRegs.CP0.n.Status & 0xf) << 2);
}

namespace
{
    using namespace ps2x::iop::detail;

    constexpr u32 kCode = 0x1000u, kHalt = 0x8000u, kVector = 0x80u, kData = 0x10000u, kDataSize = 0x800u;
    constexpr u32 kBase = 23, kJumpTemp = 22; // $s7 = data base, $s6 = jump target (never random destinations)
    constexpr u32 NOP = 0;

    u32 R(u32 rs, u32 rt, u32 rd, u32 sa, u32 fn) { return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | fn; }
    u32 I(u32 op, u32 rs, u32 rt, u32 imm) { return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFFu); }
    u32 J(u32 op, u32 target) { return (op << 26) | ((target >> 2) & 0x03FFFFFFu); }

    struct Gen
    {
        std::mt19937 rng;
        explicit Gen(u32 seed) : rng(seed) {}
        u32 pick(u32 n) { return static_cast<u32>(rng() % n); }
        u32 reg()
        {
            for (;;)
            {
                const u32 r = pick(32);
                if (r != kBase && r != kJumpTemp)
                    return r;
            }
        }
        u32 dst()
        {
            for (;;)
            {
                const u32 r = 1 + pick(31);
                if (r != kBase && r != kJumpTemp)
                    return r;
            }
        }
        // ALU instructions only (used in delay slots too).
        u32 alu()
        {
            static constexpr u32 kRType[] = {0x21, 0x23, 0x24, 0x25, 0x26, 0x27, 0x2A, 0x2B, 0x04, 0x06, 0x07, 0x20, 0x22};
            switch (pick(6))
            {
            case 0:
            case 1:
                return R(reg(), reg(), dst(), 0, kRType[pick(sizeof(kRType) / 4)]);
            case 2:
            {
                static constexpr u32 kShift[] = {0x00, 0x02, 0x03};
                return R(0, reg(), dst(), pick(32), kShift[pick(3)]);
            }
            case 3:
            {
                static constexpr u32 kIType[] = {9, 8, 10, 11, 12, 13, 14, 15};
                const u32 op = kIType[pick(8)];
                return I(op, op == 15 ? 0 : reg(), dst(), static_cast<u32>(rng()));
            }
            case 4:
            {
                static constexpr u32 kMulDiv[] = {0x18, 0x19, 0x1A, 0x1B, 0x11, 0x13};
                return R(reg(), reg(), 0, 0, kMulDiv[pick(6)]);
            }
            default:
                return R(0, 0, dst(), 0, pick(2) ? 0x10 : 0x12); // MFHI / MFLO
            }
        }
    };

    struct Program
    {
        std::vector<u32> code;
        std::vector<u8> data;
        u32 gpr[32] = {};
        u32 hi = 0, lo = 0, status = 0;
    };

    Program generate(u32 seed, u32 count)
    {
        Gen g(seed);
        Program p;
        p.data.resize(kDataSize);
        for (auto &b : p.data)
            b = static_cast<u8>(g.rng());
        for (u32 r = 1; r < 32; ++r)
            p.gpr[r] = g.pick(4) == 0 ? g.pick(16) - 8u : static_cast<u32>(g.rng());
        p.gpr[kBase] = kData;
        p.hi = static_cast<u32>(g.rng());
        p.lo = static_cast<u32>(g.rng());
        p.status = g.pick(64) & 0x3Fu; // KU/IE stack bits; BEV and cache isolation stay clear
        std::vector<u32> &c = p.code;
        auto here = [&]() { return kCode + 4u * static_cast<u32>(c.size()); };
        while (c.size() < count)
        {
            const u32 k = g.pick(100);
            if (k < 45)
                c.push_back(g.alu());
            else if (k < 60)
            {
                static constexpr struct { u32 op, align; } kLoads[] = {{0x20, 1}, {0x24, 1}, {0x21, 2}, {0x25, 2}, {0x23, 4}, {0x22, 1}, {0x26, 1}};
                const auto &l = kLoads[g.pick(7)];
                c.push_back(I(l.op, kBase, g.dst(), g.pick(kDataSize - 4) & ~(l.align - 1)));
                c.push_back(NOP); // no load-delay hazard (PCSX2 has none)
            }
            else if (k < 70)
            {
                static constexpr struct { u32 op, align; } kStores[] = {{0x28, 1}, {0x29, 2}, {0x2B, 4}, {0x2A, 1}, {0x2E, 1}};
                const auto &s = kStores[g.pick(5)];
                c.push_back(I(s.op, kBase, g.reg(), g.pick(kDataSize - 4) & ~(s.align - 1)));
            }
            else if (k < 84)
            {
                // Forward conditional branch over 0..5 instructions, ALU delay slot.
                const u32 skip = g.pick(6);
                const u32 kind = g.pick(8);
                u32 insn;
                if (kind < 2)
                    insn = I(kind == 0 ? 4 : 5, g.reg(), g.reg(), skip + 1); // BEQ / BNE (offset counts from the delay slot)
                else if (kind < 4)
                    insn = I(kind == 2 ? 6 : 7, g.reg(), 0, skip + 1);       // BLEZ / BGTZ
                else
                {
                    static constexpr u32 kRegimm[] = {0x00, 0x01, 0x10, 0x11}; // BLTZ BGEZ BLTZAL BGEZAL
                    insn = I(1, g.reg(), kRegimm[kind - 4], skip + 1);
                }
                c.push_back(insn);
                c.push_back(g.alu());
                for (u32 i = 0; i < skip; ++i)
                    c.push_back(g.alu());
            }
            else if (k < 90)
            {
                // Forward jump: J / JAL, or JR / JALR through $s6.
                const u32 skip = g.pick(4);
                const u32 kind = g.pick(4);
                if (kind < 2)
                {
                    const u32 target = here() + 8u + 4u * skip;
                    c.push_back(J(kind == 0 ? 2 : 3, target));
                }
                else
                {
                    const u32 target = here() + 16u + 4u * skip;
                    c.push_back(I(15, 0, kJumpTemp, target >> 16));
                    c.push_back(I(13, kJumpTemp, kJumpTemp, target & 0xFFFFu));
                    c.push_back(kind == 2 ? R(kJumpTemp, 0, 0, 0, 0x08) : R(kJumpTemp, 0, g.dst(), 0, 0x09));
                }
                c.push_back(g.alu());
                for (u32 i = 0; i < skip; ++i)
                    c.push_back(g.alu());
            }
            else if (k < 96)
            {
                // COP0: read Status/Cause/EPC/BadVAddr, write EPC, write Status (safe bits), RFE.
                switch (g.pick(4))
                {
                case 0:
                {
                    static constexpr u32 kRegs[] = {12, 13, 14, 8};
                    c.push_back((0x10u << 26) | (0u << 21) | (g.dst() << 16) | (kRegs[g.pick(4)] << 11)); // MFC0
                    c.push_back(NOP);
                    break;
                }
                case 1:
                    c.push_back((0x10u << 26) | (4u << 21) | (g.reg() << 16) | (14u << 11)); // MTC0 EPC
                    break;
                case 2:
                {
                    const u32 value = g.pick(64) | (g.pick(256) << 8);
                    const u32 t = g.dst();
                    c.push_back(I(13, 0, t, value));                              // ori t, zero, value
                    c.push_back((0x10u << 26) | (4u << 21) | (t << 16) | (12u << 11)); // MTC0 Status
                    break;
                }
                default:
                    c.push_back(0x42000010u); // RFE
                    break;
                }
            }
            else if (k < 98)
                c.push_back(g.alu());
            else
            {
                c.push_back(g.pick(2) ? 0x0000000Cu : 0x0000000Du); // SYSCALL / BREAK: exception, ends the run
            }
        }
        // Halt: jump to the halt loop.
        c.push_back(J(2, kHalt));
        c.push_back(NOP);
        return p;
    }

    void place(std::vector<u8> &ram, const Program &p)
    {
        std::fill(ram.begin(), ram.end(), 0);
        std::memcpy(&ram[kCode], p.code.data(), p.code.size() * 4);
        std::memcpy(&ram[kData], p.data.data(), p.data.size());
        const u32 halt[2] = {J(2, kHalt), NOP};
        std::memcpy(&ram[kHalt], halt, 8);
        std::memcpy(&ram[kVector], halt, 8);
    }
}

int main(int argc, char **argv)
{
    const int sequences = argc > 1 ? std::atoi(argv[1]) : 20000;
    const u32 count = argc > 2 ? static_cast<u32>(std::atoi(argv[2])) : 48u;
    const u32 seed0 = argc > 3 ? static_cast<u32>(std::strtoul(argv[3], nullptr, 0)) : 1u;
    uint64_t mismatches = 0, skipped = 0, exceptions = 0;
    int shown = 0;
    std::vector<u8> ram(2u << 20);

    for (int seq = 0; seq < sequences; ++seq)
    {
        const Program p = generate(seed0 * 1000003u + static_cast<u32>(seq), count);

        // Ours.
        IopMemory mem;
        place(ram, p);
        (void)mem.writeRam(0, ram.data(), ram.size());
        IopCpuCore core(mem);
        IopCpuState cpu{};
        for (u32 r = 0; r < 32; ++r)
            cpu.gpr[r] = p.gpr[r];
        cpu.hi = p.hi;
        cpu.lo = p.lo;
        cpu.cop0[12] = p.status;
        cpu.pc = kCode;
        bool skip = false, sawException = false;
        for (int steps = 0; steps < 4096; ++steps)
        {
            if ((cpu.pc & 0x1FFFFFFFu) == kHalt && !cpu.branchPending)
                break;
            (void)core.executeInstruction(cpu);
            if (cpu.exception)
            {
                const u32 code = (cpu.cop0[13] >> 2) & 31u;
                if (code != 8u && code != 9u) // only SYSCALL/BREAK are comparable
                {
                    skip = true;
                    break;
                }
                sawException = true;
                cpu.exception = false;
            }
        }
        if (skip)
        {
            ++skipped;
            continue;
        }
        exceptions += sawException;

        // PCSX2.
        g_refRam = ram;
        std::memset(&psxRegs, 0, sizeof(psxRegs));
        for (u32 r = 0; r < 32; ++r)
            psxRegs.GPR.r[r] = p.gpr[r];
        psxRegs.GPR.n.hi = p.hi;
        psxRegs.GPR.n.lo = p.lo;
        psxRegs.CP0.n.Status = p.status;
        psxRegs.pc = kCode;
        psxInt.ExecuteBlock(8 * 4096);

        auto report = [&](const char *what, u32 a, u32 b) {
            if (a == b)
                return true;
            ++mismatches;
            if (shown++ < 10)
            {
                std::printf("MISMATCH seq %d (seed %u): %s ours=0x%08x pcsx2=0x%08x\n", seq, seed0, what, a, b);
                std::printf("  code:");
                for (u32 w : p.code)
                    std::printf(" %08x", w);
                std::printf("\n");
            }
            return false;
        };
        bool ok = report("final pc", cpu.pc & 0x1FFFFFFFu, psxRegs.pc & 0x1FFFFFFFu);
        for (u32 r = 1; r < 32 && ok; ++r)
        {
            char what[16];
            std::snprintf(what, sizeof(what), "r%u", r);
            ok = report(what, cpu.gpr[r], psxRegs.GPR.r[r]);
        }
        ok = ok && report("hi", cpu.hi, psxRegs.GPR.n.hi) && report("lo", cpu.lo, psxRegs.GPR.n.lo);
        ok = ok && report("Status", cpu.cop0[12], psxRegs.CP0.n.Status);
        ok = ok && report("Cause (code, BD)", cpu.cop0[13] & 0x8000007Cu, psxRegs.CP0.n.Cause & 0x8000007Cu);
        ok = ok && report("EPC", cpu.cop0[14], psxRegs.CP0.n.EPC);
        for (u32 off = 0; off < kDataSize && ok; off += 4)
        {
            char what[32];
            std::snprintf(what, sizeof(what), "data +0x%x", off);
            ok = report(what, mem.read32(kData + off), iopMemRead32(kData + off));
        }
    }
    std::printf("cpu_diff: %d sequences x %u instructions, %llu skipped (exceptions PCSX2 does not model), %llu with SYSCALL/BREAK, %llu mismatches\n",
                sequences, count, static_cast<unsigned long long>(skipped), static_cast<unsigned long long>(exceptions),
                static_cast<unsigned long long>(mismatches));
    return mismatches == 0 ? 0 : 1;
}
