// The reference side of ps2x_vu1_pcsx2_diff: PCSX2 2.8.2's VU1 interpreter (VUops.cpp, VUflags.cpp,
// VU1microInterp.cpp, compiled from PS2X_PCSX2_SRC against the stand-ins in shim/), driven the way
// PCSX2 drives it for a VU1 microprogram: vu1ExecMicro (VPU_STAT.VBS1 set, TPC = start) and then
// InterpVU1::Execute's loop of InterpVU1::Step until VBS1 clears (E-bit end, or a D/T-bit stop).
//
// Excluded PCSX2 behaviour, and why:
// - MTVU (VU1 on its own thread) and microVU (the recompiler): THREAD_VU1 / REC_VU1 are false. The
//   interpreter is the reference; MTVU only changes when results become visible to the EE.
// - Timing: VU1.cycle runs (stalls, XGKICK pacing at 2 cycles per qword) but is not compared, and
//   the EE side of the cycle accounting (cpuRegs.cycle, nextBlockCycles, ExecuteBlock's delta runs)
//   is not modelled: each run executes to its end in one go, like PCSX2's INSTANT_VU1 (default on).
//   For a run that the recording stopped on its cycle budget (no end), the same number of pairs runs.
// - The GS: PATH1 data is captured as bytes (shim/Gif_Unit.h), with no PATH arbitration, masking or
//   GIF FIFO, and no MTGS.
// - Interrupts: a D/T-bit stop only records that INTC_VU1 was raised (hwIntcIrq).
// - VU0 and the COP2 macro instructions (compiled, never run), VU0-mapped VU1 registers.
// Host floating point is set like PCSX2's VU1FPCR (round toward zero, DaZ + FtZ) for the whole run,
// and the PCSX2 files are compiled with FP contraction off (x86 PCSX2 never fuses a*b+c; an AArch64
// compiler would by default) and dynamic rounding respected (-frounding-math).

#include "Common.h"
#include "VUmicro.h"
#include "Gif_Unit.h"
#include "MTVU.h"
#include "vu1_ref.h"

#include <cfenv>
#include <cstring>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#define VU1REF_X86 1
#elif defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>
#endif

// --- what the PCSX2 files expect from the rest of the emulator ---
alignas(16) VURegs vuRegs[2];
EmuConfigShim EmuConfig;
CpuRegsShim cpuRegs;
VIFregisters vif0Regs, vif1Regs;
VU_Thread_Shim vu1Thread;
Gif_Unit gifUnit;
static bool g_irq = false;
void hwIntcIrq(int) { g_irq = true; }

FPControlRegisterBackup::FPControlRegisterBackup(const FPControlRegister &)
{
    m_savedRound = std::fegetround();
#if defined(VU1REF_X86)
    m_saved = _mm_getcsr();
    std::fesetround(FE_TOWARDZERO);
    _mm_setcsr(_mm_getcsr() | 0x8040u); // FTZ (bit 15) | DAZ (bit 6)
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    m_saved = fpcr;
    std::fesetround(FE_TOWARDZERO);
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= 1ull << 24; // FZ: denormal inputs and results flush to zero (DaZ + FtZ)
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#elif defined(_MSC_VER) && defined(_M_ARM64)
    m_saved = _ReadStatusReg(ARM64_FPCR);
    std::fesetround(FE_TOWARDZERO);
    _WriteStatusReg(ARM64_FPCR, _ReadStatusReg(ARM64_FPCR) | (1ll << 24));
#else
    std::fesetround(FE_TOWARDZERO); // no DaZ/FtZ control known for this host
#endif
}

FPControlRegisterBackup::~FPControlRegisterBackup()
{
#if defined(VU1REF_X86)
    _mm_setcsr(static_cast<unsigned>(m_saved));
#elif defined(__aarch64__)
    __asm__ volatile("msr fpcr, %0" : : "r"(m_saved));
#elif defined(_MSC_VER) && defined(_M_ARM64)
    _WriteStatusReg(ARM64_FPCR, static_cast<__int64>(m_saved));
#endif
    std::fesetround(m_savedRound);
}

namespace
{
    // Slack after each 16 KB so a qword access at the last address stays inside (PCSX2 masks with 0x3fff).
    alignas(16) u8 g_vu1Mem[0x4000 + 64];
    alignas(16) u8 g_vu1Micro[0x4000 + 64];
    alignas(16) u8 g_vu0Mem[0x1000 + 64];
    alignas(16) u8 g_vu0Micro[0x1000 + 64];

    vu1ref::ArchState snapshot(bool tpcInBytes)
    {
        const VURegs &v = VU1;
        vu1ref::ArchState s{};
        std::memcpy(s.vf, v.VF, sizeof(s.vf));
        for (int r = 0; r < 16; ++r)
            s.vi[r] = v.VI[r].US[0];
        std::memcpy(s.acc, &v.ACC, sizeof(s.acc));
        s.q = v.VI[REG_Q].UL;
        s.p = v.VI[REG_P].UL;
        s.i = v.VI[REG_I].UL;
        s.r = v.VI[REG_R].UL;
        s.mac = v.VI[REG_MAC_FLAG].UL;
        s.status = v.VI[REG_STATUS_FLAG].UL;
        s.clip = v.VI[REG_CLIP_FLAG].UL;
        s.pc = tpcInBytes ? v.VI[REG_TPC].UL : v.VI[REG_TPC].UL << 3;
        return s;
    }
}

namespace vu1ref
{
    void init()
    {
        for (VURegs &v : vuRegs)
            std::memset(static_cast<void *>(&v), 0, sizeof(VURegs));
        vuRegs[0].idx = 0;
        vuRegs[0].Mem = g_vu0Mem;
        vuRegs[0].Micro = g_vu0Micro;
        vuRegs[1].idx = 1;
        vuRegs[1].Mem = g_vu1Mem;
        vuRegs[1].Micro = g_vu1Micro;
        VU1.VF[0].f.w = 1.0f;
        CpuIntVU1.Reset();
    }

    void loadCode(const uint8_t *code) { std::memcpy(g_vu1Micro, code, 0x4000); }
    void loadData(const uint8_t *d) { std::memcpy(g_vu1Mem, d, 0x4000); }
    const uint8_t *data() { return g_vu1Mem; }
    ArchState state() { return snapshot(false); }
    const std::vector<uint8_t> &path1() { return gifUnit.path1; }
    uint32_t path1Packets() { return gifUnit.packetsEnded; }

    RunResult run(const RunSetup &setup, uint64_t maxPairs, std::vector<ArchState> *steps)
    {
        VURegs &v = VU1;
        if (!setup.resume)
        {
            // MSCAL starts with empty pipelines (the recording's contract; in PCSX2 a program that
            // ended has flushed them all in _vuFlushAll anyway).
            v.fmacreadpos = v.fmacwritepos = v.fmaccount = 0;
            v.ialureadpos = v.ialuwritepos = v.ialucount = 0;
            std::memset(v.fmac, 0, sizeof(v.fmac));
            std::memset(v.ialu, 0, sizeof(v.ialu));
            v.fdiv.enable = 0;
            v.efu.enable = 0;
            v.branch = 0;
            v.takedelaybranch = false;
            v.ebit = 0;
            v.xgkickenable = 0;
            v.VIBackupCycles = 0;
        }
        const ArchState &in = setup.regs;
        std::memcpy(v.VF, in.vf, sizeof(in.vf));
        for (int r = 0; r < 16; ++r)
            v.VI[r].UL = in.vi[r];
        std::memcpy(&v.ACC, in.acc, sizeof(in.acc));
        v.VI[REG_Q].UL = v.q.UL = in.q;
        v.VI[REG_P].UL = v.p.UL = in.p;
        v.VI[REG_I].UL = in.i;
        v.VI[REG_R].UL = in.r;
        v.VI[REG_MAC_FLAG].UL = v.macflag = in.mac;
        v.VI[REG_STATUS_FLAG].UL = v.statusflag = in.status;
        v.VI[REG_CLIP_FLAG].UL = v.clipflag = in.clip;
        v.cycle = cpuRegs.cycle = setup.cycle;

        vif1Regs.top = setup.top;
        vif1Regs.itop = setup.itop;
        VU0.VI[REG_FBRST].UL = (setup.dBit ? 0x400u : 0u) | (setup.tBit ? 0x800u : 0u);
        gifUnit.path1.clear();
        gifUnit.transfers = gifUnit.packetsEnded = 0;
        g_irq = false;

        // vu1ExecMicro
        VU0.VI[REG_VPU_STAT].UL &= ~0xFF00u;
        VU0.VI[REG_VPU_STAT].UL |= 0x0100u;
        v.VI[REG_TPC].UL = ((setup.resume ? in.pc : setup.startPc) >> 3) & 0x7FF;
        CpuIntVU1.SetStartPC(v.VI[REG_TPC].UL << 3);

        // InterpVU1::Execute, with a pair limit instead of a cycle limit and the per-pair snapshots.
        RunResult result;
        {
            const FPControlRegisterBackup fpcr(EmuConfig.Cpu.VU1FPCR);
            v.VI[REG_TPC].UL <<= 3;
            for (;;)
            {
                if (!(VU0.VI[REG_VPU_STAT].UL & 0x100))
                {
                    if (v.branch == 1)
                    {
                        v.VI[REG_TPC].UL = v.branchpc;
                        v.branch = 0;
                    }
                    result.ended = true;
                    break;
                }
                if (result.pairs >= maxPairs)
                    break;
                if (steps)
                    steps->push_back(snapshot(true));
                CpuIntVU1.Step();
                ++result.pairs;
            }
            v.VI[REG_TPC].UL >>= 3;
        }
        if (steps)
            steps->push_back(snapshot(false));
        result.interrupted = g_irq;
        return result;
    }
}
