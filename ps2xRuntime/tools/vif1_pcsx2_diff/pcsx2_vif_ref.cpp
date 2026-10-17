// The reference side of ps2x_vif1_pcsx2_diff: PCSX2 2.8.2's VIF1 command interpreter
// (Vif_Codes.cpp: every VIFcode handler; Vif_Transfer.cpp: vifTransferLoop / VIF1transfer;
// Vif_Unpack.cpp: vifUnpackSetup, nVifUnpack, _nVifUnpackLoop and the C unpackers VIFfuncTable),
// compiled from PS2X_PCSX2_SRC against the stand-ins in shim/.
//
// What runs, and what is excluded:
// - UNPACK goes through nVifUnpack -> _nVifUnpack -> _nVifUnpackLoop with the unpack dynarec off
//   (newVifDynaRec false). For MODE 1-3 that loop calls PCSX2's C unpackers (VIFfuncTable) itself.
//   For MODE 0 PCSX2 calls nVifUpk[], unpackers generated at startup by its x86 emitter
//   (VifUnpackSSE.cpp, or NEON on ARM64); here nVifUpk[] forwards to the same C table at MODE 0,
//   which implements the same function. Not reproduced, therefore: the dynarec's own rules, which
//   differ from the C table in edge cases (PCSX2's normal unpack path is the dynarec):
//     * filling write (CL < WL): the dynarec applies MASK even when the m bit is clear;
//     * V3-32 / V3-8 / V3-16 and aligned V2-32: the dynarec zeroes W in some positions
//       (start_aligned), where the C table takes W from the next element of the stream;
//     * WL = 0: the dynarec takes WL as 256 everywhere; the C loop does only for the size.
// - VU1 does not run: vu1ExecMicro only records (pc, TOP, ITOP); VPU_STAT stays idle, so
//   FLUSHE/FLUSH/MSCAL never wait. MSCAL starts are queued and started by vifExecQueue like in
//   PCSX2 with INSTANT_VU1 (its default) and no MTVU.
// - The GIF is idle (no PATH busy) and DIRECT/DIRECTHL are not generated; MSKPATH3 only sets M3P.
// - DMA: packets go in with TTE = true (no MADR/QWC accounting); no interrupts (i bit not used).

#include "Common.h"
#include "Gif_Unit.h"
#include "VUmicro.h"
#include "MTVU.h"
#include "Vif_Dma.h"
#include "Vif_Dynarec.h"
#include "vif1_ref.h"

#include <cstring>

// --- what the PCSX2 files expect from the rest of the emulator ---
alignas(16) u8 eeHw[0x10000];
alignas(16) VURegs vuRegs[2];
static VUmicroCpuShim g_cpuStub;
VUmicroCpuShim *CpuVU0 = &g_cpuStub, *CpuVU1 = &g_cpuStub;
alignas(16) vifStruct vif0, vif1; // Vif.cpp
VU_Thread_Shim vu1Thread{vif1, (VIFregisters &)eeHw[0x3C00]};
u32 g_vif0Cycles = 0, g_vif1Cycles = 0; // Vif0_Dma.cpp / Vif1_Dma.cpp
void vif0FLUSH() {}
void vif1FLUSH() {} // waits for VU1 to finish: it never runs here
static u32 g_dmaStalls = 0;
void CPU_SET_DMASTALL(EE_EventType, bool set)
{
    if (set)
        ++g_dmaStalls;
}

namespace
{
    alignas(16) u8 g_vu1Mem[0x4000 + 64];
    alignas(16) u8 g_vu1Micro[0x4000 + 64];
    alignas(16) u8 g_vu0Mem[0x1000 + 64];
    alignas(16) u8 g_vu0Micro[0x1000 + 64];
    std::vector<vif1ref::Start> g_starts;
}

void vu0ExecMicro(u32) {}
void vu1ExecMicro(u32 addr)
{
    // VU1micro.cpp: TPC = addr & 0x7ff unless addr is -1 (MSCNT); VU1 reads TOP/ITOP from vif1Regs.
    const bool cont = static_cast<s32>(addr) == -1;
    g_starts.push_back({cont, cont ? 0u : (addr & 0x7FFu), vif1Regs.top, vif1Regs.itop});
}

// nVifUpk[(usn * 32 + upkNum) * 4 + cl3]: the C unpacker of that type at MODE 0 (see the top).
u32 nVifCall::operator()(void *dest, const void *src) const
{
    const ptrdiff_t index = this - nVifUpk;
    VIFfuncTable[1][0][index / 4](dest, src);
    return 0;
}

namespace vif1ref
{
    void reset(const Regs &r, const uint8_t *vuData, const uint8_t *vuMicro)
    {
        for (VURegs &v : vuRegs)
            std::memset(static_cast<void *>(&v), 0, sizeof(VURegs));
        vuRegs[0].Mem = g_vu0Mem;
        vuRegs[0].Micro = g_vu0Micro;
        vuRegs[1].Mem = g_vu1Mem;
        vuRegs[1].Micro = g_vu1Micro;
        std::memcpy(g_vu1Mem, vuData, 0x4000);
        std::memcpy(g_vu1Micro, vuMicro, 0x4000);

        // vif1Reset()
        std::memset(static_cast<void *>(&vif1), 0, sizeof(vif1));
        std::memset(&vif1Regs, 0, sizeof(vif1Regs));
        resetNewVif(1);
        vif1ch = {};
        vif1ch.chcr.STR = 1; // a running DMA: stalls are enabled (VifStallEnable)
        gifRegs = {};
        g_dmaStalls = 0;
        g_starts.clear();

        vif1Regs.stat._u32 = r.stat;
        vif1Regs.err._u32 = r.err;
        vif1Regs.mark = r.mark;
        vif1Regs.cycle.cl = static_cast<u8>(r.cycle);
        vif1Regs.cycle.wl = static_cast<u8>(r.cycle >> 8);
        vif1Regs.mode = r.mode;
        vif1Regs.num = r.num;
        vif1Regs.mask = r.mask;
        vif1Regs.code = r.code;
        vif1Regs.itops = r.itops;
        vif1Regs.base = r.base;
        vif1Regs.ofst = r.ofst;
        vif1Regs.tops = r.tops;
        vif1Regs.itop = r.itop;
        vif1Regs.top = r.top;
        std::memcpy(vif1.MaskRow._u32, r.row, 16);
        std::memcpy(vif1.MaskCol._u32, r.col, 16);
    }

    Result transfer(uint32_t *words, size_t count)
    {
        Result res;
        VIF1transfer(words, static_cast<int>(count), true);
        res.stalled = vif1.vifstalled.enabled;
        res.wordsLeft = vif1.vifpacketsize;
        res.cmdOpen = vif1.cmd != 0;
        res.dmaStalls = g_dmaStalls;
        return res;
    }

    Regs regs()
    {
        Regs r{};
        r.stat = vif1Regs.stat._u32;
        r.err = vif1Regs.err._u32;
        r.mark = vif1Regs.mark;
        r.cycle = vif1Regs.cycle.cl | (static_cast<u32>(vif1Regs.cycle.wl) << 8);
        r.mode = vif1Regs.mode;
        r.num = vif1Regs.num;
        r.mask = vif1Regs.mask;
        r.code = vif1Regs.code;
        r.itops = vif1Regs.itops;
        r.base = vif1Regs.base;
        r.ofst = vif1Regs.ofst;
        r.tops = vif1Regs.tops;
        r.itop = vif1Regs.itop;
        r.top = vif1Regs.top;
        std::memcpy(r.row, vif1.MaskRow._u32, 16);
        std::memcpy(r.col, vif1.MaskCol._u32, 16);
        return r;
    }

    const uint8_t *vuData() { return g_vu1Mem; }
    const uint8_t *vuMicro() { return g_vu1Micro; }
    const std::vector<Start> &starts() { return g_starts; }
}
