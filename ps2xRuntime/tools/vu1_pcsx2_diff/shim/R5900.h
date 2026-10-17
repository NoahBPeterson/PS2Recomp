// Stand-in for PCSX2's R5900.h: the EE state the VU files reference (COP2 macro code reads
// cpuRegs.code; VU1 bookkeeping reads cpuRegs.cycle) and the event scheduler hook XGKICK calls.
#pragma once
#include "common/Pcsx2Defs.h"
struct CpuRegsShim
{
    u32 code = 0;
    u64 cycle = 0;
};
extern CpuRegsShim cpuRegs;
enum { DMAC_VIF1 = 1 };
inline void CPU_INT(int, int) {}
