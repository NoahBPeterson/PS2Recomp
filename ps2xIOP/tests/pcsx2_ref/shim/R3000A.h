// Stand-in for PCSX2's R3000A.h: the IOP clock and next-event bookkeeping the counters use.
#pragma once
#include "common/Pcsx2Defs.h"
struct psxRegisters { u64 cycle = 0; u64 iopNextEventCycle = 0; };
extern psxRegisters psxRegs;
void psxSetNextBranch(u64 startCycle, s32 delta);
#define PSXCLK (36864000)
