// Stand-in for PCSX2's MTVU.h: VU1 is not threaded (THREAD_VU1 false), and INSTANT_VU1 is on
// (PCSX2's default), so a MSCAL is queued and started by vifExecQueue.
#pragma once
#include "Common.h"
#include "Vif_Dma.h"
#define THREAD_VU1 false
#define INSTANT_VU1 true
struct VU_Thread_Shim
{
    vifStruct &vif;
    VIFregisters &vifRegs;
    void VifUnpack(vifStruct &, VIFregisters &, u8 *, u32) {}
    void WriteCol(vifStruct &) {}
    void WriteRow(vifStruct &) {}
    void WriteMicroMem(u32, void *, s32) {}
};
extern VU_Thread_Shim vu1Thread;
