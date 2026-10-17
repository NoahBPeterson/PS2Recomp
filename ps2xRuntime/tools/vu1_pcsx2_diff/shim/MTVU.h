// Stand-in for PCSX2's MTVU.h. MTVU (VU1 on its own thread) is excluded: THREAD_VU1 is false, so
// only the type is needed for the dead `vu1Thread.vifRegs` branches of XTOP/XITOP.
#pragma once
#include "common/Pcsx2Defs.h"
struct VU_Thread_Shim
{
    struct
    {
        u32 top = 0, itop = 0;
    } vifRegs;
};
extern VU_Thread_Shim vu1Thread;
