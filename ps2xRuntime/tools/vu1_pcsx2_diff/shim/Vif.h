// Stand-in for PCSX2's Vif.h: the VIF registers the VU code touches (TOP/ITOP for XTOP/XITOP, the
// VEW/VGW status bits it clears at program end / XGKICK end).
#pragma once
#include "common/Pcsx2Defs.h"
struct VIFregisters
{
    struct
    {
        bool VEW = false, VGW = false;
    } stat;
    u32 top = 0, itop = 0;
};
extern VIFregisters vif0Regs, vif1Regs;
