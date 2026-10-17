// Stand-in for PCSX2's MemoryTypes.h: the EE hardware register block that Vif.h maps
// vif0Regs / vif1Regs onto (eeHw[0x3800] / eeHw[0x3C00]).
#pragma once
#include "Common.h"
alignas(16) extern u8 eeHw[0x10000];
