#pragma once
#include "common/Pcsx2Defs.h"
extern u32 g_refIopHw[0x1000];
#define HW_ICFG 0x1f801450
#define psxHu32(addr) g_refIopHw[((addr) & 0xfff) >> 2]
