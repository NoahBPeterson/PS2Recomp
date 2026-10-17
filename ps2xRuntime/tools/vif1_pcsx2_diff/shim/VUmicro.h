// Stand-in for PCSX2's VUmicro.h / VU.h: just the VURegs fields the VIF code uses (data and micro
// memory, VI[REG_VPU_STAT]/VI[REG_TPC]) and the VU1 start hook. vu1ExecMicro only records the
// start (pc, TOP, ITOP) and leaves VPU_STAT idle: the VU program "finishes" at once.
#pragma once
#include "Common.h"
enum VURegIndex { REG_TPC = 26, REG_VPU_STAT = 29, REG_FBRST = 28 };
union VIRegShim
{
    u32 UL;
    s32 SL;
    u16 US[2];
};
struct VURegs
{
    VIRegShim VI[32];
    u8 *Mem;
    u8 *Micro;
};
alignas(16) extern VURegs vuRegs[2];
#define VU0 vuRegs[0]
#define VU1 vuRegs[1]
struct VUmicroCpuShim
{
    void Clear(u32, u32) {}
};
extern VUmicroCpuShim *CpuVU0, *CpuVU1;
void vu0ExecMicro(u32 addr);
void vu1ExecMicro(u32 addr);
