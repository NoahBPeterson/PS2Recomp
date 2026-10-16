#pragma once
#include "common/Pcsx2Defs.h"
u8 iopMemRead8(u32 mem); u16 iopMemRead16(u32 mem); u32 iopMemRead32(u32 mem);
void iopMemWrite8(u32 mem, u8 value); void iopMemWrite16(u32 mem, u16 value); void iopMemWrite32(u32 mem, u32 value);
