#pragma once
#include "common/Pcsx2Defs.h"
#define IS_STORE 1
#define MEMTYPE_MASK 6
#define MEMTYPE_BYTE 2
#define MEMTYPE_HALF 4
#define MEMTYPE_WORD 6
#define MEMTYPE_DWORD 8
namespace R5900 { struct OPCODE { u32 flags = 0; }; inline const OPCODE &GetInstruction(u32) { static OPCODE o; return o; } }
