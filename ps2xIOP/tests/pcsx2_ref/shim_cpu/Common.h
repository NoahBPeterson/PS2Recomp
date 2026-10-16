#pragma once
#include "common/Pcsx2Defs.h"
#include "common/Console.h"
#define PSXCPU_LOG(...) ((void)0)
inline const char *disR3000AF(u32, u32) { return ""; }
namespace Ps2MemSize { inline constexpr u32 ExposedIopRam = 2u << 20; }
struct SymbolGuardianShim { void ClearIrxModules() {} };
inline SymbolGuardianShim R3000SymbolGuardian;
struct CpuShim { void ExitExecution() {} };
inline CpuShim CpuObj, *Cpu = &CpuObj;
#include "IopMem.h"
namespace R3000A {}
