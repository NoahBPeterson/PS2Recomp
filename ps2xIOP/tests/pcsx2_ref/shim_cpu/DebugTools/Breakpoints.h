#pragma once
#include "common/Pcsx2Defs.h"
#include <vector>
enum BreakPointCpu { BREAKPOINT_IOP = 2 };
#define MEMCHECK_READ 1
#define MEMCHECK_WRITE 2
struct BpCondShim { bool Evaluate() const { return true; } };
struct MemCheckShim { u32 result = 0, memCond = 0, start = 0, end = 0; bool hasCond = false; BpCondShim cond; };
struct CBreakPoints {
  static int CheckSkipFirst(int, u32) { return 0; } static void ClearSkipFirst(int) {}
  static BpCondShim *GetBreakPointCondition(int, u32) { return nullptr; } static void SetBreakpointTriggered(bool, int) {}
  static std::vector<MemCheckShim> GetMemChecks(int) { return {}; } static void CommitClearSkipFirst(int) {} };
inline int psxIsBreakpointNeeded(u32) { return 0; }
inline int psxIsMemcheckNeeded(u32) { return 0; }
