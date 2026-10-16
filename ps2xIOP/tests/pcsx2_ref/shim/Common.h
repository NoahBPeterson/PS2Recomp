#pragma once
#include "common/Pcsx2Defs.h"
struct ConsoleShim { template <typename... A> void Warning(const char *, A...) {} template <typename... A> void WriteLn(const char *, A...) {} };
inline ConsoleShim Console, DevCon;
#define PSXCNT_LOG(...) ((void)0)
// SaveStateBase stand-in (psxRcntFreeze is compiled but never called).
class SaveStateBase { public:
  bool FreezeTag(const char *) { return true; } template <typename T> void Freeze(T &) {}
  bool IsOkay() const { return true; } bool IsLoading() const { return false; } bool psxRcntFreeze(); };
