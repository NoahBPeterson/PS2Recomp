// Stand-in for PCSX2's Common.h (the VU interpreter's view of the emulator): logging off, the
// config switches the VU code reads (Config.h), and the host FP control used around execution.
#pragma once
#include "common/Pcsx2Defs.h"
#include "Config.h"
#include "R5900.h"

struct ConsoleShim
{
    template <typename... A> void Warning(const char *, A...) {}
    template <typename... A> void WriteLn(A...) {}
    template <typename... A> void Error(const char *, A...) {}
};
inline ConsoleShim Console, DevCon, DbgCon;

// Defined (to nothing) so VUops.cpp's VUM_LOG calls compile away; VUmicro.h then also defines
// IdebugUPPER/IdebugLOWER as VUM_LOG calls, which drop their (disassembler) arguments unevaluated.
#define VUM_LOG(...) ((void)0)
#define CPU_LOG(...) ((void)0)

// INTC (Hw.h): a D/T-bit stop raises INTC_VU1; the harness only records that it happened.
enum { INTC_VU1 = 7 };
void hwIntcIrq(int n);
