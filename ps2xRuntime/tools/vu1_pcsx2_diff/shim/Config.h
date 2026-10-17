// Stand-in for PCSX2's Config.h: the EmuConfig fields and CHECK_* macros the VU1 interpreter reads,
// with PCSX2 2.8.2's defaults (Pcsx2Config.cpp: RecompilerOptions() vu0Overflow = vu1Overflow =
// true, extra/sign overflow off; VU0FPCR = VU1FPCR = DEFAULT_VU_FP_CONTROL_REGISTER: exceptions
// off, DaZ + FtZ, round toward zero; gamefixes and speedhacks off).
// The interpreter reads CHECK_VU_OVERFLOW only: vuDouble() clamps Inf/NaN operands to +-max with
// CHECK_VU_OVERFLOW(0) (VU0's setting, even on VU1) and VU_MAC_UPDATE clamps an Inf/NaN result with
// CHECK_VU_OVERFLOW(1). Extra/sign overflow are recompiler-only options (unused here).
#pragma once
#include "common/Pcsx2Defs.h"

struct FPControlRegister
{
    // Only one setting is ever used here: DEFAULT_VU_FP_CONTROL_REGISTER (see FPControlRegisterBackup).
    u32 bits = 0;
};

// Host FP control for the reference run (PCSX2 common/FPControl.h's FPControlRegisterBackup with
// VU1FPCR): round toward zero plus denormals-are-zero / flush-to-zero, restored on scope exit.
class FPControlRegisterBackup
{
public:
    explicit FPControlRegisterBackup(const FPControlRegister &);
    ~FPControlRegisterBackup();
    FPControlRegisterBackup(const FPControlRegisterBackup &) = delete;
    FPControlRegisterBackup &operator=(const FPControlRegisterBackup &) = delete;

private:
    u64 m_saved = 0;
    int m_savedRound = 0;
};

struct EmuConfigShim
{
    struct
    {
        struct
        {
            bool EnableVU1 = false;
            bool vu0Overflow = true, vu0ExtraOverflow = false, vu0SignOverflow = false, vu0Underflow = false;
            bool vu1Overflow = true, vu1ExtraOverflow = false, vu1SignOverflow = false, vu1Underflow = false;
        } Recompiler;
        FPControlRegister VU0FPCR, VU1FPCR;
    } Cpu;
    struct
    {
        bool VuAddSubHack = false, XgKickHack = false;
    } Gamefixes;
    struct
    {
        bool vu1Instant = true, vuThread = false;
    } Speedhacks;
};
extern EmuConfigShim EmuConfig;

// Interpreter only: no microVU, no MTVU (the VU1 thread), as with PCSX2's VU1 interpreter selected.
#define REC_VU1 false
#define THREAD_VU1 false
#define INSTANT_VU1 (EmuConfig.Speedhacks.vu1Instant)
#define CHECK_VUADDSUBHACK (EmuConfig.Gamefixes.VuAddSubHack)
#define CHECK_XGKICKHACK (EmuConfig.Gamefixes.XgKickHack)
#define CHECK_VU_OVERFLOW(vunum) (((vunum) == 0) ? EmuConfig.Cpu.Recompiler.vu0Overflow : EmuConfig.Cpu.Recompiler.vu1Overflow)
#define CHECK_VU_EXTRA_OVERFLOW(vunum) (((vunum) == 0) ? EmuConfig.Cpu.Recompiler.vu0ExtraOverflow : EmuConfig.Cpu.Recompiler.vu1ExtraOverflow)
#define CHECK_VU_SIGN_OVERFLOW(vunum) (((vunum) == 0) ? EmuConfig.Cpu.Recompiler.vu0SignOverflow : EmuConfig.Cpu.Recompiler.vu1SignOverflow)
