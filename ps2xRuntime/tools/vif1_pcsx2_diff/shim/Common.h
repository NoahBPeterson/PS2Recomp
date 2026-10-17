// Stand-in for PCSX2's Common.h as Vif_Codes.cpp / Vif_Unpack.cpp / Vif_Transfer.cpp see it (the
// reference side of ps2x_vif1_pcsx2_diff): basic types, logging compiled away, and the few pieces
// of the DMAC / EE core that the VIF command code touches. See pcsx2_vif_ref.cpp for the
// definitions and for what is excluded.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t s8; typedef int16_t s16; typedef int32_t s32; typedef int64_t s64;
typedef uintptr_t uptr; typedef intptr_t sptr;
typedef unsigned int uint;
union alignas(16) u128
{
    u64 _u64[2];
    u32 _u32[4];
    u16 _u16[8];
    u8 _u8[16];
};

#define __fi
#define __ri
#define __noinline
#define pxAssert(x) ((void)0)
#define pxAssume(x) ((void)0)
#define pxAssertMsg(x, m) ((void)0)
#define pxFail(m) ((void)0)
#define jNO_DEFAULT default: break;

struct ConsoleShim
{
    template <typename... A> void Warning(A...) {}
    template <typename... A> void WriteLn(A...) {}
    template <typename... A> void Error(A...) {}
};
inline ConsoleShim Console, DevCon, DbgCon;

#define VIF_LOG(...) ((void)0)
#define VifCodeLog(...) ((void)0)
#define GUNIT_WARN(...) ((void)0)
#define GUNIT_LOG(...) ((void)0)
#define MSKPATH3_LOG(...) ((void)0)

// Vif_Transfer.cpp's dev-build trace hook and the VIF1 stall game fix: both off.
constexpr bool IsDevBuild = false;
struct TraceLogShim
{
    struct { struct { bool IsActive() const { return false; } } VIFcode; } EE;
};
inline TraceLogShim TraceLogging;
#define CHECK_VIF1STALLHACK false
#define BIAS 2

// DMAC (Dmac.h): only CHCR.STR (stalls are enabled while the channel runs), MADR and QWC are read.
struct DMAChShim
{
    struct { u32 STR; } chcr;
    u32 madr, qwc;
};
inline DMAChShim vif0ch, vif1ch;
inline void hwDmacSrcTadrInc(DMAChShim &) {}
struct DmacRegsShim
{
    struct { bool is_mfifo(bool) const { return false; } } ctrl;
};
inline DmacRegsShim dmacRegs;
enum EE_EventType { DMAC_VIF0 = 0, DMAC_VIF1 = 1, DMAC_MFIFO_VIF = 2 };
// An EE-side stall of the DMA (VIF waiting for the VU or the GIF): recorded, never raised here.
void CPU_SET_DMASTALL(EE_EventType n, bool set);
