// Stand-in for PCSX2's Gif_Unit.h / Gif.h: the GIF is idle (no PATH busy, so FLUSH/FLUSHA/MSCALF
// never stall), MSKPATH3 only sets GIF_STAT.M3P, and DIRECT/DIRECTHL (out of scope) accept data.
#pragma once
#include "Common.h"
enum GIF_TRANSFER_TYPE { GIF_TRANS_DIRECT = 0x301, GIF_TRANS_DIRECTHL = 0x401 };
struct Gif_Unit_Shim
{
    u32 checkPaths(bool, bool, bool) const { return 0; }
    uint TransferGSPacketData(GIF_TRANSFER_TYPE, u8 *, uint size) { return size; }
};
inline Gif_Unit_Shim gifUnit;
struct GifRegsShim
{
    struct { u32 APATH, M3P; } stat;
};
inline GifRegsShim gifRegs;
inline void gifInterrupt() {}
