// Stand-in for PCSX2's Vif_Dynarec.h: the unpack dynarec (newVifDynaRec) is off, and nVifCall,
// the type of the generated SSE/NEON "interpreted" unpackers (nVifUpk[], used for MODE 0), is a
// callable that forwards to PCSX2's C unpack table VIFfuncTable[1][0][...] instead (no JIT here).
#pragma once
#include "Vif.h"
#include "Vif_Unpack.h"
#include "VUmicro.h"

struct nVifCall
{
    u32 operator()(void *dest, const void *src) const;
};
typedef void (*nVifrecCall)(uptr dest, uptr src);

extern void _nVifUnpack(int idx, const u8 *data, uint mode, bool isFill);
inline void dVifReset(int) {}
inline void dVifRelease(int) {}
inline void VifUnpackSSE_Init() {}
_vifT inline void dVifUnpack(const u8 *, bool) {}

struct nVifStruct
{
    alignas(16) u8 buffer[256 * 16];
    u32 bSize;
    u32 idx;
};

alignas(16) extern nVifStruct nVif[2];
alignas(16) extern nVifCall nVifUpk[(2 * 2 * 16) * 4];
alignas(16) extern u32 nVifMask[3][4][4];

static constexpr bool newVifDynaRec = false;
