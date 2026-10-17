// Stand-in for PCSX2's Gif_Unit.h: the two calls the VU1 interpreter's XGKICK makes. Instead of a
// GS, PATH1 data goes to a byte capture (the harness compares it with the recorded PATH1 packets).
#pragma once
#include "common/Pcsx2Defs.h"
#include <vector>

enum GIF_PATH { GIF_PATH_1 = 0, GIF_PATH_2, GIF_PATH_3 };
enum GIF_TRANSFER_TYPE { GIF_TRANS_INVALID = 0x000, GIF_TRANS_XGKICK = 0x100 };

struct Gif_Unit
{
    std::vector<u8> path1;       // every PATH1 byte transferred, in order
    u32 transfers = 0;           // TransferGSPacketData calls with data
    u32 packetsEnded = 0;        // GIFtags with EOP whose data was sized (one per completed packet)

    // PCSX2 Gif_Unit::GetGSPacketSize for the VU1 interpreter: as REC_VU1 is false, it returns the
    // size of the one GIFtag at `offset` (tag + data, Gif_Tag::setTag lengths) with bit 31 = EOP,
    // or 0 when a packet would reach 16 KB.
    u32 GetGSPacketSize(GIF_PATH pathIdx, u8 *pMem, u32 offset = 0, u32 size = ~0u, bool flush = false)
    {
        (void)flush;
        const u32 memMask = pathIdx ? ~0u : 0x3fffu;
        u64 lo;
        std::memcpy(&lo, &pMem[offset & memMask], 8);
        const u32 nloop = static_cast<u32>(lo & 0x7fff), eop = static_cast<u32>((lo >> 15) & 1);
        const u32 flg = static_cast<u32>((lo >> 58) & 3), nreg = ((static_cast<u32>((lo >> 60) & 0xf) - 1) & 0xf) + 1;
        u32 len;
        if (flg == 0)
            len = nreg * nloop * 16;
        else if (flg == 1)
            len = ((nreg * nloop + 1) >> 1) * 16;
        else
            len = nloop * 16;
        const u32 curSize = 16 + len;
        if (pathIdx == GIF_PATH_1 && curSize >= 0x4000)
            return 0;
        if (curSize >= size)
            return size;
        if (eop)
            ++packetsEnded;
        return curSize | (eop << 31);
    }

    u32 TransferGSPacketData(GIF_TRANSFER_TYPE, u8 *pMem, u32 size, bool = false)
    {
        if (size == 0)
            return 0;
        path1.insert(path1.end(), pMem, pMem + size);
        ++transfers;
        return size;
    }
};
extern Gif_Unit gifUnit;
