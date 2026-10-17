// Interface between ps2x_vif1_pcsx2_diff's driver (runtime side) and pcsx2_vif_ref.cpp, which
// drives PCSX2's VIF1 command interpreter (Vif_Codes.cpp, Vif_Unpack.cpp, Vif_Transfer.cpp,
// compiled from PS2X_PCSX2_SRC). Plain types only: neither side sees the other's headers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vif1ref
{
    // The VIF1 registers both interpreters model (row/col: VIF1_R0-R3 / C0-C3).
    struct Regs
    {
        uint32_t stat, err, mark, cycle, mode, num, mask, code, itops, base, ofst, tops, itop, top;
        uint32_t row[4], col[4];
    };

    // A VU1 start: MSCAL/MSCALF (cont = false, pc in instruction units) or MSCNT (cont = true),
    // with the TOP/ITOP that VU1 would see.
    struct Start
    {
        bool cont;
        uint32_t pc, top, itop;
    };

    struct Result
    {
        bool stalled = false;     // the transfer stopped early (vifstalled: an unknown code, a stall)
        uint32_t wordsLeft = 0;   // words of the packet not consumed
        bool cmdOpen = false;     // a command still waits for data at the end (vif1.cmd != 0)
        uint32_t dmaStalls = 0;   // CPU_SET_DMASTALL calls (VIF waiting for VU/GIF): never expected
    };

    // Resets VIF1 (like vif1Reset) and loads the initial registers and the 16 KB VU1 memories.
    void reset(const Regs &regs, const uint8_t *vuData, const uint8_t *vuMicro);
    // One VIF1 packet through VIF1transfer (TTE = true: no DMA channel accounting). `words` must
    // stay readable for 16 bytes past the end (V3 unpacks read the next element).
    Result transfer(uint32_t *words, size_t count);
    Regs regs();
    const uint8_t *vuData();
    const uint8_t *vuMicro();
    const std::vector<Start> &starts();
}
