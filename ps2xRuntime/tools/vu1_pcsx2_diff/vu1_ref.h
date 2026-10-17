// Interface between ps2x_vu1_pcsx2_diff's driver (runtime side) and pcsx2_ref.cpp, which drives
// PCSX2's VU1 interpreter (compiled from PS2X_PCSX2_SRC). Plain types only: neither side sees the
// other's headers (PCSX2's VU.h and the runtime both define VU0/VU1-ish names).
#pragma once

#include <cstdint>
#include <vector>

namespace vu1ref
{
    // Architectural VU1 state as both interpreters expose it (raw float bits).
    struct ArchState
    {
        uint32_t vf[32][4];
        uint32_t vi[16]; // 16-bit values, zero-extended
        uint32_t acc[4];
        uint32_t q, p, i, r;
        uint32_t mac, status, clip;
        uint32_t pc; // bytes
    };

    struct RunSetup
    {
        bool resume = false;      // MSCNT: continue with PCSX2's own pipeline state from the previous run
        uint32_t startPc = 0;     // bytes (MSCAL)
        uint32_t top = 0, itop = 0;
        bool dBit = false, tBit = false; // FBRST.DE1 / TE1
        uint64_t cycle = 0;
        ArchState regs{};         // loaded before the run (pc used for a resume)
    };

    struct RunResult
    {
        bool ended = false;       // VPU_STAT VBS1 cleared (E-bit, or a D/T-bit stop)
        bool interrupted = false; // D/T bit raised INTC_VU1
        uint64_t pairs = 0;       // instruction pairs stepped
    };

    void init();
    void loadCode(const uint8_t *code);   // 16 KB micro memory
    void loadData(const uint8_t *data);   // 16 KB data memory
    const uint8_t *data();                // data memory after a run
    // Runs until the program ends or maxPairs pairs ran. With `steps`, the state before every pair
    // (and once more after the last) is appended.
    RunResult run(const RunSetup &setup, uint64_t maxPairs, std::vector<ArchState> *steps);
    ArchState state();
    // PATH1 bytes the last run sent (XGKICK), and the number of EOP GIFtags among them.
    const std::vector<uint8_t> &path1();
    uint32_t path1Packets();
}
