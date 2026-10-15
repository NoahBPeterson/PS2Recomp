// True Crime: New York City (SLUS-21106, v1.03) — TC2.ELF game overrides.
#include "game_overrides.h"

#include <iostream>

namespace
{
    void applyTrueCrimeNYC(PS2Runtime &runtime)
    {
        // The game's debug printers end in DECI2/SIO output on hardware, which the
        // runtime drops; route them to the runtime printf so its progress log
        // ("Loading world...", "Done loading menu script.") is visible.
        const bool kprintfBound = ps2_game_overrides::bindAddressHandler(runtime, 0x004A73E0u, "printf");
        std::cerr << "[override:tcnyc] kprintf@0x4A73E0 -> printf " << (kprintfBound ? "bound" : "FAILED") << std::endl;
    }
}

PS2_REGISTER_GAME_OVERRIDE("True Crime: New York City (SLUS-21106)", "TC2.ELF", 0x004B2468u, 0u, applyTrueCrimeNYC)
