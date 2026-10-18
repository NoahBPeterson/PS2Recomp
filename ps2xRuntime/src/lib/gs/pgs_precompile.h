#pragma once

// Shared by gs_parallel_backend.cpp and the paraLLEl-GS / Granite sources patched by
// cmake/ParallelGSPrecompile.cmake: ahead-of-time pipeline compilation for paraLLEl-GS.
//
// paraLLEl-GS precompiles a fixed list of shader variants at startup; anything else compiles
// synchronously mid-game. Granite reports each such compute pipeline through
// Vulkan::ps2x_compute_compile_hook; the backend records it, and the next start precompiles the
// recorded variants as well (ParallelGS::ps2x_recorded_variants), after re-deriving each one's
// pipeline hash through Granite: an entry whose hash no longer matches (new shaders, a Granite or
// driver change) is skipped.

#include <atomic>
#include <cstdint>
#include <vector>

namespace Vulkan
{
    struct DeferredPipelineCompile;
    extern void (*ps2x_compute_compile_hook)(const DeferredPipelineCompile &compile);
}

namespace ParallelGS
{
    struct Ps2xPipelineVariant
    {
        uint64_t program = 0;  // Vulkan::Program::get_hash()
        uint64_t pipeline = 0; // DeferredPipelineCompile::hash
        uint32_t constants[8] = {};
        uint8_t mask = 0;
        uint8_t robustness = 0;
        uint8_t subgroupControl = 0, subgroupFull = 0, subgroupMinLog2 = 0, subgroupMaxLog2 = 0;
    };

    // Set before GSInterface::init.
    extern std::vector<Ps2xPipelineVariant> ps2x_recorded_variants;
    // Progress of GSRenderer::kick_compilation_tasks; `recorded`: entries of ps2x_recorded_variants
    // still valid and queued.
    extern std::atomic<uint32_t> ps2x_precompile_done;
    extern std::atomic<uint32_t> ps2x_precompile_total;
    extern std::atomic<uint32_t> ps2x_precompile_recorded;
}
