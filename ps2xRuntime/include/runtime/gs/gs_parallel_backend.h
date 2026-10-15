#pragma once
#include "runtime/gs/gs_backend.h"

#include <memory>

struct GSRegisters;

// paraLLEl-GS (Vulkan compute) raster backend. Throws std::runtime_error when Vulkan or the
// required device features are unavailable; callers fall back to the CPU backend.
std::unique_ptr<GSRasterBackend> CreateParallelGSBackend(GSRegisters &registers);
