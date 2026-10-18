# paraLLEl-GS (https://github.com/Arntzen-Software/parallel-gs, LGPL-3.0) as the GS raster backend.
# Pinned to the commit upstream PR #285 uses. PS2X_PARALLEL_GS_SOURCE_DIR points at a local
# checkout instead (macOS needs Granite's timer.cpp clock_nanosleep fallback until upstreamed).
include(FetchContent)
set(PARALLEL_GS_STANDALONE ON CACHE BOOL "Build parallel-gs as a library" FORCE)
set(GRANITE_TOOLS OFF CACHE BOOL "" FORCE)
set(GRANITE_INSTALL_TARGETS OFF CACHE BOOL "" FORCE)

set(PS2X_PARALLEL_GS_SOURCE_DIR "" CACHE PATH "Local parallel-gs checkout (empty = fetch the pinned commit)")
if(PS2X_PARALLEL_GS_SOURCE_DIR)
    FetchContent_Declare(parallel_gs SOURCE_DIR "${PS2X_PARALLEL_GS_SOURCE_DIR}" EXCLUDE_FROM_ALL)
else()
    FetchContent_Declare(parallel_gs
        GIT_REPOSITORY https://github.com/Arntzen-Software/parallel-gs.git
        GIT_TAG 3a66c1976170cbc2cb53a3593fabbc7c4b2ccfbd
        GIT_SUBMODULES_RECURSE TRUE
        EXCLUDE_FROM_ALL)
endif()
FetchContent_MakeAvailable(parallel_gs)

include("${CMAKE_CURRENT_LIST_DIR}/ParallelGSTransferTails.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ParallelGSPrecompile.cmake")
target_sources(ps2_runtime PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/lib/gs/gs_parallel_backend.cpp")
target_link_libraries(ps2_runtime PRIVATE parallel-gs)
target_compile_definitions(ps2_runtime PUBLIC PS2X_GS_PARALLEL=1)
