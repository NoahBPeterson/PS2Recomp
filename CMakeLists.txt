cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VITASDK})
    set(PS2X_VITA_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake")
    if(EXISTS "${PS2X_VITA_TOOLCHAIN_FILE}")
        set(CMAKE_TOOLCHAIN_FILE "${PS2X_VITA_TOOLCHAIN_FILE}" CACHE PATH
            "Toolchain file used for cross-compiling" FORCE)
        message(STATUS "Using VitaSDK toolchain from VITASDK: ${CMAKE_TOOLCHAIN_FILE}")
    endif()
endif()

project(PS2RetroX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PS2X_BUILD_RECOMP "Build ps2xRecomp" ON)
option(PS2X_BUILD_RUNTIME "Build ps2xRuntime" ON)
option(PS2X_BUILD_ANALYZER "Build ps2xAnalyzer" ON)
option(PS2X_BUILD_TEST "Build ps2xTest" ON)
option(PS2X_BUILD_STUDIO "Build ps2xStudio" ON)

if(ANDROID)
    message(STATUS "Android target detected, building runtime only")
    set(PS2X_BUILD_RECOMP OFF)
    set(PS2X_BUILD_ANALYZER OFF)
    set(PS2X_BUILD_TEST OFF)
    set(PS2X_BUILD_STUDIO OFF)
endif()

set(PS2X_IS_ARM_TARGET OFF)
set(PS2X_IS_AARCH64_TARGET OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(PS2X_IS_ARM_TARGET ON)
    set(PS2X_IS_AARCH64_TARGET ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm|^ARM")
    set(PS2X_IS_ARM_TARGET ON)
endif()

if((CMAKE_C_COMPILER MATCHES "arm-vita-eabi") OR
   (CMAKE_CXX_COMPILER MATCHES "arm-vita-eabi"))
    set(PS2X_IS_ARM_TARGET ON)
endif()

if(PS2X_IS_ARM_TARGET)
    message(STATUS "ARM target detected, fetching sse2neon")

    include(FetchContent)
    FetchContent_Declare(
        sse2neon
        GIT_REPOSITORY https://github.com/DLTcollab/sse2neon.git
        GIT_TAG v1.9.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(sse2neon)

    include_directories(${sse2neon_SOURCE_DIR})
    add_compile_definitions(USE_SSE2NEON)

    if(PS2X_IS_AARCH64_TARGET AND APPLE)
        # macOS ARM64 already uses optimal defaults
        message(STATUS "macOS ARM64 - using default compiler flags")
    elseif(PS2X_IS_AARCH64_TARGET AND MSVC)
        # Windows ARM64 - MSVC already uses optimal defaults
        message(STATUS "Windows ARM64 (MSVC) - using default compiler flags")
    elseif(PS2X_IS_AARCH64_TARGET)
        # Linux ARM64 - add NEON flags
        message(STATUS "Non-Apple ARM64 - adding NEON compiler flags")
        add_compile_options(-march=armv8-a+fp+simd)

        # Try to enable crypto and CRC extensions if supported
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=armv8-a+fp+simd+crypto+crc" COMPILER_SUPPORTS_CRYPTO_CRC)
        if(COMPILER_SUPPORTS_CRYPTO_CRC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+fp+simd+crypto+crc")
            message(STATUS "Crypto and CRC extensions enabled")
        else()
            add_compile_options(-march=armv8-a+fp+simd)
        endif()
    endif()
endif()

# Every float operation the code writes rounds once, the same on every platform: GCC/Clang would
# otherwise fuse a * b + c into one FMA on some targets (AArch64, x86 with -mfma) and not on others,
# while the PS2 rounds a VU or FPU multiply-add twice. (MSVC does not contract under /fp:precise.)
# Code that wants a fused operation says so (std::fma, vfmaq_*). Cost: ~3% of the CPU GS
# rasterizer's time on AArch64 (2026-10-16); in exchange its pixels are the same on every platform.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-ffp-contract=off)
endif()

if(PS2X_BUILD_RECOMP)
    add_subdirectory("ps2xRecomp")
endif()

if(PS2X_BUILD_RUNTIME)
    add_subdirectory("ps2xIOP")
    add_subdirectory("ps2xRuntime")
endif()

if(PS2X_BUILD_ANALYZER)
    add_subdirectory("ps2xAnalyzer")
endif()

if(PS2X_BUILD_TEST)
    add_subdirectory("ps2xTest")
endif()

if(PS2X_BUILD_STUDIO)
    add_subdirectory("ps2xStudio")
endif()

# Coverage of the runtime and IOP code (tools/conformance/coverage.sh): clang source-based coverage on
# the libraries and tools only, not on the generated game code (src/runner, thousands of files), so an
# instrumented game runner stays quick to build and the game's own run shows which runtime lines it uses.
option(PS2X_COVERAGE "Instrument ps2_runtime, ps2_iop and the test/tool executables for llvm-cov" OFF)
if(PS2X_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_ps2x_cov_flags -fprofile-instr-generate -fcoverage-mapping)
    foreach(_t ps2_runtime ps2_iop ps2_test_lib ps2x_tests ps2x_gs_replay ps2x_vu1_replay ps2x_vu1_rec_gen
               ps2x_vu1_pcsx2_diff)
        if(TARGET ${_t})
            target_compile_options(${_t} PRIVATE ${_ps2x_cov_flags})
            target_compile_definitions(${_t} PRIVATE PS2X_COVERAGE=1)
        endif()
    endforeach()
    # Every executable that links an instrumented library needs the profile runtime.
    foreach(_t ps2x_tests ps2x_gs_replay ps2x_vu1_replay ps2x_vu1_rec_gen ps2x_vu1_pcsx2_diff ps2EntryRunner
               ps2x_ee_oracle_wide ps2_gs_clut_cache_tests ps2_gs_depth_test_tests ps2_gs_memory_cache_tests
               ps2_gs_texture_cache_tests)
        if(TARGET ${_t})
            target_link_options(${_t} PRIVATE -fprofile-instr-generate)
        endif()
    endforeach()
endif()
