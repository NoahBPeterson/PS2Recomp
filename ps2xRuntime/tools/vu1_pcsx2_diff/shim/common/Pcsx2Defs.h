// Stand-in for PCSX2's common/Pcsx2Defs.h: just what VUops.cpp / VUflags.cpp / VU1microInterp.cpp
// use, so those files compile as the reference for ps2x_vu1_pcsx2_diff.
#pragma once
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t s8; typedef int16_t s16; typedef int32_t s32; typedef int64_t s64;
typedef uintptr_t uptr; typedef intptr_t sptr;
typedef unsigned int uint;
struct alignas(16) u128 { u64 lo, hi; };
struct alignas(16) s128 { s64 lo, hi; };
// PCSX2 defines __fi as a forced inline but also calls __fi functions across files (VUflags.cpp);
// plain functions here.
#define __fi
#define __ri
#define __noinline
#define pxAssert(x) ((void)0)
#define pxAssume(x) ((void)0)
#define pxAssertMsg(x, m) ((void)0)
#define pxFail(m) ((void)0)
#define jNO_DEFAULT default: break;
