// Minimal stand-in for PCSX2's common/Pcsx2Defs.h, so PCSX2 source files compile as test oracles.
#pragma once
#include <cstdint>
#include <cstring>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t s8; typedef int16_t s16; typedef int32_t s32; typedef int64_t s64;
typedef unsigned int uint;
#define __fi
#define pxAssert(x) ((void)0)
#define pxAssume(x) ((void)0)
