#ifndef PS2_VU1_TRACE_H
#define PS2_VU1_TRACE_H
// VU1 run traces: every input of consecutive VU1 microprogram runs (MSCAL/MSCNT) and a digest of
// their outputs, for replaying the VU1 interpreter offline (ps2x_vu1_replay: benchmark + bit-exact
// regression check). Recorded by the runtime with PS2X_VU1_TRACE=<file>.
//
// File: "PS2XVU1T" u32 version, then records until EOF. Record:
//   Vu1TraceRecordHeader
//   u8 code[16384]                     only when header.codeIncluded (code changed since last record)
//   u8 data[16384]                     VU1 data memory before the run
//   VU1State before, VU1State after    raw structs (sizeof checked through header.stateSize)
//   u64 dataAfterHash                  FNV-1a of data memory after the run
//   u64 gifHash, u32 gifPackets, u32 gifBytes   PATH1 packets the run submitted (FNV-1a over all)
// Traces hold game data (disc-derived): keep them out of git.

#include "runtime/ps2_vu1.h"

#include <cstdint>
#include <cstring>

namespace ps2_vu1_trace
{
    constexpr char kMagic[8] = {'P', 'S', '2', 'X', 'V', 'U', '1', 'T'};
    constexpr uint32_t kVersion = 1u;
    constexpr uint32_t kMemSize = 0x4000u;

    enum class RunKind : uint32_t
    {
        Execute = 0u, // MSCAL / MSCALF: starts at startPC with fresh pipelines
        Resume = 1u,  // MSCNT: continues where the previous run stopped
    };

    struct Vu1TraceRecordHeader
    {
        uint32_t kind = 0u;
        uint32_t startPc = 0u;
        uint32_t top = 0u;
        uint32_t itop = 0u;
        uint32_t stateSize = sizeof(VU1State);
        uint8_t dBitEnabled = 0u;
        uint8_t tBitEnabled = 0u;
        uint8_t codeIncluded = 0u;
        uint8_t reserved = 0u;
    };

    inline uint64_t fnv1a(const void *data, size_t size, uint64_t hash = 0xcbf29ce484222325ull)
    {
        const auto *bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= bytes[i];
            hash *= 0x100000001b3ull;
        }
        return hash;
    }
}

#endif
