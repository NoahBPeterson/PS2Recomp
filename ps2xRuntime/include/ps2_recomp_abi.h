#pragma once
// Everything recompiled guest code needs from the runtime, without the runtime's class definition.
//
// Generated functions take a PS2Runtime* but only call it through the ps2x_abi:: wrappers below, so
// they compile against a forward declaration. Editing the GS/VU/IOP/audio/pad headers (or
// ps2_runtime.h itself) then rebuilds the runtime library, not the ~73k generated files (T-0001).
// Keep this header and what it includes small and stable.

#include <cstdint>

#include "ps2_cpu_context.h"
#include "runtime/ps2_address.h"

class PS2Runtime;

enum class Ps2GuestBranchKind
{
    DirectJump,
    DirectCall,
    IndirectJump,
    IndirectCall,
    Return,
};

namespace ps2x_abi
{
    uint8_t Load8(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint16_t Load16(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint32_t Load32(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    uint64_t Load64(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    __m128i Load128(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr);
    void Store8(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value);
    void Store16(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value);
    void Store32(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value);
    void Store64(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value);
    void Store128(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value);

    bool dispatchGuestBranch(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t targetPc,
                             uint32_t sourcePc, uint32_t fallthroughPc, Ps2GuestBranchKind kind,
                             const char *debugName);
    bool eeCheckpointDue(PS2Runtime *runtime, uint32_t cycles = 32u) noexcept;

    void SignalException(PS2Runtime *runtime, R5900Context *ctx, PS2Exception exception);
    void handleSyscall(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx);
    void handleSyscall(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId);
    void handleBreak(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx);
    void handleTrap(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx);
    void handleTLBR(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx);
    void handleTLBWI(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx);
    void handleTLBWR(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx);
    void handleTLBP(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx);
    void clearLLBit(PS2Runtime *runtime, R5900Context *ctx);

    void executeVU0Microprogram(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t address);
    void vu0StartMicroProgram(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t address);
    uint8_t *vu0Data(PS2Runtime *runtime);
    void kickGifDmaChainFromMMIO(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t dPcrValue,
                                 uint32_t dStatValue, uint32_t tadr, uint32_t chcr);
}
