#include "ps2_recomp_abi.h"
#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#if defined(_MSC_VER)
#include <intrin.h>
#define PS2X_ABI_RETURN_ADDRESS() _ReturnAddress()
#else
#define PS2X_ABI_RETURN_ADDRESS() __builtin_return_address(0)
#endif
// ps2_host_write_trace.cpp
void ps2GuestWatchSpecial(uint32_t vaddr, uint32_t size, uint64_t valueLo, uint64_t valueHi, const R5900Context *ctx,
                          const void *hostPc);

namespace ps2x_abi
{
    uint8_t Load8(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load8(rdram, ctx, vaddr); }
    uint16_t Load16(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load16(rdram, ctx, vaddr); }
    uint32_t Load32(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load32(rdram, ctx, vaddr); }
    uint64_t Load64(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load64(rdram, ctx, vaddr); }
    __m128i Load128(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr) { return runtime->Load128(rdram, ctx, vaddr); }
    void Store8(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value) { runtime->Store8(rdram, ctx, vaddr, value); }
    void Store16(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value) { runtime->Store16(rdram, ctx, vaddr, value); }
    void Store32(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value) { runtime->Store32(rdram, ctx, vaddr, value); }
    // Special-address stores (scratchpad, hardware) bypass ps2TraceGuestWrite; PS2X_GUEST_WATCH
    // covers them here when it spans the address (scratchpad: 0x70000000 + offset).
    void Store64(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
    {
        if (g_ps2GuestWatchSpan != 0u) [[unlikely]]
            ps2GuestWatchSpecial(vaddr, 8u, value, 0u, ctx, PS2X_ABI_RETURN_ADDRESS());
        runtime->Store64(rdram, ctx, vaddr, value);
    }
    void Store128(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
    {
        if (g_ps2GuestWatchSpan != 0u) [[unlikely]]
            ps2GuestWatchSpecial(vaddr, 16u, static_cast<uint64_t>(PS2_EXTRACT_EPI64_0(value)),
                                 static_cast<uint64_t>(PS2_EXTRACT_EPI64_1(value)), ctx, PS2X_ABI_RETURN_ADDRESS());
        runtime->Store128(rdram, ctx, vaddr, value);
    }

    bool dispatchGuestBranch(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t targetPc,
                             uint32_t sourcePc, uint32_t fallthroughPc, Ps2GuestBranchKind kind,
                             const char *debugName)
    {
        return runtime->dispatchGuestBranch(rdram, ctx, targetPc, sourcePc, fallthroughPc, kind, debugName);
    }
    bool eeCheckpointDue(PS2Runtime *runtime, uint32_t cycles) noexcept { return runtime->eeCheckpointDue(cycles); }

    void SignalException(PS2Runtime *runtime, R5900Context *ctx, PS2Exception exception) { runtime->SignalException(ctx, exception); }
    void handleSyscall(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleSyscall(rdram, ctx); }
    void handleSyscall(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId) { runtime->handleSyscall(rdram, ctx, encodedSyscallId); }
    void handleBreak(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleBreak(rdram, ctx); }
    void handleTrap(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTrap(rdram, ctx); }
    void handleTLBR(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTLBR(rdram, ctx); }
    void handleTLBWI(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTLBWI(rdram, ctx); }
    void handleTLBWR(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTLBWR(rdram, ctx); }
    void handleTLBP(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx) { runtime->handleTLBP(rdram, ctx); }
    void clearLLBit(PS2Runtime *runtime, R5900Context *ctx) { runtime->clearLLBit(ctx); }

    void executeVU0Microprogram(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t address) { runtime->executeVU0Microprogram(rdram, ctx, address); }
    void vu0StartMicroProgram(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t address) { runtime->vu0StartMicroProgram(rdram, ctx, address); }
    uint8_t *vu0Data(PS2Runtime *runtime) { return runtime->memory().getVU0Data(); }
    void kickGifDmaChainFromMMIO(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t dPcrValue,
                                 uint32_t dStatValue, uint32_t tadr, uint32_t chcr)
    {
        runtime->kickGifDmaChainFromMMIO(rdram, ctx, dPcrValue, dStatValue, tadr, chcr);
    }
}
