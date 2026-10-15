#pragma once

// PCSX2-compatible GS dump writer (format: tickets/notes/gs-dump-format.md).
//
// Opt-in through the environment:
//   PS2X_GS_DUMP=<file.gs>       output path (raw, uncompressed PCSX2 dump)
//   PS2X_GS_DUMP_START=<n>       first host frame (g_ps2xHostFrame) to record, default 0
//   PS2X_GS_DUMP_FRAMES=<n>      number of VSyncs to record, default 2
//   PS2X_GS_DUMP_SERIAL=<serial> serial stored in the header, default empty
//
// The GS frontend calls these hooks with its state mutex held. Only include this header
// from .cpp files: it is deliberately kept out of the ps2_runtime.h include chain.

#include <cstdint>

struct GSRegisters;
struct GSPresentationRequest;
class GSRasterBackend;

namespace ps2x_gs_dump
{
    // True while the writer wants hooks (PS2X_GS_DUMP set and recording not finished).
    bool active();

    // A GIF packet as submitted to the GS on PATH1/2/3.
    void transfer(uint32_t path, const uint8_t *data, uint32_t sizeBytes);
    // A register written outside any GIF packet (re-encoded as PATH3 A+D).
    void registerWrite(uint8_t addr, uint64_t value);
    // A HOST->LOCAL upload that bypassed the GIF (re-encoded as PATH3 A+D + IMAGE).
    void imageUpload(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, uint64_t trxdir,
                     const uint8_t *data, uint32_t sizeBytes);
    // A direct framebuffer clear (re-encoded as a PATH3 sprite; contextIndex < 0 = PRIM.CTXT).
    void clearFramebuffer(int contextIndex, uint32_t rgba);
    // LOCAL->HOST bytes drained by the EE.
    void readFifo(uint32_t bytes);
    // A presented frame; starts or ends the recording and writes Registers + VSync.
    void vsync(const GSRegisters *priv, const GSPresentationRequest &request, GSRasterBackend *backend);
}
