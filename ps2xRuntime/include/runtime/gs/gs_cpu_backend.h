#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

class GSCpuBackend final : public GSRasterBackend
{
public:
    // rasterThreads <= 1: rasterize on the calling thread. N > 1: N worker threads rasterize
    // queued primitives, each owning interleaved bands of rows, so every pixel still sees the
    // primitives in submission order; the output is bit-identical to the single-threaded path.
    // asyncCommands: a dispatcher thread executes Submit/LoadClut/transfers/TEXFLUSH/clears in
    // order; the caller only waits in calls that need finished VRAM (Sync, Present, ReadVram,
    // SnapshotVram, local-to-host reads, ...). Flush does not wait in this mode.
    GSCpuBackend();
    explicit GSCpuBackend(uint32_t rasterThreads, bool asyncCommands = false);
    ~GSCpuBackend() override; // out of line: DecodedTexture is incomplete here

    // PS2X_GS_THREADS (default 3 with asynchronous commands, else 1; 1 = rasterize on the calling thread).
    static uint32_t DefaultRasterThreads();
    // On unless PS2X_GS_ASYNC=0.
    static bool DefaultAsyncCommands();

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    void ResetUnlocked();
    // directReads: read VRAM directly instead of through the texture page cache (only valid
    // when the two agree, see LoadClutQueuedUnlocked).
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut, bool directReads = false);
    // CLUT load while primitives are queued, without waiting for them; false when it must wait.
    bool LoadClutQueuedUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

public:
    // Decoded-texture cache: each texture is unswizzled and CLUT-resolved to RGBA8888 once
    // (lazily, per 16x16 tile) instead of on every sample. Entries are invalidated through
    // per-page VRAM write stamps, so uploads and render-to-texture are picked up.
    // (Public only so the rasterizer helpers in gs_cpu_backend.cpp can name it.)
    struct DecodedTexture;

private:
    void DrawPrimitive(const GSPrimitiveBatch &batch);
    // Waits until queued raster work is done and folds its results (written pages, texture
    // page cache state) back into the backend. Every path that touches VRAM or the texture
    // caches outside the rasterizer calls this first.
    void DrainUnlocked();
    void PrintParallelStats() const; // PS2X_GS_STATS: why primitives went to the workers or not
    void DrainUnlocked() const { const_cast<GSCpuBackend *>(this)->DrainUnlocked(); }
    void RetireTexture(std::unique_ptr<DecodedTexture> texture);
    // Producer-side VRAM access that may run while primitives are queued: drains first unless
    // the queued primitives neither write `reads` nor access `writes` (512-bit page sets).
    void DrainForAccessUnlocked(const std::array<uint64_t, 8> &reads, const std::array<uint64_t, 8> &writes);
    void StartWorkers(uint32_t rasterThreads);
    void StartAsync();

    DecodedTexture *PrepareDecodedTexture(const GSDrawState &state);
    void MarkVramWritten(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    // Stamps every page set in `pages` (a 512-bit set) as written, as if by one VRAM write.
    void MarkPagesWritten(const std::array<uint64_t, 8> &pages);
    uint32_t LookupCLUT(const GSTexaReg &texa, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm) const;
    // palette[i] = LookupCLUT(..., i, ...) for the texture in `tex0`.
    void BuildPalette(const GSTex0Reg &tex0, const GSTexaReg &texa, uint32_t *palette) const;

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    std::array<uint16_t, 512> m_clut{};
    std::array<uint32_t, 2> m_clutCbp{};
    GSMem::TexturePageCache m_texturePageCache;
    std::array<uint64_t, 512> m_pageWriteStamp{};
    // Pages written since the last TEXFLUSH. Textures decoded meanwhile read the stale texture
    // page buffer, so TEXFLUSH re-stamps these pages to force a re-decode.
    std::array<bool, 512> m_pageWrittenSinceFlush{};
    uint64_t m_vramWriteStamp = 1;
    uint64_t m_clutVersion = 1;
    std::unordered_map<uint64_t, std::unique_ptr<DecodedTexture>> m_decodedTextures;
    size_t m_decodedTextureBytes = 0;
    bool m_decodedTexturesDisabled = false;

    // Worker pool and the state of the current batch of queued primitives (gs_cpu_backend.cpp).
    struct Parallel;
    std::unique_ptr<Parallel> m_parallel;
    struct AsyncQueue;
    std::unique_ptr<AsyncQueue> m_async;
    bool OnAsyncCaller() const; // async mode and not on the dispatcher thread
    void WaitAsyncIdle() const;
    uint32_t m_primSeq = 0;      // sequence number of the last primitive (decoded-tile epochs)
    bool m_cacheDirty = false;   // the texture page cache's page may have been written since it was copied
    std::vector<std::unique_ptr<DecodedTexture>> m_retiredTextures; // freed when the queue drains

    std::array<uint64_t, 8> m_transferPages{}; // pages the current host-to-local transfer writes
    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};
