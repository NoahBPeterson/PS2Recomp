#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <condition_variable>
#include <chrono>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <type_traits>

using namespace GSInternal;

namespace
{
    float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    // The GS denormalises each perspective-divided texel coordinate into 12.4
    // fixed point and saturates it before the wrap mode runs (libgpu2, Sony's
    // GS model: TexCoordN in src/txm.c, then WrapU/WrapV). A positive overflow
    // becomes 0x7FFF; a negative overflow becomes zero, not -0x7FFF. The same
    // bound keeps the static_cast<int> below defined: converting a NaN or an
    // out-of-range float is undefined behaviour, and games such as Ridge Racer
    // V emit ST values that reach it. NaN has no hardware meaning and maps to
    // zero rather than to the host's default NaN sign, so hosts agree.
    constexpr float kGsTexCoordOverflow = 2048.0f; // 0x8000 in 12.4
    constexpr float kGsTexCoordMax = 2047.9375f;   // 0x7FFF in 12.4

    float saturateTexCoord(float coordinate)
    {
        if (std::isnan(coordinate))
            return 0.0f;
        if (coordinate >= kGsTexCoordOverflow)
            return kGsTexCoordMax;
        if (coordinate <= -kGsTexCoordOverflow)
            return 0.0f;
        return coordinate;
    }

    u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    [[gnu::always_inline]] inline TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    bool isFourBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    bool isEightBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
    }

    uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, float fx, float fy)
    {
        const float top = static_cast<float>(c00) + (static_cast<float>(c10) - static_cast<float>(c00)) * fx;
        const float bottom = static_cast<float>(c01) + (static_cast<float>(c11) - static_cast<float>(c01)) * fx;
        return clampU8(static_cast<int>(std::lround(top + (bottom - top) * fy)));
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }
}

GSCpuBackend::GSCpuBackend() : GSCpuBackend(1u, false)
{
}

GSCpuBackend::GSCpuBackend(uint32_t rasterThreads, bool asyncCommands)
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    Reset();
    StartWorkers(rasterThreads);
    if (asyncCommands)
        StartAsync();
}

namespace
{
    constexpr uint32_t kGsPageCount = 512u; // 4 MiB of VRAM in 8 KiB pages
    constexpr uint32_t kDecodedTileShift = 4u; // 16x16 texel tiles
    constexpr uint32_t kTileBusy = UINT32_MAX;
    constexpr uint32_t kMaxRasterThreads = 16u;
    constexpr size_t kDecodedTextureBudgetBytes = 256u * 1024u * 1024u;

    // Page dimensions in pixels for a storage format (GS page = 8 KiB).
    void gsPageSize(uint8_t psm, uint32_t &width, uint32_t &height)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            width = 64u;
            height = 64u;
            return;
        case GS_PSM_T8:
            width = 128u;
            height = 64u;
            return;
        case GS_PSM_T4:
            width = 128u;
            height = 128u;
            return;
        default: // 32-bit layouts, including T8H/T4HL/T4HH stored in CT32 words
            width = 64u;
            height = 32u;
            return;
        }
    }

    uint32_t gsPagesPerRow(uint8_t psm, uint32_t bw)
    {
        uint32_t pageWidth = 0u, pageHeight = 0u;
        gsPageSize(psm, pageWidth, pageHeight);
        return std::max<uint32_t>(1u, (bw * 64u) / pageWidth);
    }

    // Contiguous (wrapping) page range covering a width x height region at block base `bp`.
    void gsRegionPages(uint8_t psm, uint32_t bp, uint32_t bw, uint32_t width, uint32_t height,
                       uint32_t &firstPage, uint32_t &pageCount)
    {
        uint32_t pageWidth = 0u, pageHeight = 0u;
        gsPageSize(psm, pageWidth, pageHeight);
        const uint32_t rows = (std::max<uint32_t>(height, 1u) + pageHeight - 1u) / pageHeight;
        const uint32_t cols = (std::max<uint32_t>(width, 1u) + pageWidth - 1u) / pageWidth;
        const uint64_t span = static_cast<uint64_t>(rows - 1u) * gsPagesPerRow(psm, bw) + cols + ((bp & 31u) ? 1u : 0u);
        firstPage = (bp >> 5u) % kGsPageCount;
        pageCount = static_cast<uint32_t>(std::min<uint64_t>(span, kGsPageCount));
    }

    bool gsPageRangesOverlap(uint32_t firstA, uint32_t countA, uint32_t firstB, uint32_t countB)
    {
        if (countA >= kGsPageCount || countB >= kGsPageCount)
            return true;
        const uint32_t offset = (firstB + kGsPageCount - firstA) % kGsPageCount; // B's start relative to A
        return offset < countA || (kGsPageCount - offset) < countB;
    }
}

namespace
{
    // False when the texture may overlap the frame or depth buffer being drawn: sampling the
    // surface being drawn (feedback) must see each new pixel, so it uses the direct path.
    bool decodedTextureAllowed(const GSDrawState &state)
    {
        const GSContext &ctx = state.context;
        const GSTex0Reg &tex = ctx.tex0;
        uint32_t firstPage = 0u, pageCount = 0u;
        gsRegionPages(tex.psm, tex.tbp0, tex.tbw, state.textureWidth, state.textureHeight, firstPage, pageCount);
        uint32_t frameFirst = 0u, frameCount = 0u;
        const uint32_t drawHeight = static_cast<uint32_t>(ctx.scissor.y1) + 1u;
        gsRegionPages(ctx.frame.psm, framePageBaseToBlock(ctx.frame.fbp), ctx.frame.fbw, ctx.frame.fbw * 64u, drawHeight,
                      frameFirst, frameCount);
        if (gsPageRangesOverlap(firstPage, pageCount, frameFirst, frameCount))
            return false;
        if (!ctx.zbuf.zmask)
        {
            uint32_t zFirst = 0u, zCount = 0u;
            gsRegionPages(ctx.zbuf.psm, framePageBaseToBlock(ctx.zbuf.zbp), ctx.frame.fbw, ctx.frame.fbw * 64u, drawHeight,
                          zFirst, zCount);
            if (gsPageRangesOverlap(firstPage, pageCount, zFirst, zCount))
                return false;
        }
        return true;
    }
}

struct GSCpuBackend::DecodedTexture
{
    // Identity: everything DecodeTexel reads.
    uint32_t tbp0 = 0;
    uint8_t tbw = 0, psm = 0, tw = 0, th = 0;
    uint32_t cbp = 0;
    uint8_t cpsm = 0, csm = 0, csa = 0;
    uint8_t ta0 = 0, ta1 = 0;
    bool aem = false;
    uint64_t clutVersion = 0; // 0 for direct-colour formats

    uint32_t width = 0, height = 0, tilesX = 0;
    uint32_t firstPage = 0, pageCount = 0;
    uint64_t decodeStamp = 0;    // tiles reflect VRAM as of this write stamp
    uint64_t validatedStamp = 0; // page stamps last checked at this write stamp
    std::vector<uint32_t> texels;
    // Per 16x16 tile: 0 = not decoded, kTileBusy = being decoded, else the sequence number of
    // the primitive that decoded it (lets parallel workers find tiles decoded "in this primitive").
    uint32_t tileCount = 0;
    std::unique_ptr<std::atomic<uint32_t>[]> tileEpoch;
    // Parallel mode: per worker, the last primitive in which it touched each tile.
    std::unique_ptr<uint32_t[]> touched[kMaxRasterThreads];
    std::array<uint32_t, 256> palette{}; // indexed formats: the CLUT (fixed by clutVersion)

    bool matches(const GSTex0Reg &tex, const GSTexaReg &texa, uint64_t clut) const
    {
        return tbp0 == tex.tbp0 && tbw == tex.tbw && psm == tex.psm && tw == tex.tw && th == tex.th &&
               cbp == tex.cbp && cpsm == tex.cpsm && csm == tex.csm && csa == tex.csa &&
               ta0 == texa.ta0 && ta1 == texa.ta1 && aem == texa.aem && clutVersion == clut;
    }
};



// ---------------------------------------------------------------------------------------------
// Asynchronous command execution (PS2X_GS_ASYNC=1): the caller's commands are queued and a
// dispatcher thread runs them, in order, through the same public entry points. Calls that hand
// back VRAM-derived data wait for the queue to empty first.
// ---------------------------------------------------------------------------------------------
namespace
{
    thread_local const GSCpuBackend *t_asyncDispatching = nullptr;

    struct AsyncCommand
    {
        enum Kind : uint8_t
        {
            Submit,
            LoadClut,
            TextureFlush,
            BeginTransfer,
            Upload,
            Clear,
            Snapshot, // copy VRAM for presentation (AsyncQueue::snapshot)
        } kind = Submit;
        GSPrimitiveBatch batch{};
        GSTex0Reg tex0{};
        GSTexClutReg texclut{};
        GSTransferCommand transfer{};
        GSContext context{};
        uint32_t rgba = 0;
        std::vector<uint8_t> data;
        GSPresentationRequest request{};
    };

    constexpr size_t kAsyncQueueLimit = 32768u; // commands in flight before the caller waits
}

struct GSCpuBackend::AsyncQueue
{
    std::mutex mutex;
    std::condition_variable wake;  // dispatcher: work arrived / stop
    std::condition_variable idleCv; // callers: commands done / room again
    std::vector<AsyncCommand> pending;
    // Commands pushed / executed so far. A caller waits for the commands pushed before its call,
    // not for an empty queue: the presentation thread would otherwise starve while the EE keeps
    // the dispatcher busy.
    uint64_t pushedCount = 0;
    uint64_t doneCount = 0;
    bool stop = false;
    bool dispatcherSleeping = false;
    std::thread thread;
    // Presentation does not wait for the queue: Present queues a Snapshot (with the display
    // registers read at that moment) and shows the newest finished one, so each presented frame
    // pairs registers and VRAM from the same point of the command stream, as in synchronous mode.
    std::vector<uint8_t> snapshot;
    GSPresentationRequest snapshotRequest{};
    uint64_t snapshotGeneration = 0; // finished snapshots
    bool snapshotQueued = false;

    void push(AsyncCommand &&command)
    {
        std::unique_lock<std::mutex> lock(mutex);
        pushLocked(lock, std::move(command));
    }

    void pushLocked(std::unique_lock<std::mutex> &lock, AsyncCommand &&command)
    {
        if (pending.size() >= kAsyncQueueLimit)
            idleCv.wait(lock, [&]
                        { return pending.size() < kAsyncQueueLimit / 2u; });
        pending.push_back(std::move(command));
        ++pushedCount;
        if (dispatcherSleeping)
            wake.notify_one();
    }

    void waitIdle()
    {
        std::unique_lock<std::mutex> lock(mutex);
        const uint64_t target = pushedCount;
        idleCv.wait(lock, [&]
                    { return doneCount >= target; });
    }

    void run(GSCpuBackend &backend)
    {
        t_asyncDispatching = &backend;
        std::vector<AsyncCommand> batch;
        for (;;)
        {
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (!batch.empty())
                {
                    doneCount += batch.size();
                    batch.clear();
                    idleCv.notify_all();
                }
                dispatcherSleeping = true;
                wake.wait(lock, [&]
                          { return !pending.empty() || stop; });
                dispatcherSleeping = false;
                if (pending.empty() && stop)
                    return;
                batch.swap(pending);
                idleCv.notify_all(); // room in the queue again
            }
            for (AsyncCommand &command : batch)
            {
                switch (command.kind)
                {
                case AsyncCommand::Submit:
                    backend.Submit(command.batch);
                    break;
                case AsyncCommand::LoadClut:
                    backend.LoadClut(command.tex0, command.texclut);
                    break;
                case AsyncCommand::TextureFlush:
                    backend.TextureFlush();
                    break;
                case AsyncCommand::BeginTransfer:
                    backend.BeginTransfer(command.transfer);
                    break;
                case AsyncCommand::Upload:
                    backend.UploadImage(command.data.data(), static_cast<uint32_t>(command.data.size()));
                    break;
                case AsyncCommand::Clear:
                    backend.ClearFramebuffer(command.context, command.rgba);
                    break;
                case AsyncCommand::Snapshot:
                {
                    std::vector<uint8_t> copy;
                    backend.SnapshotVram(copy);
                    std::lock_guard<std::mutex> lock(mutex);
                    snapshot.swap(copy);
                    snapshotRequest = command.request;
                    ++snapshotGeneration;
                    snapshotQueued = false;
                    idleCv.notify_all();
                    break;
                }
                }
            }
            // (counted as done and cleared under the lock at the top of the loop)
        }
    }
};

bool GSCpuBackend::DefaultAsyncCommands()
{
    const char *env = std::getenv("PS2X_GS_ASYNC");
    return env && env[0] == '1';
}

void GSCpuBackend::StartAsync()
{
    m_async = std::make_unique<AsyncQueue>();
    m_async->thread = std::thread([this]()
                                  { m_async->run(*this); });
}

bool GSCpuBackend::OnAsyncCaller() const
{
    return m_async && t_asyncDispatching != this;
}

void GSCpuBackend::WaitAsyncIdle() const
{
    if (OnAsyncCaller())
        m_async->waitIdle();
}

void GSCpuBackend::MarkVramWritten(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    uint32_t pageWidth = 0u, pageHeight = 0u;
    gsPageSize(static_cast<uint8_t>(psm), pageWidth, pageHeight);
    const uint32_t page = (base >> 5u) + (y / pageHeight) * gsPagesPerRow(static_cast<uint8_t>(psm), bw) + x / pageWidth;
    const uint64_t stamp = ++m_vramWriteStamp;
    if (m_texturePageCache.PageBase() != UINT32_MAX)
    {
        const uint32_t cached = m_texturePageCache.PageBase() / GSMem::TexturePageCache::kPageSize;
        if (cached == page % kGsPageCount || ((base & 31u) && cached == (page + 1u) % kGsPageCount))
            m_cacheDirty = true;
    }
    m_pageWriteStamp[page % kGsPageCount] = stamp;
    m_pageWrittenSinceFlush[page % kGsPageCount] = true;
    if (base & 31u) // block-offset base: the write can land in the following page
    {
        m_pageWriteStamp[(page + 1u) % kGsPageCount] = stamp;
        m_pageWrittenSinceFlush[(page + 1u) % kGsPageCount] = true;
    }
}

GSCpuBackend::DecodedTexture *GSCpuBackend::PrepareDecodedTexture(const GSDrawState &state)
{
    if (m_decodedTexturesDisabled || !state.prim.tme)
        return nullptr;

    const GSContext &ctx = state.context;
    const GSTex0Reg &tex = ctx.tex0;
    const bool indexed = isFourBitIndexedPsm(tex.psm) || isEightBitIndexedPsm(tex.psm);
    const uint64_t clutVersion = indexed ? m_clutVersion : 0u;
    const uint32_t width = state.textureWidth;
    const uint32_t height = state.textureHeight;

    uint32_t firstPage = 0u, pageCount = 0u;
    gsRegionPages(tex.psm, tex.tbp0, tex.tbw, width, height, firstPage, pageCount);
    if (!decodedTextureAllowed(state))
        return nullptr;

    uint64_t key = tex.tbp0;
    key = key * 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(tex.tbw) | (static_cast<uint64_t>(tex.psm) << 8) |
                                         (static_cast<uint64_t>(tex.tw) << 16) | (static_cast<uint64_t>(tex.th) << 24));
    key = key * 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(tex.cbp) | (static_cast<uint64_t>(tex.cpsm) << 32) |
                                         (static_cast<uint64_t>(tex.csm) << 40) | (static_cast<uint64_t>(tex.csa) << 48));
    key = key * 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(state.texa.ta0) | (static_cast<uint64_t>(state.texa.ta1) << 8) |
                                         (static_cast<uint64_t>(state.texa.aem) << 16));
    key = key * 0x9E3779B97F4A7C15ull ^ clutVersion;

    auto found = m_decodedTextures.find(key);
    if (found == m_decodedTextures.end() || !found->second->matches(tex, state.texa, clutVersion))
    {
        if (found != m_decodedTextures.end())
        {
            m_decodedTextureBytes -= found->second->texels.size() * sizeof(uint32_t);
            RetireTexture(std::move(found->second)); // queued primitives may still sample it
            m_decodedTextures.erase(found);
        }
        const size_t bytes = static_cast<size_t>(width) * height * sizeof(uint32_t);
        if (m_decodedTextureBytes + bytes > kDecodedTextureBudgetBytes)
        {
            for (auto &entry : m_decodedTextures)
                RetireTexture(std::move(entry.second));
            m_decodedTextures.clear();
            m_decodedTextureBytes = 0u;
        }
        auto texture = std::make_unique<DecodedTexture>();
        texture->tbp0 = tex.tbp0;
        texture->tbw = tex.tbw;
        texture->psm = tex.psm;
        texture->tw = tex.tw;
        texture->th = tex.th;
        texture->cbp = tex.cbp;
        texture->cpsm = tex.cpsm;
        texture->csm = tex.csm;
        texture->csa = tex.csa;
        texture->ta0 = state.texa.ta0;
        texture->ta1 = state.texa.ta1;
        texture->aem = state.texa.aem;
        texture->clutVersion = clutVersion;
        texture->width = width;
        texture->height = height;
        texture->tilesX = (width + (1u << kDecodedTileShift) - 1u) >> kDecodedTileShift;
        const uint32_t tilesY = (height + (1u << kDecodedTileShift) - 1u) >> kDecodedTileShift;
        texture->firstPage = firstPage;
        texture->pageCount = pageCount;
        texture->texels.resize(static_cast<size_t>(width) * height);
        texture->tileCount = texture->tilesX * tilesY;
        texture->tileEpoch.reset(new std::atomic<uint32_t>[texture->tileCount]());
        texture->decodeStamp = m_vramWriteStamp;
        texture->validatedStamp = m_vramWriteStamp;
        if (indexed)
            BuildPalette(tex, state.texa, texture->palette.data());
        m_decodedTextureBytes += bytes;
        DecodedTexture *raw = texture.get();
        m_decodedTextures[key] = std::move(texture);
        return raw;
    }

    DecodedTexture &texture = *found->second;
    if (texture.validatedStamp != m_vramWriteStamp)
    {
        uint64_t newest = 0u;
        for (uint32_t i = 0; i < texture.pageCount; ++i)
            newest = std::max(newest, m_pageWriteStamp[(texture.firstPage + i) % kGsPageCount]);
        if (newest > texture.decodeStamp)
        {
            for (uint32_t i = 0; i < texture.tileCount; ++i)
                texture.tileEpoch[i].store(0u, std::memory_order_relaxed);
            texture.decodeStamp = m_vramWriteStamp;
        }
        texture.validatedStamp = m_vramWriteStamp;
    }
    return &texture;
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    WaitAsyncIdle();
    if (vram && vramSize < GSMem::MEMORY_SIZE)
        throw std::invalid_argument("GS CPU backend requires at least 4 MiB of VRAM");

    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
}

void GSCpuBackend::Reset()
{
    WaitAsyncIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    m_clut.fill(0u);
    m_clutCbp.fill(0u);
    m_texturePageCache.Invalidate();
    m_cacheDirty = false;
    m_decodedTextures.clear();
    m_decodedTextureBytes = 0u;
    ++m_clutVersion;
    // PS2X_GS_NO_TEXCACHE=1 disables the decoded-texture cache (A/B checks against the direct path).
    m_decodedTexturesDisabled = std::getenv("PS2X_GS_NO_TEXCACHE") != nullptr;
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    if (OnAsyncCaller())
    {
        if (batch.vertexCount == 0u)
            return;
        AsyncCommand command;
        command.kind = AsyncCommand::Submit;
        command.batch = batch;
        m_async->push(std::move(command));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || batch.vertexCount == 0u)
        return;
    DrawPrimitive(batch);
}

void GSCpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    if (OnAsyncCaller())
    {
        AsyncCommand command;
        command.kind = AsyncCommand::LoadClut;
        command.tex0 = tex0;
        command.texclut = texclut;
        m_async->push(std::move(command));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    if (LoadClutQueuedUnlocked(tex0, texclut))
        return;
    DrainUnlocked();
    LoadClutUnlocked(tex0, texclut);
}

namespace
{
    // Source pixel of CLUT entry `entry` (LoadClutUnlocked's addressing).
    void clutEntrySource(const GSTex0Reg &tex0, const GSTexClutReg &texclut, uint32_t entry,
                         uint32_t &sourceX, uint32_t &sourceY, uint32_t &sourceWidth)
    {
        sourceX = 0u;
        sourceY = 0u;
        sourceWidth = 1u;
        if (tex0.csm == 0u)
        {
            const uint32_t sourceIndex = swizzleClutIndexCSM1(entry);
            sourceX = sourceIndex & 0x0Fu;
            sourceY = sourceIndex >> 4u;
        }
        else
        {
            sourceWidth = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            sourceX = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
            sourceY = static_cast<uint32_t>(texclut.cov);
        }
    }
}

void GSCpuBackend::LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut, bool directReads)
{
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;

    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;
    const std::array<uint16_t, 512> previousClut = m_clut;
    const uint32_t cacheBefore = m_texturePageCache.PageBase();

    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u, sourceY = 0u, sourceWidth = 1u;
        clutEntrySource(tex0, texclut, entry, sourceX, sourceY, sourceWidth);
        const uint32_t raw = directReads ? ReadVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY)
                                         : ReadTextureVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY);
        const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
        if (sixteenBit)
        {
            m_clut[destination] = static_cast<uint16_t>(raw);
        }
        else
        {
            m_clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
            m_clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
        }
    }
    if (!directReads && m_texturePageCache.PageBase() != cacheBefore)
        m_cacheDirty = false; // freshly copied (no writes happen during a CLUT load)
    if (m_clut != previousClut)
        ++m_clutVersion;
}

void GSCpuBackend::Flush()
{
    if (OnAsyncCaller())
        return; // queued work proceeds on its own; Sync and readbacks wait for it
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
}

void GSCpuBackend::TextureFlush()
{
    if (OnAsyncCaller())
    {
        AsyncCommand command;
        command.kind = AsyncCommand::TextureFlush;
        m_async->push(std::move(command));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    m_texturePageCache.Invalidate();
    m_cacheDirty = false;
    const uint64_t stamp = ++m_vramWriteStamp;
    for (size_t page = 0; page < m_pageWrittenSinceFlush.size(); ++page)
    {
        if (m_pageWrittenSinceFlush[page])
        {
            m_pageWriteStamp[page] = stamp;
            m_pageWrittenSinceFlush[page] = false;
        }
    }
}

void GSCpuBackend::Sync(GSSyncReason reason)
{
    if (reason == GSSyncReason::Presentation && OnAsyncCaller())
        return; // Present takes a snapshot in order instead (AsyncQueue::snapshot)
    WaitAsyncIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    WaitAsyncIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    return ReadVramUnlocked(psm, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    if (!m_vram)
        return 0u;

    return GSMem::ReadTexture(m_texturePageCache, m_vram, psm, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    WaitAsyncIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    WriteVramUnlocked(psm, base, bw, x, y, value);
}

void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
    MarkVramWritten(psm, base, bw, x, y);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    WaitAsyncIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    WaitAsyncIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

uint32_t GSCpuBackend::LookupCLUT(const GSTexaReg &texa,
                                  uint8_t index,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm) const
{
    const bool sixteenBit = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;
    const uint32_t sourceIndex = isFourBitIndexedPsm(sourcePsm)
                                     ? (static_cast<uint32_t>(index) & 0x0Fu)
                                     : static_cast<uint32_t>(index);

    uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    if (!sixteenBit && csm == 0u && isEightBitIndexedPsm(sourcePsm))
    {
        const uint32_t block = std::min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }

    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        const uint32_t raw = static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
        return applyTexa(texa, cpsm, raw);
    }
    case GS_PSM_CT24:
    {
        const uint32_t raw = static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
        return applyTexa(texa, cpsm, raw & 0x00FFFFFFu);
    }
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return applyTexa(texa, cpsm, Rgba5551ToRgba8888(m_clut[clutIndex]));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

// ---------------------------------------------------------------------------------------------
// Rasterizer.
//
// Everything below must stay bit-exact with the straightforward per-pixel implementation it
// replaced (kept in git history): the same float expressions in the same form (clang contracts
// `a * b + c` into fused multiply-adds per expression, so hoisting a whole product out of an
// expression is safe but re-associating is not), the same VRAM reads and writes in the same
// order, and the same texture page cache accesses in the same order. Per-primitive state is
// decoded once into PixelSetup/TexSetup, and the VRAM swizzle is inlined for the common
// framebuffer/depth formats through templates.
// ---------------------------------------------------------------------------------------------
namespace
{
    using GSMem::PixelStorageMode;
    using DecodedTexture = GSCpuBackend::DecodedTexture;
    using ReadFn = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
    using WriteFn = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

    constexpr int kGenericPsm = -1;

    // MarkVramWritten's page arithmetic for one (psm, base, bw), precomputed.
    struct PageMark
    {
        uint32_t basePage = 0u;
        uint32_t shiftX = 0u, shiftY = 0u;
        uint32_t pagesPerRow = 1u;
        bool plusOne = false;
    };

    uint32_t log2Pow2(uint32_t v)
    {
        uint32_t s = 0u;
        while ((1u << s) < v)
            ++s;
        return s;
    }

    PageMark makePageMark(uint8_t psm, uint32_t base, uint32_t bw)
    {
        uint32_t pageWidth = 0u, pageHeight = 0u;
        gsPageSize(psm, pageWidth, pageHeight);
        PageMark m;
        m.basePage = base >> 5u;
        m.shiftX = log2Pow2(pageWidth);
        m.shiftY = log2Pow2(pageHeight);
        m.pagesPerRow = gsPagesPerRow(psm, bw);
        m.plusOne = (base & 31u) != 0u;
        return m;
    }

    struct TexSetup;

    // First texel fetch, in raster order, that a parallel worker made from a tile decoded
    // during the current primitive (see Parallel::resolvePageCache).
    struct TileTouch
    {
        uint32_t seq;
        DecodedTexture *texture;
        uint32_t tile;
        uint64_t key; // raster order of the fetch: (y, x, fetch index)
    };

    // Raster context: VRAM, the texture page cache, the rows this rasterizer owns, and the set
    // of pages written (stamped once when the primitive or the parallel batch ends; nothing
    // reads the stamps inside a primitive).
    struct RasterCtx
    {
        uint8_t *vram = nullptr;
        GSMem::TexturePageCache *pageCache = nullptr; // null in parallel workers: direct VRAM reads
        std::array<uint64_t, 8> written{};
        bool anyWritten = false;

        // Row ownership: rows y with ((y >> bandShift) % bandCount) == bandIndex.
        uint32_t bandShift = 0, bandCount = 1, bandIndex = 0;
        uint32_t curSeq = 0;           // sequence number of the primitive being drawn
        uint64_t pixelKey = 0;         // raster order of the current pixel (parallel mode)
        uint32_t worker = 0;
        std::vector<TileTouch> *touches = nullptr; // parallel mode only
        // Parallel mode: the last direct-path (page cache) fetch of the latest primitive that
        // made one, in raster order: the serial path's page cache ends on its page.
        struct DirectFetch
        {
            uint32_t seq = 0;
            uint64_t key = 0;
            int u = 0, v = 0;
            const TexSetup *ts = nullptr;
        } lastDirect;

        inline bool owns(int y) const
        {
            return bandCount == 1u || ((static_cast<uint32_t>(y) >> bandShift) % bandCount) == bandIndex;
        }

        inline void mark(const PageMark &m, uint32_t x, uint32_t y)
        {
            const uint32_t page = m.basePage + (y >> m.shiftY) * m.pagesPerRow + (x >> m.shiftX);
            const uint32_t p0 = page % kGsPageCount;
            written[p0 >> 6u] |= 1ull << (p0 & 63u);
            if (m.plusOne)
            {
                const uint32_t p1 = (page + 1u) % kGsPageCount;
                written[p1 >> 6u] |= 1ull << (p1 & 63u);
            }
            anyWritten = true;
        }
    };

    struct PixelSetup
    {
        int sx0 = 0, sx1 = 0, sy0 = 0, sy1 = 0;
        bool fge = false;
        uint8_t fogR = 0, fogG = 0, fogB = 0;
        uint32_t fbp = 0, fbw = 1, zbp = 0;
        uint8_t fpsm = 0, zpsm = 0;
        bool fb16 = false; // bitsPerPixel(fpsm) == 16
        bool fb24 = false; // fpsm == CT24
        bool fb32 = false; // fpsm == CT32
        // TEST
        bool ate = false;
        uint8_t atst = 0, aref = 0;
        PixelWriteMask failMask{};
        bool dateRead = false; // DATE enabled on a format with destination alpha
        bool datm = false;
        bool ztestEnabled = false;
        uint32_t ztestMethod = 1u;
        // ALPHA / PABE / FBA / FRAME / ZBUF
        bool abe = false, pabe = false;
        uint8_t asel = 0, bsel = 0, csel = 0, dsel = 0, fix = 0;
        bool fbaOr = false;
        uint32_t fbmsk = 0;
        bool zwrite = false; // ZTE && !ZMSK
        PageMark fbMark, zMark;
        ReadFn fbRead = nullptr, zRead = nullptr;
        WriteFn fbWrite = nullptr, zWrite = nullptr;
    };

    PixelSetup makePixelSetup(const GSDrawState &state,
                              const std::array<ReadFn, 64> &readFuncs,
                              const std::array<WriteFn, 64> &writeFuncs)
    {
        const auto &ctx = state.context;
        PixelSetup ps;
        ps.sx0 = ctx.scissor.x0;
        ps.sx1 = ctx.scissor.x1;
        ps.sy0 = ctx.scissor.y0;
        ps.sy1 = ctx.scissor.y1;
        ps.fge = state.prim.fge;
        ps.fogR = state.fogR;
        ps.fogG = state.fogG;
        ps.fogB = state.fogB;
        ps.fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
        ps.fbw = std::max<u32>(ctx.frame.fbw, 1u);
        ps.fpsm = ctx.frame.psm;
        ps.zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
        ps.zpsm = ctx.zbuf.psm;
        ps.fb16 = bitsPerPixel(ps.fpsm) == 16;
        ps.fb24 = ps.fpsm == GS_PSM_CT24;
        ps.fb32 = ps.fpsm == GS_PSM_CT32;

        const uint64_t test = ctx.test;
        ps.ate = (test & 0x1u) != 0u;
        ps.atst = static_cast<uint8_t>((test >> 1) & 0x7u);
        ps.aref = static_cast<uint8_t>((test >> 4) & 0xFFu);
        switch (static_cast<uint8_t>((test >> 12) & 0x3u))
        {
        case 1:
            ps.failMask = {true, true, false};
            break;
        case 2:
            ps.failMask = {false, false, true};
            break;
        case 3:
            ps.failMask = ps.fpsm == GS_PSM_CT32 ? PixelWriteMask{true, false, false} : PixelWriteMask{true, true, false};
            break;
        default:
            ps.failMask = {false, false, false};
            break;
        }
        const bool date = ((test >> 14) & 0x1u) != 0u;
        ps.dateRead = date && (ps.fpsm == GS_PSM_CT32 || ps.fpsm == GS_PSM_CT16 || ps.fpsm == GS_PSM_CT16S);
        ps.datm = ((test >> 15) & 0x1u) != 0u;
        ps.ztestEnabled = ((test >> 16) & 1u) != 0u;
        ps.ztestMethod = ps.ztestEnabled ? static_cast<uint32_t>((test >> 17) & 3u) : 1u;

        ps.abe = state.prim.abe;
        ps.pabe = state.pabe;
        const uint64_t alphaReg = ctx.alpha;
        ps.asel = alphaReg & 3;
        ps.bsel = (alphaReg >> 2) & 3;
        ps.csel = (alphaReg >> 4) & 3;
        ps.dsel = (alphaReg >> 6) & 3;
        ps.fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);
        ps.fbaOr = (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24;
        ps.fbmsk = ctx.frame.fbmsk;
        ps.zwrite = ps.ztestEnabled && !ctx.zbuf.zmask;
        ps.fbMark = makePageMark(ps.fpsm, ps.fbp, ps.fbw);
        ps.zMark = makePageMark(ps.zpsm, ps.zbp, ps.fbw);
        ps.fbRead = readFuncs[ps.fpsm & 0x3Fu];
        ps.fbWrite = writeFuncs[ps.fpsm & 0x3Fu];
        ps.zRead = readFuncs[ps.zpsm & 0x3Fu];
        ps.zWrite = writeFuncs[ps.zpsm & 0x3Fu];
        return ps;
    }

    // One framebuffer/depth pixel at a byte address computed once for its read and its write.
    // load/store are PixelStorageTraits::Read/Write without the address arithmetic, for the
    // formats PixelWriter specialises (32-, 24- and 16-bit colour and depth).
    template <int PSM>
    struct PixelAccess
    {
        static constexpr auto P = static_cast<PixelStorageMode>(PSM);
        using PackedT = typename GSMem::PixelStorageTraits<P>::PackedT;
        static constexpr bool k24 = PSM == GS_PSM_CT24 || PSM == GS_PSM_Z24;

        static inline uint32_t address(uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
        {
            return GSMem::ByteAddress<P>(bp, bw, x, y);
        }
        static inline uint32_t load(const uint8_t *vram, uint32_t address)
        {
            PackedT v;
            std::memcpy(&v, vram + address, sizeof(PackedT));
            if constexpr (k24)
                return v & 0x00FFFFFFu;
            else
                return v;
        }
        static inline void store(uint8_t *vram, uint32_t address, uint32_t value)
        {
            PackedT v = static_cast<PackedT>(value);
            if constexpr (k24)
            {
                PackedT old;
                std::memcpy(&old, vram + address, sizeof(PackedT));
                v = (old & 0xFF000000u) | (v & 0x00FFFFFFu);
            }
            std::memcpy(vram + address, &v, sizeof(PackedT));
        }
    };

    // Lazily computed address of one pixel in one buffer, then read/write through it.
    template <int PSM>
    struct PixelRef
    {
        uint8_t *vram;
        uint32_t bp, bw, x, y;
        ReadFn readFn;
        WriteFn writeFn;
        uint32_t addr = UINT32_MAX;

        inline uint32_t read()
        {
            if constexpr (PSM == kGenericPsm)
                return readFn(vram, bp, bw, x, y);
            else
            {
                if (addr == UINT32_MAX)
                    addr = PixelAccess<PSM>::address(bp, bw, x, y);
                return PixelAccess<PSM>::load(vram, addr);
            }
        }
        inline void write(uint32_t value)
        {
            if constexpr (PSM == kGenericPsm)
                writeFn(vram, bp, bw, x, y, value);
            else
            {
                if (addr == UINT32_MAX)
                    addr = PixelAccess<PSM>::address(bp, bw, x, y);
                PixelAccess<PSM>::store(vram, addr, value);
            }
        }
    };

    inline bool alphaTestPasses(const PixelSetup &ps, uint8_t alpha)
    {
        switch (ps.atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < ps.aref;
        case 3:
            return alpha <= ps.aref;
        case 4:
            return alpha == ps.aref;
        case 5:
            return alpha >= ps.aref;
        case 6:
            return alpha > ps.aref;
        default:
            return alpha != ps.aref;
        }
    }

    // One pixel through scissor, fog, alpha test, DATE, depth test, blending, FBA, FBMSK and
    // the framebuffer/depth writes. FP/ZP: frame/depth psm, or kGenericPsm.
    template <int FP, int ZP>
    struct PixelWriter
    {
        static inline void write(RasterCtx &rc, const PixelSetup &ps, int x, int y, uint32_t z,
                                 uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
        {
            if (x < ps.sx0 || x > ps.sx1 || y < ps.sy0 || y > ps.sy1)
                return;

            if (ps.fge)
            {
                const uint32_t inverseFog = 255u - fog;
                auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
                {
                    return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
                };
                r = applyFog(r, ps.fogR);
                g = applyFog(g, ps.fogG);
                b = applyFog(b, ps.fogB);
            }

            PixelWriteMask writeMask{};
            if (ps.ate && !alphaTestPasses(ps, a))
            {
                writeMask = ps.failMask;
                if (!writeMask.writesAnything())
                    return;
            }

            const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && ps.fb32;
            const bool frmw = ps.dateRead || (writeMask.writesFramebuffer() && ((ps.fbmsk != 0) || ps.abe || preserveDestinationAlpha));

            uint8_t *const vram = rc.vram;
            PixelRef<FP> fbPixel{vram, ps.fbp, ps.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y), ps.fbRead, ps.fbWrite};
            PixelRef<ZP> zPixel{vram, ps.zbp, ps.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y), ps.zRead, ps.zWrite};
            u32 rawFramebufferPixel = 0;
            u32 fbrgba = 0;
            if (frmw)
            {
                rawFramebufferPixel = fbPixel.read();
                fbrgba = rawFramebufferPixel;
                if (ps.fb16)
                    fbrgba = Rgba5551ToRgba8888(fbrgba);
                else if (ps.fb24)
                    fbrgba |= 0x80000000u;
            }

            if (ps.dateRead)
            {
                const uint32_t bit = ps.fb32 ? (rawFramebufferPixel >> 31) : (rawFramebufferPixel >> 15);
                if (((bit & 0x1u) != 0u) != ps.datm)
                    return;
            }

            switch (ps.ztestMethod)
            {
            case 0:
                return;
            case 1:
                break;
            case 2:
                if (!(z >= zPixel.read()))
                    return;
                break;
            default:
                if (!(z > zPixel.read()))
                    return;
                break;
            }

            if (writeMask.writesFramebuffer())
            {
                if (ps.abe)
                {
                    const uint8_t dr = fbrgba & 0xFF;
                    const uint8_t dg = (fbrgba >> 8) & 0xFF;
                    const uint8_t db = (fbrgba >> 16) & 0xFF;
                    const uint8_t da = (fbrgba >> 24) & 0xFF;

                    // PABE disables alpha blending when the source alpha MSB is clear.
                    if (!(ps.pabe && (a & 0x80u) == 0u))
                    {
                        auto pickRGB = [](uint8_t sel, int cs, int cd) -> int
                        {
                            if (sel == 0)
                                return cs;
                            if (sel == 1)
                                return cd;
                            return 0;
                        };
                        const int cAlpha = (ps.csel == 0) ? a : (ps.csel == 1) ? da
                                                                                : ps.fix;
                        r = clampU8(((pickRGB(ps.asel, r, dr) - pickRGB(ps.bsel, r, dr)) * cAlpha >> 7) + pickRGB(ps.dsel, r, dr));
                        g = clampU8(((pickRGB(ps.asel, g, dg) - pickRGB(ps.bsel, g, dg)) * cAlpha >> 7) + pickRGB(ps.dsel, g, dg));
                        b = clampU8(((pickRGB(ps.asel, b, db) - pickRGB(ps.bsel, b, db)) * cAlpha >> 7) + pickRGB(ps.dsel, b, db));
                    }
                }

                if (writeMask.writeAlpha && ps.fbaOr)
                    a = static_cast<uint8_t>(a | 0x80u);

                u32 pixel = pack32(r, g, b, a);
                if (ps.fbmsk != 0)
                    pixel = (pixel & ~ps.fbmsk) | (fbrgba & ps.fbmsk);
                if (preserveDestinationAlpha)
                    pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
                if (ps.fb16)
                    pixel = Rgba8888ToRgba5551(pixel);

                fbPixel.write(pixel);
                rc.mark(ps.fbMark, x, y);
            }

            if (ps.zwrite && writeMask.writeDepth)
            {
                zPixel.write(z);
                rc.mark(ps.zMark, x, y);
            }
        }
    };

    // ---- texturing ----

    using DirectTexelFn = uint32_t (*)(RasterCtx &, const TexSetup &, int, int);
    using DecodeTileFn = void (*)(RasterCtx &, const TexSetup &, DecodedTexture &, uint32_t, uint32_t);

    // wrapTextureCoordinate with its mode decoded once: REPEAT and REGION_REPEAT are
    // (c & andMask) | orMask, CLAMP and REGION_CLAMP are min(max(c, lo), hi).
    struct WrapAxis
    {
        bool clamp = false;
        uint32_t andMask = 0, orMask = 0;
        int lo = 0, hi = 0;

        static WrapAxis make(int size, uint8_t mode, uint16_t regionMin, uint16_t regionMax)
        {
            WrapAxis w;
            switch (mode & 0x3u)
            {
            case 0:
                w.andMask = static_cast<uint32_t>(size - 1);
                break;
            case 1:
                w.clamp = true;
                w.lo = 0;
                w.hi = size - 1;
                break;
            case 2:
                w.clamp = true;
                w.lo = regionMin;
                w.hi = regionMax;
                break;
            default:
                w.andMask = regionMin;
                w.orMask = regionMax;
                break;
            }
            return w;
        }

        [[gnu::always_inline]] inline int operator()(int c) const
        {
            return clamp ? std::min(std::max(c, lo), hi) : static_cast<int>((static_cast<uint32_t>(c) & andMask) | orMask);
        }
    };

    struct TexSetup
    {
        WrapAxis wrapAxisU, wrapAxisV;
        int texW = 1, texH = 1;
        uint8_t wrapU = 0, wrapV = 0;
        uint16_t minU = 0, maxU = 0, minV = 0, maxV = 0;
        bool fst = false, linear = false;
        DecodedTexture *tex = nullptr;
        uint32_t tbp0 = 0;
        uint32_t tbw = 0;
        uint8_t psm = 0;
        GSTexaReg texa{};
        const uint32_t *palette = nullptr; // indexed formats: CLUT resolved per index
        DirectTexelFn direct = nullptr;    // DecodeTexel for this psm (through the page cache)
        DecodeTileFn decodeTile = nullptr; // fills one 16x16 tile of `tex`
    };

    // DecodeTexel for a psm known at compile time: one read through the texture page cache,
    // then TEXA / 16-bit expansion / CLUT lookup.
    template <int PSM>
    uint32_t directTexel(RasterCtx &rc, const TexSetup &ts, int sampleU, int sampleV)
    {
        if constexpr (PSM == kGenericPsm)
        {
            return 0xFFFF00FFu; // not a texture format: DecodeTexel's fallback (no VRAM read)
        }
        else
        {
            constexpr auto P = static_cast<PixelStorageMode>(PSM);
            const u32 out = GSMem::ReadFast<P>(rc.vram, ts.tbp0, ts.tbw, static_cast<uint32_t>(sampleU),
                                               static_cast<uint32_t>(sampleV), rc.pageCache);
            if constexpr (PSM == GS_PSM_CT32 || PSM == GS_PSM_Z32 || PSM == GS_PSM_CT24 || PSM == GS_PSM_Z24)
                return applyTexa(ts.texa, static_cast<uint8_t>(PSM), out);
            else if constexpr (PSM == GS_PSM_CT16 || PSM == GS_PSM_CT16S || PSM == GS_PSM_Z16 || PSM == GS_PSM_Z16S)
                return applyTexa(ts.texa, static_cast<uint8_t>(PSM), Rgba5551ToRgba8888(static_cast<u16>(out)));
            else
                return ts.palette[static_cast<u8>(out)];
        }
    }

    template <int PSM>
    void decodeTile(RasterCtx &rc, const TexSetup &ts, DecodedTexture &texture, uint32_t u, uint32_t v);

    template <typename F>
    void dispatchTexturePsm(uint8_t psm, F &&f)
    {
        switch (psm)
        {
        case GS_PSM_CT32: f(std::integral_constant<int, GS_PSM_CT32>{}); break;
        case GS_PSM_CT24: f(std::integral_constant<int, GS_PSM_CT24>{}); break;
        case GS_PSM_CT16: f(std::integral_constant<int, GS_PSM_CT16>{}); break;
        case GS_PSM_CT16S: f(std::integral_constant<int, GS_PSM_CT16S>{}); break;
        case GS_PSM_Z32: f(std::integral_constant<int, GS_PSM_Z32>{}); break;
        case GS_PSM_Z24: f(std::integral_constant<int, GS_PSM_Z24>{}); break;
        case GS_PSM_Z16: f(std::integral_constant<int, GS_PSM_Z16>{}); break;
        case GS_PSM_Z16S: f(std::integral_constant<int, GS_PSM_Z16S>{}); break;
        case GS_PSM_T8: f(std::integral_constant<int, GS_PSM_T8>{}); break;
        case GS_PSM_T8H: f(std::integral_constant<int, GS_PSM_T8H>{}); break;
        case GS_PSM_T4: f(std::integral_constant<int, GS_PSM_T4>{}); break;
        case GS_PSM_T4HL: f(std::integral_constant<int, GS_PSM_T4HL>{}); break;
        case GS_PSM_T4HH: f(std::integral_constant<int, GS_PSM_T4HH>{}); break;
        default: f(std::integral_constant<int, kGenericPsm>{}); break;
        }
    }

    // Decoded before the current primitive started (so no decode and no page-cache access).
    // In parallel mode a tile decoded during the current primitive takes the slow path, which
    // records its first use.
    [[gnu::always_inline]] inline bool tileReadyBefore(const DecodedTexture &texture, uint32_t tile, const RasterCtx &rc)
    {
        const uint32_t epoch = texture.tileEpoch[tile].load(std::memory_order_acquire);
        return epoch != 0u && (epoch < rc.curSeq || (rc.touches == nullptr && epoch != kTileBusy));
    }

    // Slow path of a decoded fetch: decodes the tile if nobody has (workers race through an
    // atomic claim), and in parallel mode records the first fetch from a tile decoded during
    // this primitive.
    [[gnu::noinline]] void ensureTile(RasterCtx &rc, const TexSetup &ts, DecodedTexture &texture, uint32_t u, uint32_t v,
                                      uint32_t fetchIndex)
    {
        const uint32_t tile = (v >> kDecodedTileShift) * texture.tilesX + (u >> kDecodedTileShift);
        std::atomic<uint32_t> &epochRef = texture.tileEpoch[tile];
        uint32_t epoch = epochRef.load(std::memory_order_acquire);
        for (;;)
        {
            if (epoch == 0u)
            {
                if (epochRef.compare_exchange_weak(epoch, kTileBusy, std::memory_order_acquire, std::memory_order_acquire))
                {
                    ts.decodeTile(rc, ts, texture, u, v);
                    epochRef.store(rc.curSeq, std::memory_order_release);
                    epoch = rc.curSeq;
                    break;
                }
                continue; // epoch reloaded by the failed exchange
            }
            if (epoch == kTileBusy)
            {
                std::this_thread::yield();
                epoch = epochRef.load(std::memory_order_acquire);
                continue;
            }
            break;
        }
        if (epoch == rc.curSeq && rc.touches)
        {
            uint32_t *touched = texture.touched[rc.worker].get();
            if (!touched)
            {
                texture.touched[rc.worker].reset(new uint32_t[texture.tileCount]());
                touched = texture.touched[rc.worker].get();
            }
            if (touched[tile] != rc.curSeq)
            {
                touched[tile] = rc.curSeq;
                rc.touches->push_back({rc.curSeq, &texture, tile, rc.pixelKey | fetchIndex});
            }
        }
    }

    // Texel at an already wrapped coordinate.
    [[gnu::always_inline]] inline uint32_t fetchWrapped(RasterCtx &rc, const TexSetup &ts, int sampleU, int sampleV,
                                                         uint32_t fetchIndex)
    {
        DecodedTexture *texture = ts.tex;
        if (texture && static_cast<uint32_t>(sampleU) < texture->width && static_cast<uint32_t>(sampleV) < texture->height)
        {
            const uint32_t u = static_cast<uint32_t>(sampleU), v = static_cast<uint32_t>(sampleV);
            const uint32_t tile = (v >> kDecodedTileShift) * texture->tilesX + (u >> kDecodedTileShift);
            if (!tileReadyBefore(*texture, tile, rc)) [[unlikely]]
                ensureTile(rc, ts, *texture, u, v, fetchIndex);
            return texture->texels[static_cast<size_t>(v) * texture->width + u];
        }
        if (rc.touches)
        {
            rc.lastDirect.seq = rc.curSeq;
            rc.lastDirect.key = rc.pixelKey | fetchIndex;
            rc.lastDirect.u = sampleU;
            rc.lastDirect.v = sampleV;
            rc.lastDirect.ts = &ts;
        }
        return ts.direct(rc, ts, sampleU, sampleV);
    }

    [[gnu::always_inline]] inline uint32_t fetchTexel(RasterCtx &rc, const TexSetup &ts, int sampleU, int sampleV)
    {
        return fetchWrapped(rc, ts, ts.wrapAxisU(sampleU), ts.wrapAxisV(sampleV), 0u);
    }

    [[gnu::always_inline]] inline bool tileReady(const DecodedTexture &texture, uint32_t u, uint32_t v, const RasterCtx &rc)
    {
        return tileReadyBefore(texture, (v >> kDecodedTileShift) * texture.tilesX + (u >> kDecodedTileShift), rc);
    }

    // lerpChannel on all four channels at once. Same operations per channel: fused
    // multiply-adds (what clang emits for lerpChannel's a + (b - a) * f), round half away from
    // zero (lround), clamp.
    [[gnu::always_inline]] inline uint32_t bilerpTexels(uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11, float fx, float fy)
    {
#if defined(__aarch64__)
        auto expand = [](uint32_t c) -> float32x4_t
        {
            return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(c)))));
        };
        const float32x4_t f00 = expand(c00), f10 = expand(c10), f01 = expand(c01), f11 = expand(c11);
        const float32x4_t top = vfmaq_n_f32(f00, vsubq_f32(f10, f00), fx);
        const float32x4_t bottom = vfmaq_n_f32(f01, vsubq_f32(f11, f01), fx);
        const int32x4_t rounded = vcvtaq_s32_f32(vfmaq_n_f32(top, vsubq_f32(bottom, top), fy));
        const int32x4_t clamped = vminq_s32(vmaxq_s32(rounded, vdupq_n_s32(0)), vdupq_n_s32(255));
        const uint16x4_t narrow16 = vmovn_u32(vreinterpretq_u32_s32(clamped));
        const uint8x8_t narrow8 = vmovn_u16(vcombine_u16(narrow16, narrow16));
        return vget_lane_u32(vreinterpret_u32_u8(narrow8), 0);
#else
        const uint8_t r = lerpChannel(static_cast<uint8_t>(c00 & 0xFFu), static_cast<uint8_t>(c10 & 0xFFu),
                                      static_cast<uint8_t>(c01 & 0xFFu), static_cast<uint8_t>(c11 & 0xFFu), fx, fy);
        const uint8_t g = lerpChannel(static_cast<uint8_t>((c00 >> 8) & 0xFFu), static_cast<uint8_t>((c10 >> 8) & 0xFFu),
                                      static_cast<uint8_t>((c01 >> 8) & 0xFFu), static_cast<uint8_t>((c11 >> 8) & 0xFFu), fx, fy);
        const uint8_t b = lerpChannel(static_cast<uint8_t>((c00 >> 16) & 0xFFu), static_cast<uint8_t>((c10 >> 16) & 0xFFu),
                                      static_cast<uint8_t>((c01 >> 16) & 0xFFu), static_cast<uint8_t>((c11 >> 16) & 0xFFu), fx, fy);
        const uint8_t a = lerpChannel(static_cast<uint8_t>((c00 >> 24) & 0xFFu), static_cast<uint8_t>((c10 >> 24) & 0xFFu),
                                      static_cast<uint8_t>((c01 >> 24) & 0xFFu), static_cast<uint8_t>((c11 >> 24) & 0xFFu), fx, fy);
        return static_cast<uint32_t>(r) | (static_cast<uint32_t>(g) << 8) | (static_cast<uint32_t>(b) << 16) |
               (static_cast<uint32_t>(a) << 24);
#endif
    }

    [[gnu::always_inline]] inline uint32_t sampleTexture(RasterCtx &rc, const TexSetup &ts, float s, float t, float q, uint16_t u, uint16_t v)
    {
        const int texW = ts.texW;
        const int texH = ts.texH;
        float texUf, texVf;
        if (ts.fst)
        {
            texUf = static_cast<float>(u) / 16.0f;
            texVf = static_cast<float>(v) / 16.0f;
        }
        else
        {
            const float invQ = 1.0f / fabsQ(q);
            texUf = saturateTexCoord(s * invQ * static_cast<float>(texW));
            texVf = saturateTexCoord(t * invQ * static_cast<float>(texH));
        }

        if (!ts.linear)
            return fetchTexel(rc, ts, static_cast<int>(texUf), static_cast<int>(texVf));

        const float sampleU = texUf - 0.5f;
        const float sampleV = texVf - 0.5f;
        const int u0 = static_cast<int>(std::floor(sampleU));
        const int v0 = static_cast<int>(std::floor(sampleV));
        const int u1 = u0 + 1;
        const int v1 = v0 + 1;
        const float fx = sampleU - static_cast<float>(u0);
        const float fy = sampleV - static_cast<float>(v0);

        // Wrapping is a pure function of one coordinate: wrap each of u0/u1/v0/v1 once. When all
        // four texels are inside the decoded texture and their tiles are ready (the common
        // case), read them directly; otherwise fetch them one by one in the original order
        // (c00, c10, c01, c11), which keeps tile decodes and page-cache reads in order.
        const int wu0 = ts.wrapAxisU(u0);
        const int wu1 = ts.wrapAxisU(u1);
        const int wv0 = ts.wrapAxisV(v0);
        const int wv1 = ts.wrapAxisV(v1);
        uint32_t c00, c10, c01, c11;
        const DecodedTexture *texture = ts.tex;
        if (texture && static_cast<uint32_t>(wu0) < texture->width && static_cast<uint32_t>(wu1) < texture->width &&
            static_cast<uint32_t>(wv0) < texture->height && static_cast<uint32_t>(wv1) < texture->height &&
            tileReady(*texture, wu0, wv0, rc) &&
            (((static_cast<uint32_t>(wu0 ^ wu1) | static_cast<uint32_t>(wv0 ^ wv1)) >> kDecodedTileShift) == 0u || // one tile
             (tileReady(*texture, wu1, wv0, rc) && tileReady(*texture, wu0, wv1, rc) && tileReady(*texture, wu1, wv1, rc)))) [[likely]]
        {
            const uint32_t *row0 = texture->texels.data() + static_cast<size_t>(wv0) * texture->width;
            const uint32_t *row1 = texture->texels.data() + static_cast<size_t>(wv1) * texture->width;
            c00 = row0[wu0];
            c10 = row0[wu1];
            c01 = row1[wu0];
            c11 = row1[wu1];
        }
        else
        {
            c00 = fetchWrapped(rc, ts, wu0, wv0, 0u);
            c10 = fetchWrapped(rc, ts, wu1, wv0, 1u);
            c01 = fetchWrapped(rc, ts, wu0, wv1, 2u);
            c11 = fetchWrapped(rc, ts, wu1, wv1, 3u);
        }

        return bilerpTexels(c00, c10, c01, c11, fx, fy);
    }

    // ---- primitives ----

    template <class W>
    void drawSprite(RasterCtx &rc, const GSPrimitiveBatch &batch, const PixelSetup &ps, const TexSetup &ts)
    {
        const GSDrawState &state = batch.state;
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        const auto &ctx = state.context;

        int ofx = ctx.xyoffset.ofx >> 4;
        int ofy = ctx.xyoffset.ofy >> 4;

        int x0 = static_cast<int>(v0.x) - ofx;
        int y0 = static_cast<int>(v0.y) - ofy;
        int x1 = static_cast<int>(v1.x) - ofx;
        int y1 = static_cast<int>(v1.y) - ofy;
        u32 z1 = static_cast<u32>(v1.z);

        if (x0 > x1)
            std::swap(x0, x1);
        if (y0 > y1)
            std::swap(y0, y1);

        const int unclippedX0 = x0;
        const int unclippedY0 = y0;
        const int spanX = std::max(1, x1 - x0);
        const int spanY = std::max(1, y1 - y0);
        const int unclippedX1 = unclippedX0 + spanX - 1;
        const int unclippedY1 = unclippedY0 + spanY - 1;

        // If the sprite rectangle is fully outside scissor, nothing should render.
        if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
            unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
            return;

        const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
        const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
        const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
        const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);

        uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;

        if (state.prim.tme)
        {
            const auto &tex = ctx.tex0;
            const int texW = state.textureWidth;
            const int texH = state.textureHeight;

            float u0f, v0f, u1f, v1f;
            if (state.prim.fst)
            {
                u0f = static_cast<float>(v0.u >> 4);
                v0f = static_cast<float>(v0.v >> 4);
                u1f = static_cast<float>(v1.u >> 4);
                v1f = static_cast<float>(v1.v >> 4);
            }
            else
            {
                const float q0 = fabsQ(v0.q);
                const float q1 = fabsQ(v1.q);
                u0f = (v0.s / q0) * static_cast<float>(texW);
                v0f = (v0.t / q0) * static_cast<float>(texH);
                u1f = (v1.s / q1) * static_cast<float>(texW);
                v1f = (v1.t / q1) * static_cast<float>(texH);
            }

            float spriteW = static_cast<float>(spanX);
            float spriteH = static_cast<float>(spanY);
            if (spriteW < 1.0f)
                spriteW = 1.0f;
            if (spriteH < 1.0f)
                spriteH = 1.0f;

            for (int y = drawY0; y <= drawY1; ++y)
            {
                if (!rc.owns(y))
                    continue;
                float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
                float texVf = v0f + (v1f - v0f) * ty;

                for (int x = drawX0; x <= drawX1; ++x)
                {
                    rc.pixelKey = (static_cast<uint64_t>(y) << 34) | (static_cast<uint64_t>(x) << 2);
                    float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                    float texUf = u0f + (u1f - u0f) * tx;
                    uint32_t texel = 0xFFFF00FFu;
                    if (state.prim.fst)
                    {
                        const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                        const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                        const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                        const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                        texel = sampleTexture(rc, ts, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                    }
                    else
                    {
                        texel = sampleTexture(rc, ts, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u);
                    }

                    uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                    uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                    uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                    uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                    const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                    W::write(rc, ps, x, y, z1, color.r, color.g, color.b, color.a, v1.fog);
                }
            }
        }
        else
        {
            for (int y = drawY0; y <= drawY1; ++y)
            {
                if (!rc.owns(y))
                    continue;
                for (int x = drawX0; x <= drawX1; ++x)
                    W::write(rc, ps, x, y, z1, r, g, b, a, v1.fog);
            }
        }
    }

    template <class W>
    void drawTriangle(RasterCtx &rc, const GSPrimitiveBatch &batch, const PixelSetup &ps, const TexSetup &ts)
    {
        const GSDrawState &state = batch.state;
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        const GSVertex &v2 = batch.vertices[2];
        const auto &ctx = state.context;

        int ofx = ctx.xyoffset.ofx >> 4;
        int ofy = ctx.xyoffset.ofy >> 4;

        float fx0 = v0.x - static_cast<float>(ofx);
        float fy0 = v0.y - static_cast<float>(ofy);
        float fx1 = v1.x - static_cast<float>(ofx);
        float fy1 = v1.y - static_cast<float>(ofy);
        float fx2 = v2.x - static_cast<float>(ofx);
        float fy2 = v2.y - static_cast<float>(ofy);

        int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
        int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
        int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
        int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

        minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
        maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
        minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
        maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

        float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
        if (std::fabs(denom) < 0.001f)
            return;

        const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
        const float invAbsDenom = 1.0f / std::fabs(denom);
        constexpr float kEdgeEpsilon = 1.0e-4f;

        const bool iip = state.prim.iip;
        const bool tme = state.prim.tme;
        const bool fst = state.prim.fst;
        const bool fge = state.prim.fge;
        const auto &tex = ctx.tex0;

        for (int y = minY; y <= maxY; ++y)
        {
            if (!rc.owns(y))
                continue;
            float py = static_cast<float>(y) + 0.5f;
            for (int x = minX; x <= maxX; ++x)
            {
                float px = static_cast<float>(x) + 0.5f;

                float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
                float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
                float w2 = 1.0f - w0 - w1;

                if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                    continue;

                double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

                uint8_t r, g, b, a;
                if (iip)
                {
                    r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                    g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                    b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                    a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
                }
                else
                {
                    r = v2.r;
                    g = v2.g;
                    b = v2.b;
                    a = v2.a;
                }

                if (tme)
                {
                    rc.pixelKey = (static_cast<uint64_t>(y) << 34) | (static_cast<uint64_t>(x) << 2);
                    float is, it, iq;
                    uint16_t iu, iv;
                    if (fst)
                    {
                        iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                        iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                        is = 0.0f;
                        it = 0.0f;
                        iq = 1.0f;
                    }
                    else
                    {
                        // The GS DDA interpolates the homogeneous S, T and Q values. Texel
                        // coordinates are calculated from S/Q and T/Q only after interpolation.
                        is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                        it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                        iq = v0.q * w0 + v1.q * w1 + v2.q * w2;
                        iu = 0;
                        iv = 0;
                    }

                    const uint32_t texel = sampleTexture(rc, ts, is, it, iq, iu, iv);
                    const uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                    const uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                    const uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                    const uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);
                    const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                    r = color.r;
                    g = color.g;
                    b = color.b;
                    a = color.a;
                }

                // Fog is only consumed when FGE is set.
                const uint8_t fog = fge ? clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2)) : uint8_t{0};
                W::write(rc, ps, x, y, static_cast<u32>(z + 0.5), r, g, b, a, fog);
            }
        }
    }

    template <class W>
    void drawLine(RasterCtx &rc, const GSPrimitiveBatch &batch, const PixelSetup &ps)
    {
        const GSDrawState &state = batch.state;
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        const auto &ctx = state.context;

        int ofx = ctx.xyoffset.ofx >> 4;
        int ofy = ctx.xyoffset.ofy >> 4;

        int x0 = static_cast<int>(v0.x) - ofx;
        int y0 = static_cast<int>(v0.y) - ofy;
        int x1 = static_cast<int>(v1.x) - ofx;
        int y1 = static_cast<int>(v1.y) - ofy;

        int dx = std::abs(x1 - x0);
        int dy = -std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx + dy;

        int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
        if (totalSteps == 0)
            totalSteps = 1;
        int step = 0;

        for (;;)
        {
            float t = static_cast<float>(step) / static_cast<float>(totalSteps);
            uint8_t r, g, b, a;
            if (state.prim.iip)
            {
                r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
                g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
                b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
                a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
            }
            else
            {
                r = v1.r;
                g = v1.g;
                b = v1.b;
                a = v1.a;
            }

            double z = (v0.z + (v1.z - v0.z) * t);
            const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
            if (rc.owns(y0))
                W::write(rc, ps, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

            if (x0 == x1 && y0 == y1)
                break;

            int e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
            ++step;
        }
    }

    template <class W>
    void rasterPrimitive(RasterCtx &rc, const GSPrimitiveBatch &batch, const PixelSetup &ps, const TexSetup &ts)
    {
        const GSDrawState &state = batch.state;
        switch (state.prim.type)
        {
        case GS_PRIM_SPRITE:
            drawSprite<W>(rc, batch, ps, ts);
            break;
        case GS_PRIM_TRIANGLE:
        case GS_PRIM_TRISTRIP:
        case GS_PRIM_TRIFAN:
            drawTriangle<W>(rc, batch, ps, ts);
            break;
        case GS_PRIM_LINE:
        case GS_PRIM_LINESTRIP:
            drawLine<W>(rc, batch, ps);
            break;
        case GS_PRIM_POINT:
        {
            const GSVertex &v = batch.vertices[0];
            const auto &ctx = state.context;
            int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
            int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
            if (rc.owns(py))
                W::write(rc, ps, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
            break;
        }
        default:
            break;
        }
    }

    template <int FP>
    void rasterWithFrame(RasterCtx &rc, const GSPrimitiveBatch &batch, const PixelSetup &ps, const TexSetup &ts)
    {
        switch (ps.zpsm)
        {
        case GS_PSM_Z32: rasterPrimitive<PixelWriter<FP, GS_PSM_Z32>>(rc, batch, ps, ts); break;
        case GS_PSM_Z24: rasterPrimitive<PixelWriter<FP, GS_PSM_Z24>>(rc, batch, ps, ts); break;
        case GS_PSM_Z16: rasterPrimitive<PixelWriter<FP, GS_PSM_Z16>>(rc, batch, ps, ts); break;
        case GS_PSM_Z16S: rasterPrimitive<PixelWriter<FP, GS_PSM_Z16S>>(rc, batch, ps, ts); break;
        default: rasterPrimitive<PixelWriter<FP, kGenericPsm>>(rc, batch, ps, ts); break;
        }
    }

    void rasterDispatch(RasterCtx &rc, const GSPrimitiveBatch &batch, const PixelSetup &ps, const TexSetup &ts)
    {
        switch (ps.fpsm)
        {
        case GS_PSM_CT32: rasterWithFrame<GS_PSM_CT32>(rc, batch, ps, ts); break;
        case GS_PSM_CT24: rasterWithFrame<GS_PSM_CT24>(rc, batch, ps, ts); break;
        case GS_PSM_CT16: rasterWithFrame<GS_PSM_CT16>(rc, batch, ps, ts); break;
        case GS_PSM_CT16S: rasterWithFrame<GS_PSM_CT16S>(rc, batch, ps, ts); break;
        default: rasterWithFrame<kGenericPsm>(rc, batch, ps, ts); break;
        }
    }
}

// DecodedTexture is complete here; the tile decoder reads the texels in the same order as the
// per-texel decode did, so the texture page cache sees the same access sequence.
namespace
{
    template <int PSM>
    void decodeTile(RasterCtx &rc, const TexSetup &ts, DecodedTexture &texture, uint32_t u, uint32_t v)
    {
        const uint32_t x0 = (u >> kDecodedTileShift) << kDecodedTileShift;
        const uint32_t y0 = (v >> kDecodedTileShift) << kDecodedTileShift;
        const uint32_t x1 = std::min(x0 + (1u << kDecodedTileShift), texture.width);
        const uint32_t y1 = std::min(y0 + (1u << kDecodedTileShift), texture.height);
        for (uint32_t y = y0; y < y1; ++y)
        {
            uint32_t *row = texture.texels.data() + static_cast<size_t>(y) * texture.width;
            for (uint32_t x = x0; x < x1; ++x)
                row[x] = directTexel<PSM>(rc, ts, static_cast<int>(x), static_cast<int>(y));
        }
    }
}

void GSCpuBackend::MarkPagesWritten(const std::array<uint64_t, 8> &pages)
{
    const uint64_t stamp = ++m_vramWriteStamp;
    for (uint32_t word = 0; word < pages.size(); ++word)
    {
        uint64_t bits = pages[word];
        while (bits)
        {
            const uint32_t page = word * 64u + static_cast<uint32_t>(__builtin_ctzll(bits));
            bits &= bits - 1u;
            m_pageWriteStamp[page] = stamp;
            m_pageWrittenSinceFlush[page] = true;
        }
    }
}

void GSCpuBackend::BuildPalette(const GSTex0Reg &tex0, const GSTexaReg &texa, uint32_t *palette) const
{
    for (uint32_t i = 0; i < 256u; ++i)
        palette[i] = LookupCLUT(texa, static_cast<uint8_t>(i), tex0.cpsm, tex0.csm, tex0.csa, tex0.psm);
}

// ---------------------------------------------------------------------------------------------
// Parallel rasterization.
//
// N worker threads consume a queue of primitives in submission order. Worker i draws only the
// rows y with ((y >> bandShift) % N) == i, so every pixel (and the depth value under it) is
// still read and written by one thread, in submission order. What makes this bit-exact with
// the serial path is the set of rules the producer applies before it queues a primitive
// (anything that breaks them waits for the queue to drain, or runs on the producer thread
// with the real texture page cache, exactly as in serial mode):
//   - texture pages it reads were not written by a queued primitive, and frame/depth pages it
//     writes were not read as a texture by one (decoded texels come straight from VRAM);
//   - every frame/depth page queued primitives access is accessed with one layout (psm, base,
//     width), so a VRAM byte maps to one pixel row and thus one owner;
//   - its texture is served from the decoded-texture cache (the direct path reads through the
//     texture page cache in raster order: serial), its coordinates cannot leave the texture,
//     and its frame and depth buffers do not share pages;
//   - the texture page cache cannot hand it a stale page (cached page clean, or not one it reads).
// Workers decode texture tiles with direct VRAM reads, which equal reads through the page cache
// under those rules. The page cache state the serial path would end with is reconstructed when
// the queue drains: it holds the page of the last texel of the tile whose first use (in serial
// raster order) came last among the tiles decoded by the last primitive that decoded any.
// ---------------------------------------------------------------------------------------------
namespace
{
    using PageSet = std::array<uint64_t, 8>;

    inline bool pageSetHas(const PageSet &set, uint32_t page)
    {
        return ((set[page >> 6u] >> (page & 63u)) & 1u) != 0u;
    }

    inline void pageSetAddRange(PageSet &set, uint32_t first, uint64_t count)
    {
        if (count >= kGsPageCount)
        {
            set.fill(~0ull);
            return;
        }
        uint32_t page = first % kGsPageCount;
        uint32_t left = static_cast<uint32_t>(count);
        while (left)
        {
            const uint32_t bit = page & 63u;
            const uint32_t run = std::min<uint32_t>(left, 64u - bit);
            const uint64_t mask = (run == 64u ? ~0ull : ((1ull << run) - 1u)) << bit;
            set[page >> 6u] |= mask;
            left -= run;
            page = (page + run) % kGsPageCount;
        }
    }

    inline bool pageSetIntersects(const PageSet &a, const PageSet &b)
    {
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] & b[i])
                return true;
        return false;
    }

    inline void pageSetOr(PageSet &a, const PageSet &b)
    {
        for (size_t i = 0; i < a.size(); ++i)
            a[i] |= b[i];
    }

    inline bool pageSetEmpty(const PageSet &set)
    {
        for (uint64_t word : set)
            if (word)
                return false;
        return true;
    }

    // Pages MarkVramWritten would mark for any pixel in [x0,x1] x [y0,y1] (x, y >= 0).
    void addBufferPages(PageSet &set, const PageMark &m, int x0, int y0, int x1, int y1)
    {
        if (x1 < x0 || y1 < y0)
            return;
        const uint64_t first = m.basePage + (static_cast<uint64_t>(static_cast<uint32_t>(y0) >> m.shiftY) * m.pagesPerRow) +
                               (static_cast<uint32_t>(x0) >> m.shiftX);
        const uint64_t last = m.basePage + (static_cast<uint64_t>(static_cast<uint32_t>(y1) >> m.shiftY) * m.pagesPerRow) +
                              (static_cast<uint32_t>(x1) >> m.shiftX) + (m.plusOne ? 1u : 0u);
        pageSetAddRange(set, static_cast<uint32_t>(first % kGsPageCount), last - first + 1u);
    }

    // Conservative pixel bounds of a primitive, clipped to the scissor.
    void primitiveBounds(const GSPrimitiveBatch &batch, int &x0, int &y0, int &x1, int &y1)
    {
        const auto &ctx = batch.state.context;
        x0 = ctx.scissor.x0;
        y0 = ctx.scissor.y0;
        x1 = ctx.scissor.x1;
        y1 = ctx.scissor.y1;
        const uint32_t count = std::min<uint32_t>(std::max<uint32_t>(batch.vertexCount, 1u), 3u);
        const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        float minX = 1.0e9f, minY = 1.0e9f, maxX = -1.0e9f, maxY = -1.0e9f;
        for (uint32_t i = 0; i < count; ++i)
        {
            const float vx = batch.vertices[i].x - ofx;
            const float vy = batch.vertices[i].y - ofy;
            if (!(std::fabs(vx) < 1.0e8f) || !(std::fabs(vy) < 1.0e8f)) // NaN/huge: whole scissor
                return;
            minX = std::min(minX, vx);
            minY = std::min(minY, vy);
            maxX = std::max(maxX, vx);
            maxY = std::max(maxY, vy);
        }
        x0 = clampInt(static_cast<int>(std::floor(minX)) - 2, ctx.scissor.x0, ctx.scissor.x1);
        y0 = clampInt(static_cast<int>(std::floor(minY)) - 2, ctx.scissor.y0, ctx.scissor.y1);
        x1 = clampInt(static_cast<int>(std::ceil(maxX)) + 2, ctx.scissor.x0, ctx.scissor.x1);
        y1 = clampInt(static_cast<int>(std::ceil(maxY)) + 2, ctx.scissor.y0, ctx.scissor.y1);
    }

    // Largest coordinate wrapTextureCoordinate can return.
    int maxWrappedCoordinate(int size, uint8_t mode, uint16_t regionMin, uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0:
        case 1:
            return size - 1;
        case 2:
            return regionMax;
        default:
            return static_cast<int>(regionMin | regionMax);
        }
    }

    // Range of wrapped coordinates for unwrapped coordinates in [lo, hi].
    void wrapRange(int lo, int hi, int size, uint8_t mode, uint16_t regionMin, uint16_t regionMax, int &outLo, int &outHi)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            if (hi - lo + 1 >= size)
            {
                outLo = 0;
                outHi = size - 1;
                return;
            }
            outLo = static_cast<int>(static_cast<uint32_t>(lo) & static_cast<uint32_t>(size - 1));
            outHi = static_cast<int>(static_cast<uint32_t>(hi) & static_cast<uint32_t>(size - 1));
            if (outLo > outHi)
            {
                outLo = 0;
                outHi = size - 1;
            }
            return;
        case 1: // CLAMP
            outLo = clampInt(lo, 0, size - 1);
            outHi = clampInt(hi, 0, size - 1);
            return;
        case 2: // REGION_CLAMP (monotonic)
            outLo = std::min(std::max(lo, static_cast<int>(regionMin)), static_cast<int>(regionMax));
            outHi = std::min(std::max(hi, static_cast<int>(regionMin)), static_cast<int>(regionMax));
            return;
        default: // REGION_REPEAT
            outLo = 0;
            outHi = static_cast<int>(regionMin | regionMax);
            return;
        }
    }

    // Texel coordinates a sprite can sample, before wrapping (same arithmetic as drawSprite,
    // with a margin). False when they may saturate.
    bool spriteTexelBounds(const GSPrimitiveBatch &batch, int &uLo, int &uHi, int &vLo, int &vHi)
    {
        const GSDrawState &state = batch.state;
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u >> 4);
            v0f = static_cast<float>(v0.v >> 4);
            u1f = static_cast<float>(v1.u >> 4);
            v1f = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(state.textureWidth);
            v0f = (v0.t / q0) * static_cast<float>(state.textureHeight);
            u1f = (v1.s / q1) * static_cast<float>(state.textureWidth);
            v1f = (v1.t / q1) * static_cast<float>(state.textureHeight);
        }
        const float lo[2] = {std::min(u0f, u1f), std::min(v0f, v1f)};
        const float hi[2] = {std::max(u0f, u1f), std::max(v0f, v1f)};
        for (int i = 0; i < 2; ++i)
            if (!(lo[i] > -2000.0f) || !(hi[i] < 2000.0f))
                return false;
        uLo = static_cast<int>(std::floor(lo[0])) - 2;
        uHi = static_cast<int>(std::ceil(hi[0])) + 2;
        vLo = static_cast<int>(std::floor(lo[1])) - 2;
        vHi = static_cast<int>(std::ceil(hi[1])) + 2;
        return true;
    }

    // Pages the texel reads in [u0,u1] x [v0,v1] (wrapped coordinates) can touch.
    void addTexturePages(PageSet &set, uint8_t psm, uint32_t tbp0, uint32_t tbw, int u0, int v0, int u1, int v1)
    {
        uint32_t pageWidth = 0u, pageHeight = 0u;
        gsPageSize(psm, pageWidth, pageHeight);
        const uint64_t rowPages = (static_cast<uint64_t>(tbw) * 64u) / pageWidth; // PageId's row step
        const uint64_t base = tbp0 >> 5u;
        const uint64_t first = base + (static_cast<uint32_t>(v0) / pageHeight) * rowPages + static_cast<uint32_t>(u0) / pageWidth;
        const uint64_t last = base + (static_cast<uint32_t>(v1) / pageHeight) * rowPages + static_cast<uint32_t>(u1) / pageWidth +
                              ((tbp0 & 31u) ? 1u : 0u);
        pageSetAddRange(set, static_cast<uint32_t>(first % kGsPageCount), last - first + 1u);
    }

    uint64_t layoutKey(uint8_t psm, uint32_t base, uint32_t bw)
    {
        return static_cast<uint64_t>(psm) | (static_cast<uint64_t>(base) << 8) | (static_cast<uint64_t>(bw) << 40) | (1ull << 63);
    }

    uint32_t textureByteAddress(uint8_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
    {
        uint32_t address = 0u;
        dispatchTexturePsm(psm, [&](auto psmConstant)
                           {
                               constexpr int P = decltype(psmConstant)::value;
                               if constexpr (P != kGenericPsm)
                                   address = GSMem::ByteAddress<static_cast<PixelStorageMode>(P)>(bp, bw, x, y); });
        return address;
    }

    struct RasterCommand
    {
        GSPrimitiveBatch batch;
        PixelSetup ps;
        TexSetup ts;
        uint8_t *vram = nullptr;
        uint32_t seq = 0;
        std::array<uint32_t, 256> palette; // direct-path indexed textures
    };

    constexpr uint64_t kQueueCapacity = 4096u;
}

struct GSCpuBackend::Parallel
{
    struct alignas(64) WorkerState
    {
        std::atomic<uint64_t> done{0};
        RasterCtx rc;
        std::vector<TileTouch> touches;
    };

    uint32_t threadCount = 0;
    uint32_t bandShift = 2;
    std::vector<std::thread> threads;
    std::unique_ptr<WorkerState[]> workers;
    std::vector<RasterCommand> ring = std::vector<RasterCommand>(kQueueCapacity);
    alignas(64) std::atomic<uint64_t> published{0};
    std::atomic<uint32_t> sleepers{0};
    std::atomic<bool> mainWaiting{false};
    std::atomic<bool> stop{false};
    uint64_t drained = 0; // commands whose results were folded back
    // PS2X_GS_STATS=1: counters printed when the backend is destroyed.
    bool stats = false;
    uint64_t statQueued = 0, statSerial = 0, statDrains = 0, statConflict[8] = {}, statIneligible[4] = {};
    double statWaitMs = 0.0, statSerialMs = 0.0;

    // State of the commands queued since the last drain.
    PageSet written{};  // frame/depth pages they may write
    PageSet texRead{};  // texture pages they may read
    PageSet accessed{}; // frame/depth pages they read or write (union of `layouts`)
    struct LayoutPages
    {
        uint64_t key;
        PageSet pages;
    };
    std::vector<LayoutPages> layouts; // pages accessed per buffer layout
    // Latest page-cache access made by the producer while commands were queued (CLUT loads).
    struct CacheEvent
    {
        uint32_t seq = 0;
        uint32_t address = 0;
    } cacheEvent;

    bool idle() const { return published.load(std::memory_order_relaxed) == drained; }

    void workerLoop(uint32_t index)
    {
        WorkerState &self = workers[index];
        uint64_t next = 0;
        for (;;)
        {
            if (stop.load(std::memory_order_acquire))
                return; // (set only after a drain: nothing is left to draw)
            uint64_t pub = published.load(std::memory_order_acquire);
            if (next < pub)
            {
                const RasterCommand &cmd = ring[next % kQueueCapacity];
                RasterCtx &rc = self.rc;
                rc.vram = cmd.vram;
                rc.curSeq = cmd.seq;
                rasterDispatch(rc, cmd.batch, cmd.ps, cmd.ts);
                ++next;
                self.done.store(next, std::memory_order_seq_cst);
                if (mainWaiting.load(std::memory_order_seq_cst))
                    self.done.notify_one();
                continue;
            }
            if (stop.load(std::memory_order_acquire))
                return;
            bool arrived = false;
            for (int spin = 0; spin < 4000; ++spin)
            {
                if (published.load(std::memory_order_acquire) != pub || stop.load(std::memory_order_relaxed))
                {
                    arrived = true;
                    break;
                }
                if ((spin & 15) == 15)
                    std::this_thread::yield();
            }
            if (arrived)
                continue;
            sleepers.fetch_add(1u, std::memory_order_seq_cst);
            if (published.load(std::memory_order_seq_cst) == pub && !stop.load(std::memory_order_seq_cst))
                published.wait(pub, std::memory_order_seq_cst);
            sleepers.fetch_sub(1u, std::memory_order_seq_cst);
        }
    }

    void publish()
    {
        published.fetch_add(1u, std::memory_order_seq_cst);
        if (sleepers.load(std::memory_order_seq_cst) != 0u)
            published.notify_all();
    }

    void waitAll()
    {
        const uint64_t target = published.load(std::memory_order_relaxed);
        auto allDone = [&]()
        {
            for (uint32_t i = 0; i < threadCount; ++i)
                if (workers[i].done.load(std::memory_order_acquire) < target)
                    return false;
            return true;
        };
        for (int spin = 0; spin < 20000; ++spin)
        {
            if (allDone())
                return;
            if ((spin & 15) == 15)
                std::this_thread::yield();
        }
        mainWaiting.store(true, std::memory_order_seq_cst);
        for (uint32_t i = 0; i < threadCount; ++i)
        {
            uint64_t d = workers[i].done.load(std::memory_order_seq_cst);
            while (d < target)
            {
                workers[i].done.wait(d, std::memory_order_seq_cst);
                d = workers[i].done.load(std::memory_order_seq_cst);
            }
        }
        mainWaiting.store(false, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    void resetSegment()
    {
        cacheEvent = {};
        written.fill(0u);
        texRead.fill(0u);
        accessed.fill(0u);
        layouts.clear();
    }
};

uint32_t GSCpuBackend::DefaultRasterThreads()
{
    if (const char *env = std::getenv("PS2X_GS_THREADS"))
    {
        const long value = std::strtol(env, nullptr, 10);
        return static_cast<uint32_t>(std::clamp<long>(value, 1, kMaxRasterThreads));
    }
    // Workers pay off only with asynchronous commands: in the default synchronous mode every GIF
    // packet waits for them, which costs more than it saves (and starves under CPU contention).
    // With PS2X_GS_ASYNC=1, 3 workers measured best in game on an 8P+2E M1 Pro (the EE, VU1,
    // IOP, dispatcher and host threads need cores too).
    return DefaultAsyncCommands() ? 3u : 1u;
}

void GSCpuBackend::StartWorkers(uint32_t rasterThreads)
{
    rasterThreads = std::min<uint32_t>(rasterThreads, kMaxRasterThreads);
    if (rasterThreads > 1u)
    {
        m_parallel = std::make_unique<Parallel>();
        Parallel &par = *m_parallel;
        par.threadCount = rasterThreads;
        par.stats = std::getenv("PS2X_GS_STATS") != nullptr;
        if (const char *band = std::getenv("PS2X_GS_BAND_SHIFT"))
            par.bandShift = static_cast<uint32_t>(std::clamp<long>(std::strtol(band, nullptr, 10), 0, 8));
        par.workers.reset(new Parallel::WorkerState[rasterThreads]);
        for (uint32_t i = 0; i < rasterThreads; ++i)
        {
            RasterCtx &rc = par.workers[i].rc;
            rc.bandShift = par.bandShift;
            rc.bandCount = rasterThreads;
            rc.bandIndex = i;
            rc.worker = i;
            rc.touches = &par.workers[i].touches;
        }
        for (uint32_t i = 0; i < rasterThreads; ++i)
            par.threads.emplace_back([&par, i]()
                                     { par.workerLoop(i); });
    }
}

GSCpuBackend::~GSCpuBackend()
{
    if (m_async)
    {
        {
            std::lock_guard<std::mutex> lock(m_async->mutex);
            m_async->stop = true;
        }
        m_async->wake.notify_all();
        m_async->thread.join();
        m_async.reset();
    }
    if (m_parallel)
    {
        DrainUnlocked();
        if (m_parallel->stats)
        {
            const Parallel &p = *m_parallel;
            std::fprintf(stderr, "[gs:parallel] producer waited %.1f ms, drew %.1f ms serially\n", p.statWaitMs, p.statSerialMs);
            std::fprintf(stderr, "[gs:parallel] queued=%llu serial=%llu drains=%llu conflicts raw=%llu war=%llu layout=%llu stale=%llu cachedirty=%llu ineligible direct=%llu range=%llu alias=%llu\n",
                         (unsigned long long)p.statQueued, (unsigned long long)p.statSerial, (unsigned long long)p.statDrains,
                         (unsigned long long)p.statConflict[0], (unsigned long long)p.statConflict[1], (unsigned long long)p.statConflict[2],
                         (unsigned long long)p.statConflict[3], (unsigned long long)p.statConflict[4],
                         (unsigned long long)p.statIneligible[0], (unsigned long long)p.statIneligible[1], (unsigned long long)p.statIneligible[2]);
        }
        m_parallel->stop.store(true, std::memory_order_seq_cst);
        m_parallel->published.fetch_add(1u, std::memory_order_seq_cst); // wakes sleeping workers
        m_parallel->published.notify_all();
        for (std::thread &thread : m_parallel->threads)
            thread.join();
    }
}

void GSCpuBackend::RetireTexture(std::unique_ptr<DecodedTexture> texture)
{
    if (m_parallel && !m_parallel->idle())
        m_retiredTextures.push_back(std::move(texture));
}

void GSCpuBackend::DrainUnlocked()
{
    if (!m_parallel)
        return;
    Parallel &par = *m_parallel;
    if (par.idle())
    {
        m_retiredTextures.clear();
        return;
    }
    if (par.stats)
    {
        const auto t0 = std::chrono::steady_clock::now();
        par.waitAll();
        par.statWaitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    else
        par.waitAll();
    ++par.statDrains;

    PageSet written{};
    bool anyWritten = false;
    uint32_t lastDecodeSeq = 0u;
    for (uint32_t i = 0; i < par.threadCount; ++i)
    {
        Parallel::WorkerState &w = par.workers[i];
        if (w.rc.anyWritten)
        {
            pageSetOr(written, w.rc.written);
            anyWritten = true;
            w.rc.written.fill(0u);
            w.rc.anyWritten = false;
        }
        if (!w.touches.empty())
            lastDecodeSeq = std::max(lastDecodeSeq, w.touches.back().seq);
    }
    uint32_t lastDirectSeq = 0u;
    const RasterCtx::DirectFetch *lastDirect = nullptr;
    for (uint32_t i = 0; i < par.threadCount; ++i)
    {
        const RasterCtx::DirectFetch &d = par.workers[i].rc.lastDirect;
        if (d.seq == 0u)
            continue;
        if (!lastDirect || d.seq > lastDirect->seq || (d.seq == lastDirect->seq && d.key > lastDirect->key))
            lastDirect = &d;
    }
    if (lastDirect)
        lastDirectSeq = lastDirect->seq;

    if (par.cacheEvent.seq > lastDirectSeq && par.cacheEvent.seq > lastDecodeSeq)
    {
        m_texturePageCache.Invalidate();
        m_texturePageCache.Resolve(m_vram, par.cacheEvent.address);
        m_cacheDirty = false;
    }
    else if (lastDirectSeq > lastDecodeSeq)
    {
        // Direct-path primitives read every texel through the page cache: it ends on the page
        // of the last fetch in raster order.
        const TexSetup &ts = *lastDirect->ts;
        const uint32_t address = textureByteAddress(ts.psm, ts.tbp0, ts.tbw, static_cast<uint32_t>(lastDirect->u),
                                                    static_cast<uint32_t>(lastDirect->v));
        m_texturePageCache.Invalidate();
        m_texturePageCache.Resolve(m_vram, address);
        m_cacheDirty = false;
    }
    else if (lastDecodeSeq != 0u)
    {
        // The serial path's page cache would hold the page of the last texel read by the last
        // tile decode of the last primitive that decoded any: the tile whose first use, in
        // raster order, comes last.
        struct FirstUse
        {
            DecodedTexture *texture;
            uint32_t tile;
            uint64_t key;
        };
        std::vector<FirstUse> uses;
        for (uint32_t i = 0; i < par.threadCount; ++i)
            for (const TileTouch &touch : par.workers[i].touches)
            {
                if (touch.seq != lastDecodeSeq)
                    continue;
                bool merged = false;
                for (FirstUse &use : uses)
                    if (use.texture == touch.texture && use.tile == touch.tile)
                    {
                        use.key = std::min(use.key, touch.key);
                        merged = true;
                        break;
                    }
                if (!merged)
                    uses.push_back({touch.texture, touch.tile, touch.key});
            }
        const FirstUse *last = &uses.front();
        for (const FirstUse &use : uses)
            if (use.key > last->key)
                last = &use;
        const DecodedTexture &texture = *last->texture;
        const uint32_t x1 = std::min(((last->tile % texture.tilesX) << kDecodedTileShift) + (1u << kDecodedTileShift), texture.width);
        const uint32_t y1 = std::min(((last->tile / texture.tilesX) << kDecodedTileShift) + (1u << kDecodedTileShift), texture.height);
        const uint32_t address = textureByteAddress(texture.psm, texture.tbp0, texture.tbw, x1 - 1u, y1 - 1u);
        m_texturePageCache.Invalidate();
        m_texturePageCache.Resolve(m_vram, address);
        m_cacheDirty = false; // no queued primitive wrote a page another one read
    }
    else if (anyWritten && m_texturePageCache.PageBase() != UINT32_MAX &&
             pageSetHas(written, m_texturePageCache.PageBase() / GSMem::TexturePageCache::kPageSize))
    {
        m_cacheDirty = true;
    }
    if (anyWritten)
        MarkPagesWritten(written);

    for (uint32_t i = 0; i < par.threadCount; ++i)
    {
        par.workers[i].touches.clear();
        par.workers[i].rc.lastDirect = {};
    }
    par.resetSegment();
    par.drained = par.published.load(std::memory_order_relaxed);
    m_retiredTextures.clear();
}

void GSCpuBackend::DrainForAccessUnlocked(const std::array<uint64_t, 8> &reads, const std::array<uint64_t, 8> &writes)
{
    if (!m_parallel || m_parallel->idle())
        return;
    const Parallel &par = *m_parallel;
    if (pageSetIntersects(reads, par.written) || pageSetIntersects(writes, par.texRead) || pageSetIntersects(writes, par.accessed))
        DrainUnlocked();
}

namespace
{
    PageSet rectPages(uint8_t psm, uint32_t bp, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height)
    {
        PageSet set{};
        if (width && height)
            addTexturePages(set, psm, bp, bw, static_cast<int>(x0), static_cast<int>(y0), static_cast<int>(x0 + width - 1u),
                            static_cast<int>(y0 + height - 1u));
        return set;
    }
}

bool GSCpuBackend::LoadClutQueuedUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    // The CLUT load reads through the page cache. With primitives queued, its exact state is
    // unknown, but every page it may hold is clean unless it is the (dirty) page it held when
    // the queue last drained. So when the CLUT's pages were not written by queued primitives
    // and do not include that dirty page, direct VRAM reads return the same values, and the
    // cache afterwards holds the page of the last entry read (recorded for the drain; its pages
    // count as texture reads so no later queued primitive may write them).
    if (!m_parallel || m_parallel->idle())
        return false;
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return true; // reads nothing, changes nothing
    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;
    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;
    PageSet pages{};
    uint32_t lastAddress = 0u;
    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u, sourceY = 0u, sourceWidth = 1u;
        clutEntrySource(tex0, texclut, entry, sourceX, sourceY, sourceWidth);
        lastAddress = textureByteAddress(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY);
        const uint32_t page = lastAddress / GSMem::TexturePageCache::kPageSize;
        pages[page >> 6u] |= 1ull << (page & 63u);
    }
    if (firstEntry >= entryCount)
        return true;
    Parallel &par = *m_parallel;
    const uint32_t cacheBase = m_texturePageCache.PageBase();
    if (pageSetIntersects(pages, par.written) ||
        (m_cacheDirty && cacheBase != UINT32_MAX && pageSetHas(pages, cacheBase / GSMem::TexturePageCache::kPageSize)))
        return false;
    LoadClutUnlocked(tex0, texclut, true);
    par.cacheEvent = {++m_primSeq, lastAddress};
    pageSetOr(par.texRead, pages);
    return true;
}

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=(tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << ") scissor=(" << ctx.scissor.x0 << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1 << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test << " alpha=0x" << ctx.alpha << std::dec
                      << std::endl;
        }
    });

    const uint32_t seq = ++m_primSeq;
    const PixelSetup ps = makePixelSetup(state, m_readVramFuncs, m_writeVramFuncs);
    const bool tme = state.prim.tme;
    const auto &tex = ctx.tex0;
    const uint64_t clamp = ctx.clamp;
    const uint8_t wrapU = static_cast<uint8_t>(clamp & 0x3u);
    const uint8_t wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    const uint16_t minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    const uint16_t maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    const uint16_t minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    const uint16_t maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);

    // Queue the primitive for the workers when that is provably equivalent (see above).
    bool queue = false;
    if (m_parallel)
    {
        Parallel &par = *m_parallel;
        if (par.published.load(std::memory_order_relaxed) - par.drained >= kQueueCapacity)
            DrainUnlocked(); // ring full
        const bool decoded = tme && !m_decodedTexturesDisabled && decodedTextureAllowed(state);
        const int umax = tme ? maxWrappedCoordinate(state.textureWidth, wrapU, minU, maxU) : 0;
        const int vmax = tme ? maxWrappedCoordinate(state.textureHeight, wrapV, minV, maxV) : 0;
        const bool zAccess = ps.ztestMethod >= 2u || ps.zwrite;

        int bx0, by0, bx1, by1;
        primitiveBounds(batch, bx0, by0, bx1, by1);
        PageSet fbPages{}, zPages{}, texPages{};
        addBufferPages(fbPages, ps.fbMark, bx0, by0, bx1, by1);
        if (zAccess)
            addBufferPages(zPages, ps.zMark, bx0, by0, bx1, by1);
        int uLo = 0, uHi = umax, vLo = 0, vHi = vmax;
        bool knownPsm = false;
        dispatchTexturePsm(tex.psm, [&](auto psmConstant)
                           { knownPsm = decltype(psmConstant)::value != kGenericPsm; });
        if (tme)
        {
            int rawULo, rawUHi, rawVLo, rawVHi;
            if (!decoded && state.prim.type == GS_PRIM_SPRITE && spriteTexelBounds(batch, rawULo, rawUHi, rawVLo, rawVHi))
            {
                wrapRange(rawULo, rawUHi, state.textureWidth, wrapU, minU, maxU, uLo, uHi);
                wrapRange(rawVLo, rawVHi, state.textureHeight, wrapV, minV, maxV, vLo, vHi);
            }
            addTexturePages(texPages, tex.psm, tex.tbp0, tex.tbw, uLo, vLo, uHi, vHi);
        }
        PageSet ownWrites = fbPages;
        if (ps.zwrite)
            pageSetOr(ownWrites, zPages);

        // Direct-path (page cache) primitives qualify when they cannot read what they write.
        const bool directOk = tme && !decoded && knownPsm && !pageSetIntersects(texPages, ownWrites);
        const bool eligible = (!tme || directOk || (decoded && umax < state.textureWidth && vmax < state.textureHeight)) &&
                              !(zAccess && pageSetIntersects(fbPages, zPages));
        if (eligible)
        {
            const PageSet &writes = ownWrites;
            const uint64_t fbLayout = layoutKey(ps.fpsm, ps.fbp, ps.fbw);
            const uint64_t zLayout = layoutKey(ps.zpsm, ps.zbp, ps.fbw);
            auto layoutConflict = [&](const PageSet &pages, uint64_t key)
            {
                for (const Parallel::LayoutPages &entry : par.layouts)
                    if (entry.key != key && pageSetIntersects(pages, entry.pages))
                        return true;
                return false;
            };
            const uint32_t cacheBase = m_texturePageCache.PageBase();
            const bool cachedPageRead = tme && cacheBase != UINT32_MAX &&
                                        pageSetHas(texPages, cacheBase / GSMem::TexturePageCache::kPageSize);
            const bool raw = pageSetIntersects(texPages, par.written);
            const bool war = pageSetIntersects(writes, par.texRead);
            const bool lay = layoutConflict(fbPages, fbLayout) || (zAccess && layoutConflict(zPages, zLayout));
            const bool cdirty = cachedPageRead && m_cacheDirty;
            const bool conflict = raw || war || lay || cdirty;
            if (par.stats && !par.idle())
            {
                par.statConflict[0] += raw;
                par.statConflict[1] += war;
                par.statConflict[2] += lay;
                par.statConflict[4] += cdirty;
            }
            if (conflict)
                DrainUnlocked();
            // After a drain the page cache state is exact: a stale cached page that this
            // primitive may read forces the serial path.
            const bool stale = tme && m_texturePageCache.PageBase() != UINT32_MAX && m_cacheDirty &&
                               pageSetHas(texPages, m_texturePageCache.PageBase() / GSMem::TexturePageCache::kPageSize);
            par.statConflict[3] += stale;
            if (!stale)
            {
                queue = true;
                pageSetOr(par.written, writes);
                pageSetOr(par.texRead, texPages);
                auto recordLayout = [&](const PageSet &pages, uint64_t key)
                {
                    bool found = false;
                    for (Parallel::LayoutPages &entry : par.layouts)
                        if (entry.key == key)
                        {
                            pageSetOr(entry.pages, pages);
                            found = true;
                            break;
                        }
                    if (!found)
                        par.layouts.push_back({key, pages});
                    pageSetOr(par.accessed, pages);
                };
                recordLayout(fbPages, fbLayout);
                if (zAccess)
                    recordLayout(zPages, zLayout);
            }
        }
        else if (par.stats)
        {
            if (tme && !decoded && !directOk)
                ++par.statIneligible[0];
            else if (tme)
                ++par.statIneligible[1];
            else
                ++par.statIneligible[2];
        }
        if (!queue)
            DrainUnlocked();
        ++(queue ? par.statQueued : par.statSerial);
    }

    DecodedTexture *activeTexture = PrepareDecodedTexture(state);
    TexSetup ts;
    std::array<uint32_t, 256> directPalette;
    if (tme)
    {
        ts.texW = state.textureWidth;
        ts.texH = state.textureHeight;
        ts.wrapAxisU = WrapAxis::make(ts.texW, wrapU, minU, maxU);
        ts.wrapAxisV = WrapAxis::make(ts.texH, wrapV, minV, maxV);
        ts.wrapU = wrapU;
        ts.wrapV = wrapV;
        ts.minU = minU;
        ts.maxU = maxU;
        ts.minV = minV;
        ts.maxV = maxV;
        ts.fst = state.prim.fst;
        ts.linear = state.linearFilter;
        ts.tex = activeTexture;
        ts.tbp0 = tex.tbp0;
        ts.tbw = tex.tbw;
        ts.psm = tex.psm;
        ts.texa = state.texa;
        if (isFourBitIndexedPsm(tex.psm) || isEightBitIndexedPsm(tex.psm))
        {
            if (activeTexture)
                ts.palette = activeTexture->palette.data();
            else
            {
                BuildPalette(tex, state.texa, directPalette.data());
                ts.palette = directPalette.data();
            }
        }
        dispatchTexturePsm(tex.psm, [&](auto psmConstant)
                           {
                               constexpr int P = decltype(psmConstant)::value;
                               ts.direct = &directTexel<P>;
                               ts.decodeTile = &decodeTile<P>; });
    }

    if (queue)
    {
        Parallel &par = *m_parallel;
        RasterCommand &cmd = par.ring[par.published.load(std::memory_order_relaxed) % kQueueCapacity];
        cmd.batch = batch;
        cmd.ps = ps;
        cmd.ts = ts;
        if (ts.palette == directPalette.data())
        {
            cmd.palette = directPalette;
            cmd.ts.palette = cmd.palette.data();
        }
        cmd.vram = m_vram;
        cmd.seq = seq;
        par.publish();
        return;
    }

    // Serial: this thread, every row, the real texture page cache.
    RasterCtx rc;
    rc.vram = m_vram;
    rc.pageCache = &m_texturePageCache;
    rc.curSeq = seq;
    const uint32_t cacheBefore = m_texturePageCache.PageBase();
    if (m_parallel && m_parallel->stats)
    {
        const auto t0 = std::chrono::steady_clock::now();
        rasterDispatch(rc, batch, ps, ts);
        m_parallel->statSerialMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    else
        rasterDispatch(rc, batch, ps, ts);
    if (rc.anyWritten)
        MarkPagesWritten(rc.written);
    const uint32_t cacheAfter = m_texturePageCache.PageBase();
    if (cacheAfter == UINT32_MAX)
        m_cacheDirty = false;
    else
    {
        const bool writtenNow = rc.anyWritten && pageSetHas(rc.written, cacheAfter / GSMem::TexturePageCache::kPageSize);
        m_cacheDirty = (cacheAfter == cacheBefore) ? (m_cacheDirty || writtenNow) : writtenNow;
    }
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    if (OnAsyncCaller())
    {
        AsyncCommand queued;
        queued.kind = AsyncCommand::BeginTransfer;
        queued.transfer = command;
        m_async->push(std::move(queued));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    // Transfers run here while queued primitives continue, unless they touch the same pages.
    const GSBitBltBuf &blt = command.bitbltbuf;
    const uint32_t rrw = command.trxreg.rrw, rrh = command.trxreg.rrh;
    const PageSet srcPages = command.direction == 1u || command.direction == 2u
                                 ? rectPages(blt.spsm, blt.sbp, std::max<uint32_t>(blt.sbw, 1u), command.trxpos.ssax, command.trxpos.ssay, rrw, rrh)
                                 : PageSet{};
    const PageSet dstPages = command.direction == 0u || command.direction == 2u
                                 ? rectPages(blt.dpsm, blt.dbp, std::max<uint32_t>(blt.dbw, 1u), command.trxpos.dsax, command.trxpos.dsay, rrw, rrh)
                                 : PageSet{};
    if (command.direction == 1u || command.direction == 2u)
        DrainForAccessUnlocked(srcPages, command.direction == 2u ? dstPages : PageSet{});
    m_transferPages = dstPages;
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    if (OnAsyncCaller())
    {
        if (!data || sizeBytes == 0u)
            return;
        AsyncCommand command;
        command.kind = AsyncCommand::Upload;
        command.data.assign(data, data + sizeBytes);
        m_async->push(std::move(command));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    DrainForAccessUnlocked(PageSet{}, m_transferPages);
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;

    const uint32_t dbp = m_transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    uint32_t offset = 0u;

    auto advancePixel = [&](uint32_t count)
    {
        const uint32_t totalPixels = m_transferState.totalPixels;
        m_transferState.copiedPixels =
            std::min<uint32_t>(totalPixels, m_transferState.copiedPixels + count);

        if (m_transferState.copiedPixels >= totalPixels)
        {
            m_transferState.direction = 3u;
            m_transferState.totalPixels = 0u;
            return;
        }

        m_transferState.x = dsax + (m_transferState.copiedPixels % rrw);
        m_transferState.y = m_transfer.trxpos.dsay + (m_transferState.copiedPixels / rrw);
    };

    while (offset < sizeBytes && m_transferState.direction == 0u)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 4u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 3u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 2u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, data[offset++]);
            advancePixel(1u);
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            const uint32_t firstPixel = m_transferState.copiedPixels;
            WriteVramUnlocked(dpsm, dbp, dbw,
                              dsax + (firstPixel % rrw),
                              m_transfer.trxpos.dsay + (firstPixel / rrw),
                              packed & 0x0Fu);
            if (firstPixel + 1u < m_transferState.totalPixels)
            {
                const uint32_t secondPixel = firstPixel + 1u;
                WriteVramUnlocked(dpsm, dbp, dbw,
                                  dsax + (secondPixel % rrw),
                                  m_transfer.trxpos.dsay + (secondPixel / rrw),
                                  (packed >> 4u) & 0x0Fu);
            }
            advancePixel(std::min<uint32_t>(2u, m_transferState.totalPixels - firstPixel));
            break;
        }
        default:
            return;
        }
    }
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    WaitAsyncIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    if (OnAsyncCaller())
    {
        // Same result as the synchronous call; m_vram only changes in Initialize, which waits.
        const uint8_t psm = context.frame.psm;
        const bool handled = m_vram && context.frame.fbw != 0u &&
                             (psm == GS_PSM_CT32 || psm == GS_PSM_CT24 || psm == GS_PSM_CT16 || psm == GS_PSM_CT16S);
        if (handled)
        {
            AsyncCommand command;
            command.kind = AsyncCommand::Clear;
            command.context = context;
            command.rgba = rgba;
            m_async->push(std::move(command));
        }
        return handled;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || context.frame.fbw == 0u)
        return false;
    {
        const uint32_t cx0 = context.scissor.x0, cy0 = context.scissor.y0;
        const uint32_t cx1 = std::max<uint32_t>(cx0, context.scissor.x1), cy1 = std::max<uint32_t>(cy0, context.scissor.y1);
        DrainForAccessUnlocked(PageSet{}, rectPages(context.frame.psm, GSInternal::framePageBaseToBlock(context.frame.fbp),
                                                    std::max<uint32_t>(context.frame.fbw, 1u), cx0, cy0, cx1 - cx0 + 1u, cy1 - cy0 + 1u));
    }

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }
    return false;
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    if (OnAsyncCaller())
    {
        // Queue a snapshot at this point of the command stream, show the newest finished one.
        // Waits only for the very first, or when nothing else is queued (then it is immediate;
        // a caller that Syncs first, like gs_replay, gets the synchronous result).
        thread_local std::vector<uint8_t> vram;
        thread_local uint64_t shownGeneration = 0;
        thread_local PresentationFrame shown;
        thread_local GSCpuBackend converter;
        GSPresentationRequest shownRequest{};
        bool fresh = false;
        {
            std::unique_lock<std::mutex> lock(m_async->mutex);
            uint64_t waitFor = 1u;
            if (!m_async->snapshotQueued)
            {
                if (m_async->doneCount == m_async->pushedCount)
                    waitFor = m_async->snapshotGeneration + 1u;
                AsyncCommand command;
                command.kind = AsyncCommand::Snapshot;
                command.request = request;
                m_async->snapshotQueued = true;
                m_async->pushLocked(lock, std::move(command));
            }
            m_async->idleCv.wait(lock, [&] { return m_async->snapshotGeneration >= waitFor; });
            if (m_async->snapshotGeneration != shownGeneration)
            {
                vram = m_async->snapshot;
                shownRequest = m_async->snapshotRequest;
                shownGeneration = m_async->snapshotGeneration;
                fresh = true;
            }
        }
        if (fresh && !vram.empty())
        {
            converter.Initialize(vram.data(), static_cast<uint32_t>(vram.size()));
            shown = converter.PresentFromLocalMemory(shownRequest);
        }
        return shown;
    }
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    SnapshotVram(snapshot);
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(displayFrame1, origin1, width1, height1, false, true, selected1, crt1, preferred1) &&
            copySource(displayFrame2, origin2, width2, height2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < height1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            if (fieldMode)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    if (!copySource(displayFrame, origin, result.width, result.height, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    if (fieldMode)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}
