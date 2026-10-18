#include "runtime/gs/gs_parallel_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "runtime/ps2_memory.h"
#include "ps2_log.h"
#include "context.hpp"
#include "device.hpp"
#include "gs_interface.hpp"
#include "thread_id.hpp"
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>

namespace
{
    constexpr size_t VramSize = 4 * 1024 * 1024;
    uint64_t word(const void *data)
    {
        uint64_t value;
        std::memcpy(&value, data, 8);
        return value;
    }

    unsigned bitsPerPixel(unsigned psm)
    {
        switch (psm & 63)
        {
        case GS_PSM_CT24:
        case GS_PSM_Z24:
            return 24;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            return 16;
        case GS_PSM_T8:
        case GS_PSM_T8H:
            return 8;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            return 4;
        default:
            return 32;
        }
    }

    using Reader = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
    using Writer = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    struct PixelAccess
    {
        Reader read;
        Writer write;
    };

    // PS2X_PGS_DEBUG=1: periodic present/readback/register-state logging (diagnostics).
    bool pgsDebug()
    {
        static const bool enabled = std::getenv("PS2X_PGS_DEBUG") != nullptr;
        return enabled;
    }

    PixelAccess pixelAccess(unsigned psm)
    {
        using namespace GSMem;
        switch (psm & 63)
        {
#define PSM(id, name) \
    case GS_PSM_##id: \
        return {Read##name, Write##name}
            PSM(CT32, CT32);
            PSM(CT24, CT24);
            PSM(CT16, CT16);
            PSM(CT16S, CT16S);
            PSM(T8, P8);
            PSM(T8H, P8H);
            PSM(T4, P4);
            PSM(T4HL, P4HL);
            PSM(T4HH, P4HH);
            PSM(Z32, Z32);
            PSM(Z24, Z24);
            PSM(Z16, Z16);
            PSM(Z16S, Z16S);
#undef PSM
        default:
            return {ReadNull, WriteNull};
        }
    }

    // PCRTC merge: with only one read circuit enabled, its picture is shown as is, as the CPU
    // backend and PCSX2 2.8.2 do. paraLLEl-GS always blends circuit 1 over the background by
    // the framebuffer's alpha when PMODE.MMOD=0; TC displays a CT16S buffer whose A bits are all 0
    // (PMODE=0xff45: EN1 only, MMOD=0), so its whole picture came out black. Rewriting PMODE to
    // MMOD=1/ALP=0xff gives that pass-through. With both circuits enabled PMODE is passed on as is.
    uint64_t effectivePmode(uint64_t pmode)
    {
        const bool en1 = (pmode & 1u) != 0u, en2 = (pmode & 2u) != 0u;
        if (!en1 || en2 || (pmode & 0x20u))
            return pmode;
        return (pmode & ~0xff20ull) | (1ull << 5) | (0xffull << 8);
    }

    // Owns the Vulkan context/device and returns each scanout as host RGBA pixels. #285 shared
    // the image with OpenGL instead; macOS cannot import Vulkan memory into GL, and a 640x448
    // readback per frame is cheap on unified memory.
    class VulkanHost
    {
    public:
        VulkanHost()
        {
#ifdef __APPLE__
            // Granite dlopen()s the loader by bare name; Homebrew's is not on the default path.
            if (!std::getenv("GRANITE_VULKAN_LIBRARY"))
            {
                for (const char *path : {"/opt/homebrew/lib/libvulkan.1.dylib", "/usr/local/lib/libvulkan.1.dylib"})
                {
                    struct stat st{};
                    if (stat(path, &st) == 0)
                    {
                        setenv("GRANITE_VULKAN_LIBRARY", path, 0);
                        break;
                    }
                }
            }
#endif
            Util::register_thread_index(0);
            if (!Vulkan::Context::init_loader(nullptr))
                throw std::runtime_error("parallel-gs: Vulkan loader unavailable");
            m_context.set_num_thread_indices(1);
            if (!m_context.init_instance_and_device(nullptr, 0, nullptr, 0,
                                                    Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT |
                                                        Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_HEAP_BIT |
                                                        Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_BUFFER_BIT))
                throw std::runtime_error("parallel-gs: Vulkan instance/device creation failed");
            m_device.set_context(m_context);
            m_device.init_frame_contexts(4);
        }

        Vulkan::Device &device() { return m_device; }

        PresentationFrame present(const ParallelGS::ScanoutResult &scanout)
        {
            PresentationFrame frame;
            static uint32_t presents = 0;
            if (pgsDebug() && (presents < 12 || (presents % 600) == 0))
                std::printf("[pgs] present #%u image=%d fmt=%d size=%ux%u internal=%ux%u mode=%ux%u\n", presents,
                            scanout.image ? 1 : 0, scanout.image ? int(scanout.image->get_format()) : -1,
                            scanout.image ? scanout.image->get_width() : 0u, scanout.image ? scanout.image->get_height() : 0u,
                            scanout.internal_width, scanout.internal_height, scanout.mode_width, scanout.mode_height);
            ++presents;
            if (!scanout.image || scanout.image->get_format() != VK_FORMAT_R8G8B8A8_UNORM)
                return frame;
            const uint32_t width = scanout.image->get_width();
            const uint32_t height = scanout.image->get_height();
            const VkDeviceSize bytes = VkDeviceSize(width) * height * 4u;
            if (!m_readback || m_readback->get_create_info().size < bytes)
            {
                Vulkan::BufferCreateInfo info{};
                info.domain = Vulkan::BufferDomain::CachedHost;
                info.size = bytes;
                info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                m_readback = m_device.create_buffer(info);
                if (!m_readback)
                    return frame;
            }

            auto cmd = m_device.request_command_buffer();
            cmd->copy_image_to_buffer(*m_readback, *scanout.image, 0, {0, 0, 0}, {width, height, 1}, 0, 0,
                                      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
            cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
            Vulkan::Fence fence;
            m_device.submit(cmd, &fence);
            fence->wait();

            const auto *pixels = static_cast<const uint8_t *>(m_device.map_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT));
            if (!pixels)
                return frame;
            frame.pixels.assign(pixels, pixels + bytes);
            m_device.unmap_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT);
            // Scanout alpha is the game's framebuffer alpha (0x00-0x80); the host draws the frame
            // with blending, so present it opaque like the CPU backend does.
            uint64_t rgbSum = 0;
            for (size_t i = 0; i < frame.pixels.size(); i += 4)
            {
                rgbSum += frame.pixels[i] + frame.pixels[i + 1] + frame.pixels[i + 2];
                frame.pixels[i + 3] = 0xFFu;
            }
            static uint32_t readbacks = 0;
            if (pgsDebug() && (readbacks < 8 || (readbacks % 600) == 0))
                std::printf("[pgs] readback #%u %ux%u rgb_sum=%llu\n", readbacks, width, height,
                            static_cast<unsigned long long>(rgbSum));
            ++readbacks;
            m_device.next_frame_context();
            frame.width = width;
            frame.height = height;
            return frame;
        }

    private:
        Vulkan::Context m_context;
        Vulkan::Device m_device;
        Vulkan::BufferHandle m_readback;
    };

    class ParallelBackend final : public GSRasterBackend, private ParallelGS::SignalInterface
    {
        // All GSInterface/Device entry points are serialized, including presentation.
        mutable std::recursive_mutex mutex;
        VulkanHost interop;
        mutable ParallelGS::GSInterface gs;
        GSRegisters &priv;
        struct Lock
        {
            std::lock_guard<std::recursive_mutex> guard;
            explicit Lock(std::recursive_mutex &m) : guard(m) { Util::register_thread_index(0); }
        };
        uint64_t bitblt = 0, trxreg = 0;
        GSTransferSnapshot transfer;
        size_t transferBytes = 0, uploadedBytes = 0;
        std::array<uint8_t, 8> uploadTail{};
        size_t uploadTailSize = 0;
        std::array<uint8_t, 16> fifoTail{};
        size_t fifoOffset = 16;
        mutable uint64_t readbacks = 0;
        uint64_t statPackets[4]{}, statBytes[4]{}, statTags[4]{}, statVerts[4]{};
        std::map<std::string, uint64_t> statStates;
        void noteState(uint32_t path)
        {
            const auto &r = gs.get_register_state();
            const auto &c = r.ctx[r.prim.desc.CTXT];
            char key[384];
            std::snprintf(key, sizeof(key),
                          "p%u prim=%u tme=%u abe=%u fst=%u ctx=%u fbp=%u fbw=%u psm=%u fbmsk=%x zbp=%u zpsm=%u zmsk=%u zte=%u ztst=%u ate=%u atst=%u afail=%u ofs=%u,%u sc=%u-%u,%u-%u tbp=%u tbw=%u tpsm=%u tw=%u th=%u tfx=%u tcc=%u alpha=%llx",
                          path, r.prim.desc.PRIM, r.prim.desc.TME, r.prim.desc.ABE, r.prim.desc.FST, r.prim.desc.CTXT,
                          c.frame.desc.FBP, c.frame.desc.FBW, c.frame.desc.PSM, c.frame.desc.FBMSK, c.zbuf.desc.ZBP,
                          c.zbuf.desc.PSM, c.zbuf.desc.ZMSK, c.test.desc.ZTE, c.test.desc.ZTST, c.test.desc.ATE,
                          c.test.desc.ATST, c.test.desc.AFAIL, c.xyoffset.desc.OFX >> 4, c.xyoffset.desc.OFY >> 4,
                          c.scissor.desc.SCAX0, c.scissor.desc.SCAX1, c.scissor.desc.SCAY0, c.scissor.desc.SCAY1,
                          unsigned(c.tex0.desc.TBP0), unsigned(c.tex0.desc.TBW), unsigned(c.tex0.desc.PSM),
                          unsigned(c.tex0.desc.TW), unsigned(c.tex0.desc.TH), unsigned(c.tex0.desc.TFX),
                          unsigned(c.tex0.desc.TCC), (unsigned long long)c.alpha.bits);
            ++statStates[key];
        }

        // PS2X_PGS_TRACE_LINES (diagnostic): line budget of the PS2X_PGS_TRACE_AT trace (default 3000).
        static uint32_t traceLimit()
        {
            static const uint32_t limit = std::getenv("PS2X_PGS_TRACE_LINES") ? uint32_t(std::atoi(std::getenv("PS2X_PGS_TRACE_LINES"))) : 3000u;
            return limit;
        }
        bool traceActive = false;
        uint32_t traceLines = 0, tracePath = 0;
        void observeRegister(uint8_t addr, uint64_t value)
        {
            if (traceActive && traceLines < traceLimit() &&
                (addr <= 0x01 || addr == 0x06 || addr == 0x07 || addr == 0x42 || addr == 0x43 || addr == 0x47 ||
                 addr == 0x48 || addr == 0x4a || addr == 0x4b || addr == 0x4c || addr == 0x4d || addr == 0x50 || addr == 0x53))
            {
                ++traceLines;
                std::printf("[pgs-trace] p%u reg %02x = %016llx\n", tracePath, addr, (unsigned long long)value);
            }
            if (addr == GS_REG_BITBLTBUF)
                bitblt = value;
            if (addr == GS_REG_TRXREG)
                trxreg = value;
            if (addr == GS_REG_TRXDIR)
            {
                transfer = {};
                transfer.direction = value & 3;
                transfer.totalPixels = uint32_t(trxreg & 0xfff) * uint32_t((trxreg >> 32) & 0xfff);
                unsigned psm = unsigned(bitblt >> (transfer.direction == 1 ? 24 : 56)) & 63;
                transferBytes = (size_t(transfer.totalPixels) * bitsPerPixel(psm) + 7) / 8;
                transfer.localToHostPendingBytes = transfer.direction == 1 ? transferBytes : 0;
                if (transfer.direction == 1 && transferBytes)
                    ++readbacks;
                transfer.copiedPixels = transfer.direction == 2 ? transfer.totalPixels : 0;
                uploadedBytes = uploadTailSize = 0;
                fifoOffset = fifoTail.size();
            }

            if (addr == GS_REG_HWREG)
                observeImage(8);
        }
        void observeImage(size_t bytes)
        {
            if (transfer.direction != 0)
                return;
            uploadedBytes = std::min(transferBytes, uploadedBytes + bytes);
            transfer.copiedPixels = std::min<size_t>(transfer.totalPixels, uploadedBytes * 8 / bitsPerPixel(unsigned(bitblt >> 56)));
            auto width = uint32_t(trxreg & 0xfff);
            if (width)
            {
                transfer.x = transfer.copiedPixels % width;
                transfer.y = transfer.copiedPixels / width;
            }
        }

        // Observe transfer boundaries only; the upstream decoder alone executes commands.
        // Start from upstream's saved PATH state, including packets split across DMA calls.
        void observeGIF(uint32_t path, const uint8_t *data, size_t size)
        {
            auto state = gs.get_gif_path(path);
            for (size_t offset = 0; offset < size;)
            {
                if (state.loop == state.tag.NLOOP)
                {
                    std::memcpy(&state.tag, data + offset, 16);
                    state.reg = state.loop = 0;
                    offset += 16;
                    ++statTags[path];
                    if (traceActive && traceLines < traceLimit())
                    {
                        ++traceLines;
                        std::printf("[pgs-trace] p%u tag nloop=%u eop=%u pre=%u prim=%03x flg=%u nreg=%u regs=%016llx\n", path,
                                    unsigned(state.tag.NLOOP), unsigned(state.tag.EOP), unsigned(state.tag.PRE),
                                    unsigned(state.tag.PRIM), unsigned(state.tag.FLG), unsigned(state.tag.NREG),
                                    (unsigned long long)word(reinterpret_cast<const uint8_t *>(&state.tag) + 8));
                    }
                    if (state.tag.FLG < 2)
                    {
                        const uint64_t desc = word(reinterpret_cast<const uint8_t *>(&state.tag) + 8);
                        const unsigned n = state.tag.NREG ? state.tag.NREG : 16;
                        for (unsigned r = 0; r < n; ++r)
                        {
                            const unsigned d = (desc >> (r * 4)) & 15;
                            if (d == 4 || d == 5 || d == 12 || d == 13)
                                statVerts[path] += state.tag.NLOOP;
                        }
                    }
                    continue;
                }
                unsigned nreg = state.tag.NREG ? state.tag.NREG : 16;
                if (state.tag.FLG >= 2)
                {
                    size_t count = std::min<size_t>(state.tag.NLOOP - state.loop, (size - offset) / 16);
                    observeImage(count * 16);
                    state.loop += uint32_t(count);
                    offset += count * 16;
                }
                else
                {
                    // Only packed A+D can address the transfer registers (>= 0x50).
                    const uint64_t descriptors = word(reinterpret_cast<const uint8_t *>(&state.tag) + 8);
                    if (state.tag.FLG == 0 && ((descriptors >> (state.reg * 4)) & 15) == 14)
                        observeRegister(uint8_t(word(data + offset + 8) & 127), word(data + offset));
                    unsigned count = state.tag.FLG == 0 ? 1 : 2;
                    while (count-- && state.loop < state.tag.NLOOP)
                        if (++state.reg == nreg)
                        {
                            state.reg = 0;
                            ++state.loop;
                        }
                    offset += 16;
                }
            }
        }

        bool on_signal(uint64_t value) override
        {
            auto mask = uint32_t(value >> 32);
            auto id = (uint32_t(priv.siglblid) & ~mask) | (uint32_t(value) & mask);
            priv.siglblid = (priv.siglblid & 0xffffffff00000000ull) | id;
            priv.csr.fetch_or(1);
            return false;
        }

        bool on_finish(uint64_t) override
        {
            gs.flush();
            interop.device().wait_idle();
            priv.csr.fetch_or(2);
            return false;
        }

        bool on_label(uint64_t value) override
        {
            auto mask = uint32_t(value >> 32);
            auto id = (uint32_t(priv.siglblid >> 32) & ~mask) | (uint32_t(value) & mask);
            priv.siglblid = (uint64_t(id) << 32) | uint32_t(priv.siglblid);
            return false;
        }

    public:
        explicit ParallelBackend(GSRegisters &registers) : priv(registers)
        {
            if (!gs.init(&interop.device(), ParallelGS::GSOptions{}))
                throw std::runtime_error("parallel-gs: GS initialization failed; check Vulkan feature diagnostics above");
            gs.set_signal_interface(this);
        }

        ~ParallelBackend() override
        {
            Lock lock(mutex);
            gs.flush();
            interop.device().wait_idle();
        }

        bool UsesRawCommands() const override
        {
            return true;
        }


        void Initialize(uint8_t *vram, uint32_t size) override
        {
            Lock lock(mutex);
            if (!vram || size < VramSize)
                throw std::invalid_argument("parallel-gs requires 4 MiB initial VRAM");
            std::memcpy(gs.map_vram_write(0, VramSize), vram, VramSize);
            gs.end_vram_write(0, VramSize);
        }

        void Reset() override
        {
            Lock lock(mutex);
            gs.flush();
            interop.device().wait_idle();
            gs.reset_context_state();
            bitblt = trxreg = transferBytes = uploadedBytes = uploadTailSize = 0;
            fifoOffset = 16;
            transfer = {};
        }

        void ProcessGIF(uint32_t path, const uint8_t *data, uint32_t size) override
        {
            Lock lock(mutex);
            if (path < 1 || path > 3 || size % 16)
                throw std::invalid_argument("parallel-gs: invalid GIF path or size");
            if (!data || !size)
                return;
            ++statPackets[path];
            tracePath = path;
            statBytes[path] += size;
            observeGIF(path, data, size);
            gs.gif_transfer(path, data, size);
            if (pgsDebug())
                noteState(path); // per-packet state statistics, printed by Present under PS2X_PGS_DEBUG
        }

        void WriteRegisterRaw(uint8_t addr, uint64_t value) override
        {
            Lock lock(mutex);
            tracePath = 0;
            if (addr >= 128)
                return; // Reserved addresses never index upstream's 128-entry table.
            observeRegister(addr, value);
            // Compatibility aliases used by native stubs, not actual GIF registers.
            switch (addr)
            {
            case 0x59:
                priv.dispfb1 = value;
                return;
            case 0x5a:
                priv.display1 = value;
                return;
            case 0x5b:
                priv.dispfb2 = value;
                return;
            case 0x5c:
                priv.display2 = value;
                return;
            case 0x5f:
                priv.bgcolor = value;
                return;
            }
            gs.write_register(static_cast<ParallelGS::RegisterAddr>(addr), value);
        }

        void Submit(const GSPrimitiveBatch &) override
        {
            throw std::logic_error("parallel-gs received a CPU primitive batch");
        }

        void LoadClut(const GSTex0Reg &, const GSTexClutReg &) override
        {
        }

        void BeginTransfer(const GSTransferCommand &) override
        {
            throw std::logic_error("parallel-gs requires raw transfer registers");
        }

        void UploadImage(const uint8_t *data, uint32_t size) override
        {
            Lock lock(mutex);
            if (!data || transfer.direction != 0)
                return;
            size = uint32_t(std::min<size_t>(size, transferBytes - uploadedBytes));
            observeImage(size);
            while (size)
            {
                auto n = std::min<size_t>(size, 8 - uploadTailSize);
                std::memcpy(uploadTail.data() + uploadTailSize, data, n);
                uploadTailSize += n;
                data += n;
                size -= uint32_t(n);
                if (uploadTailSize == 8 || uploadedBytes == transferBytes && size == 0)
                {
                    std::fill(uploadTail.begin() + uploadTailSize, uploadTail.end(), 0);
                    gs.write_register(ParallelGS::RegisterAddr::HWREG, word(uploadTail.data()));
                    uploadTailSize = 0;
                }
            }
        }

        void Flush() override
        {
            Lock lock(mutex);
            gs.flush();
        }

        void TextureFlush() override
        {
            WriteRegisterRaw(GS_REG_TEXFLUSH, 0);
        }

        void Sync(GSSyncReason reason) override
        {
            Lock lock(mutex);
            if (reason == GSSyncReason::Finish || reason == GSSyncReason::Reset)
            {
                gs.flush();
                interop.device().wait_idle();
            }
            // map_vram_read and the upstream FIFO synchronize their own readbacks.
        }

        PresentationFrame Present(const GSPresentationRequest &request) override
        {
            Lock lock(mutex);
            gs.flush();
            auto &p = gs.get_priv_register_state();
            p.qwords_lo[0] = effectivePmode(request.pmode);
            // PS2X_PGS_FORCE_OPAQUE=1 (diagnostic): merge circuit 1 with constant alpha 0xff
            // (PMODE.MMOD=1, ALP=0xff) instead of the framebuffer's own alpha.
            static const bool forceOpaque = std::getenv("PS2X_PGS_FORCE_OPAQUE") != nullptr;
            if (forceOpaque)
                p.qwords_lo[0] = (request.pmode & ~0xff20ull) | (1ull << 5) | (0xffull << 8);
            p.qwords_lo[2] = request.smode1;
            p.qwords_lo[4] = request.smode2;
            p.qwords_lo[12] = request.syncv;
            p.qwords_lo[14] = request.dispfb1;
            p.qwords_lo[16] = request.display1;
            p.qwords_lo[18] = request.dispfb2;
            p.qwords_lo[20] = request.display2;
            p.qwords_lo[28] = request.bgcolor;
            static uint32_t presentLogs = 0;
            if (pgsDebug() && (presentLogs % 200) == 0)
            {
                for (unsigned i = 1; i <= 3; ++i)
                    std::printf("[pgs] path%u packets=%llu bytes=%llu tags=%llu verts=%llu\n", i,
                                (unsigned long long)statPackets[i], (unsigned long long)statBytes[i],
                                (unsigned long long)statTags[i], (unsigned long long)statVerts[i]);
                std::memset(statPackets, 0, sizeof(statPackets));
                std::memset(statBytes, 0, sizeof(statBytes));
                std::memset(statTags, 0, sizeof(statTags));
                std::memset(statVerts, 0, sizeof(statVerts));
                std::vector<std::pair<uint64_t, std::string>> top;
                for (auto &kv : statStates)
                    top.emplace_back(kv.second, kv.first);
                std::sort(top.rbegin(), top.rend());
                for (size_t i = 0; i < top.size() && i < 12; ++i)
                    std::printf("[pgs] state %llu x %s\n", (unsigned long long)top[i].first, top[i].second.c_str());
                statStates.clear();
            }
            if (pgsDebug() && (presentLogs < 6 || (presentLogs % 600) == 0))
                std::printf("[pgs] priv pmode=%llx (effective %llx, texa=%llx) smode1=%llx smode2=%llx dispfb1=%llx display1=%llx dispfb2=%llx display2=%llx bg=%llx\n",
                            (unsigned long long)request.pmode, (unsigned long long)p.qwords_lo[0],
                            (unsigned long long)gs.get_register_state().texa.bits, (unsigned long long)request.smode1, (unsigned long long)request.smode2,
                            (unsigned long long)request.dispfb1, (unsigned long long)request.display1,
                            (unsigned long long)request.dispfb2, (unsigned long long)request.display2,
                            (unsigned long long)request.bgcolor);
            // PS2X_PGS_DUMP_EVERY=N (diagnostic): every N presents write the 3D target (FBP 70, CT32),
            // the display buffer (FBP 0, CT16S) and the Z buffer (ZBP 210, Z24) as PPM files.
            static const int dumpEvery = std::getenv("PS2X_PGS_DUMP_EVERY") ? std::atoi(std::getenv("PS2X_PGS_DUMP_EVERY")) : 0;
            if (dumpEvery > 0 && presentLogs % dumpEvery == 0)
            {
                auto *vram = const_cast<uint8_t *>(static_cast<const uint8_t *>(gs.map_vram_read(0, VramSize)));
                struct Dump { const char *name; unsigned psm; uint32_t base; };
                const Dump dumps[] = {{"fb70", GS_PSM_CT32, 70 * 32}, {"fb0", GS_PSM_CT16S, 0}, {"z210", GS_PSM_Z24, 210 * 32}};
                for (const auto &d : dumps)
                {
                    char name[64];
                    std::snprintf(name, sizeof(name), "pgs_%s_%05u.ppm", d.name, presentLogs);
                    FILE *f = std::fopen(name, "wb");
                    if (!f)
                        continue;
                    std::fprintf(f, "P6 640 448 255\n");
                    auto access = pixelAccess(d.psm);
                    for (uint32_t y = 0; y < 448; ++y)
                        for (uint32_t x = 0; x < 640; ++x)
                        {
                            uint32_t v = access.read(vram, d.base, 10, x, y);
                            uint8_t rgb[3];
                            if (d.psm == GS_PSM_CT16S)
                            {
                                rgb[0] = uint8_t((v & 31) << 3);
                                rgb[1] = uint8_t(((v >> 5) & 31) << 3);
                                rgb[2] = uint8_t(((v >> 10) & 31) << 3);
                            }
                            else if (d.psm == GS_PSM_Z24)
                            {
                                rgb[0] = uint8_t(v >> 16);
                                rgb[1] = uint8_t(v >> 8);
                                rgb[2] = uint8_t(v);
                            }
                            else
                            {
                                rgb[0] = uint8_t(v);
                                rgb[1] = uint8_t(v >> 8);
                                rgb[2] = uint8_t(v >> 16);
                            }
                            std::fwrite(rgb, 1, 3, f);
                        }
                    std::fclose(f);
                }
                uint32_t hist70[8]{}, a16 = 0;
                auto a32 = pixelAccess(GS_PSM_CT32), c16 = pixelAccess(GS_PSM_CT16S);
                for (uint32_t y = 0; y < 448; ++y)
                    for (uint32_t x = 0; x < 640; ++x)
                    {
                        ++hist70[a32.read(vram, 70 * 32, 10, x, y) >> 29];
                        a16 += (c16.read(vram, 0, 10, x, y) >> 15) & 1;
                    }
                std::printf("[pgs] alpha #%u fb70 a>>5: %u %u %u %u %u %u %u %u  fb0 A=1: %u/286720 fba1=%llx fba2=%llx\n", presentLogs,
                            hist70[0], hist70[1], hist70[2], hist70[3], hist70[4], hist70[5], hist70[6], hist70[7], a16,
                            (unsigned long long)gs.get_register_state().ctx[0].fba.bits,
                            (unsigned long long)gs.get_register_state().ctx[1].fba.bits);
            }
            static const int traceAt = std::getenv("PS2X_PGS_TRACE_AT") ? std::atoi(std::getenv("PS2X_PGS_TRACE_AT")) : -1;
            traceActive = int(presentLogs) == traceAt;
            if (traceActive)
                std::printf("[pgs-trace] begin at present %u\n", presentLogs);
            ++presentLogs;
            ParallelGS::VSyncInfo info{};
            info.phase = request.field;
            info.dst_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            info.dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            info.dst_access = VK_ACCESS_2_TRANSFER_READ_BIT;
            info.crtc_offsets = true;
            info.force_progressive = request.fullFrame;
            return interop.present(gs.vsync(info));
        }

        uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override
        {
            Lock lock(mutex);
            auto *mapped = const_cast<uint8_t *>(static_cast<const uint8_t *>(gs.map_vram_read(0, VramSize)));
            ++readbacks;
            return pixelAccess(psm).read(mapped, base, bw, x, y);
        }

        void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
        {
            Lock lock(mutex);
            gs.map_vram_read(0, VramSize); // Preserve untouched pixels and packed-format lanes.
            auto *mapped = static_cast<uint8_t *>(gs.map_vram_write(0, VramSize));
            ++readbacks;
            pixelAccess(psm).write(mapped, base, bw, x, y, value);
            gs.end_vram_write(0, VramSize);
        }

        void SnapshotVram(std::vector<uint8_t> &out) const override
        {
            Lock lock(mutex);
            auto *mapped = static_cast<const uint8_t *>(gs.map_vram_read(0, VramSize));
            ++readbacks;
            out.assign(mapped, mapped + VramSize);
        }

        bool ClearFramebuffer(const GSContext &ctx, uint32_t rgba) override
        {
            Lock lock(mutex);
            if (!ctx.frame.fbw || (ctx.frame.psm != GS_PSM_CT32 && ctx.frame.psm != GS_PSM_CT24 &&
                                   ctx.frame.psm != GS_PSM_CT16 && ctx.frame.psm != GS_PSM_CT16S))
                return false;

            gs.map_vram_read(0, VramSize); // FBMSK and untouched VRAM require current contents.
            auto *mapped = static_cast<uint8_t *>(gs.map_vram_write(0, VramSize));
            ++readbacks;
            auto access = pixelAccess(ctx.frame.psm);

            if (ctx.fba & 1)
                rgba |= 0x80000000u;
            if (bitsPerPixel(ctx.frame.psm) == 16)
                rgba = ((rgba >> 3) & 31) | ((rgba >> 6) & 0x3e0) | ((rgba >> 9) & 0x7c00) | ((rgba >> 16) & 0x8000);

            for (uint32_t y = ctx.scissor.y0; y <= ctx.scissor.y1; ++y)
                for (uint32_t x = ctx.scissor.x0; x <= ctx.scissor.x1; ++x)
                {
                    auto old = access.read(mapped, ctx.frame.fbp * 32, ctx.frame.fbw, x, y);
                    access.write(mapped, ctx.frame.fbp * 32, ctx.frame.fbw, x, y,
                                 (rgba & ~ctx.frame.fbmsk) | (old & ctx.frame.fbmsk));
                }
            gs.end_vram_write(0, VramSize);
            return true;
        }

        uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override
        {
            Lock lock(mutex);
            if (!dst)
                return 0;
            auto count = uint32_t(std::min<size_t>(maxBytes, transfer.localToHostPendingBytes));
            for (uint32_t i = 0; i < count; ++i)
            {
                if (fifoOffset == 16)
                {
                    gs.read_transfer_fifo(fifoTail.data(), 1);
                    fifoOffset = 0;
                }
                dst[i] = fifoTail[fifoOffset++];
            }
            transfer.localToHostPendingBytes -= count;
            return count;
        }

        GSTransferSnapshot GetTransferSnapshot() const override
        {
            Lock lock(mutex);
            return transfer;
        }

        void ReadRegisterState(GSDebugSnapshot &s) const override
        {
            Lock lock(mutex);
            const auto &r = gs.get_register_state();
            for (unsigned i = 0; i < 2; ++i)
            {
                auto &c = s.ctx[i];
                const auto &v = r.ctx[i];
                auto f = v.frame.bits, t = v.tex0.bits, sc = v.scissor.bits, z = v.zbuf.bits;
                c.frame = {uint32_t(f & 511), uint32_t((f >> 16) & 63), uint8_t((f >> 24) & 63), uint32_t(f >> 32)};
                c.zbuf = {uint32_t(z & 511), uint8_t(((z >> 24) & 15) | 0x30), bool((z >> 32) & 1)};
                c.scissor = {uint16_t(sc & 2047), uint16_t((sc >> 16) & 2047), uint16_t((sc >> 32) & 2047), uint16_t((sc >> 48) & 2047)};
                c.xyoffset = {uint16_t(v.xyoffset.bits), uint16_t(v.xyoffset.bits >> 32)};
                c.tex0 = {uint32_t(t & 16383), uint8_t((t >> 14) & 63), uint8_t((t >> 20) & 63), uint8_t((t >> 26) & 15),
                          uint8_t((t >> 30) & 15), uint8_t((t >> 34) & 1), uint8_t((t >> 35) & 3), uint32_t((t >> 37) & 16383),
                          uint8_t((t >> 51) & 15), uint8_t((t >> 55) & 1), uint8_t((t >> 56) & 31), uint8_t(t >> 61)};
                c.tex1 = v.tex1.bits;
                c.miptbp1 = v.miptbl_1_3.bits;
                c.miptbp2 = v.miptbl_4_6.bits;
                c.clamp = v.clamp.bits;
                c.alpha = v.alpha.bits;
                c.test = v.test.bits;
                c.fba = v.fba.bits;
            }

            auto p = r.prim.bits;
            s.prim = {GSPrimType(p & 7), bool(p & 8), bool(p & 16), bool(p & 32), bool(p & 64), bool(p & 128), bool(p & 256), bool(p & 512), bool(p & 1024)};
            auto b = r.bitbltbuf.bits, t = r.trxpos.bits, a = r.texa.bits, c = r.texclut.bits;
            s.texa = {uint8_t(a), bool(a & 0x8000), uint8_t(a >> 32)};
            s.texclut = {uint8_t(c & 63), uint8_t((c >> 6) & 63), uint16_t((c >> 12) & 1023)};
            s.scanmsk = r.scanmsk.bits;
            s.dimx = r.dimx.bits;
            s.dthe = r.dthe.bits;
            s.colclamp = r.colclamp.bits;
            s.bitbltbuf = {uint32_t(b & 16383), uint8_t((b >> 16) & 63), uint8_t((b >> 24) & 63),
                           uint32_t((b >> 32) & 16383), uint8_t((b >> 48) & 63), uint8_t((b >> 56) & 63)};

            s.trxpos = {uint16_t(t & 2047), uint16_t((t >> 16) & 2047), uint16_t((t >> 32) & 2047), uint16_t((t >> 48) & 2047), uint8_t((t >> 59) & 3)};
            s.trxreg = {uint16_t(r.trxreg.bits & 4095), uint16_t((r.trxreg.bits >> 32) & 4095)};
            s.trxdir = uint32_t(r.trxdir.bits & 3);
        }
    };
}

std::unique_ptr<GSRasterBackend> CreateParallelGSBackend(GSRegisters &registers)
{
    return std::make_unique<ParallelBackend>(registers);
}
