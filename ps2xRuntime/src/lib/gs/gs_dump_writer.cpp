#include "runtime/gs/gs_dump_writer.h"
#include "runtime/gs/gs_backend.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif


namespace ps2x_gs_dump
{
    namespace
    {
        constexpr uint32_t kVramSize = 4u * 1024u * 1024u;
        constexpr uint32_t kStateVersion = 8u;
        constexpr uint32_t kPrivRegsSize = 8192u;

        enum class Phase : uint8_t
        {
            Disabled,
            Waiting,
            Recording,
            Done,
        };

        // Raw GS register values plus the vertex-queue values PCSX2 freezes (m_v, m_q),
        // maintained by parsing every GIF packet the runtime submits.
        struct Shadow
        {
            uint64_t regs[0x80]{};
            uint32_t uv = 0u;
            uint32_t fog = 0u;
            float q = 1.0f;

            struct Path
            {
                uint64_t tagLo = 0u;
                uint32_t nloop = 0u;
                uint32_t nreg = 0u;
                uint32_t reg = 0u;
                uint8_t flg = 0u;
                uint8_t regs[16]{};
            } paths[4];
        };

        struct Writer
        {
            std::mutex mutex;
            Phase phase = Phase::Disabled;
            std::string path;
            std::string serial;
            uint32_t startFrame = 0u;
            uint32_t frameCount = 2u;
            uint32_t framesWritten = 0u;
            uint64_t bytesWritten = 0u;
            uint32_t fifoBytes = 0u;
            FILE *file = nullptr;
            std::vector<char> fileBuffer;
            Shadow shadow;
        };

        std::atomic<bool> s_active{false};
        uint32_t s_presentations = 0u;

        // Host frame counter (g_ps2xHostFrame, ps2_runtime.cpp), looked up at run time so the
        // GS sources keep linking without the runtime (gs_cache tests, ps2x_gs_replay).
        // Falls back to counting presentations.
        uint32_t hostFrame()
        {
#if !defined(_WIN32)
            static auto *counter = static_cast<std::atomic<uint32_t> *>(dlsym(RTLD_DEFAULT, "g_ps2xHostFrame"));
            if (counter)
                return counter->load(std::memory_order_relaxed);
#endif
            return s_presentations;
        }

        Writer &writer()
        {
            static Writer *w = []
            {
                auto *created = new Writer();
                const char *path = std::getenv("PS2X_GS_DUMP");
                if (path && path[0] != '\0')
                {
                    created->path = path;
                    if (const char *start = std::getenv("PS2X_GS_DUMP_START"))
                        created->startFrame = static_cast<uint32_t>(std::strtoul(start, nullptr, 10));
                    if (const char *frames = std::getenv("PS2X_GS_DUMP_FRAMES"))
                        created->frameCount = std::max<uint32_t>(1u, static_cast<uint32_t>(std::strtoul(frames, nullptr, 10)));
                    if (const char *serial = std::getenv("PS2X_GS_DUMP_SERIAL"))
                        created->serial = serial;
                    created->shadow.regs[0x1a] = 1u; // PRMODECONT.AC resets to 1
                    created->phase = Phase::Waiting;
                    s_active.store(true, std::memory_order_release);
                }
                return created;
            }();
            return *w;
        }

        uint64_t load64(const uint8_t *p)
        {
            uint64_t v;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }

        uint32_t floatBits(float f)
        {
            uint32_t v;
            std::memcpy(&v, &f, sizeof(v));
            return v;
        }

        // Raw register write (A+D, REGLIST, direct writes).
        void shadowRaw(Shadow &s, uint8_t addr, uint64_t value)
        {
            addr &= 0x7Fu;
            s.regs[addr] = value;
            switch (addr)
            {
            case 0x01: // RGBAQ
            case 0x02: // ST
                break;
            case 0x03: // UV
                s.uv = static_cast<uint32_t>(value) & 0x3FFF3FFFu;
                break;
            case 0x04: // XYZF2
            case 0x0C: // XYZF3
                s.fog = static_cast<uint32_t>(value >> 56);
                break;
            case 0x0A: // FOG
                s.fog = static_cast<uint32_t>(value >> 56);
                break;
            default:
                break;
            }
        }

        void shadowPacked(Shadow &s, uint8_t desc, uint64_t lo, uint64_t hi)
        {
            switch (desc)
            {
            case 0x00: // PRIM
                shadowRaw(s, 0x00, lo & 0x7FFu);
                break;
            case 0x01: // RGBAQ: Q comes from the internal Q register
                s.regs[0x01] = (lo & 0xFFu) | (((lo >> 32) & 0xFFu) << 8) | ((hi & 0xFFu) << 16) |
                               (((hi >> 32) & 0xFFu) << 24) | (static_cast<uint64_t>(floatBits(s.q)) << 32);
                break;
            case 0x02: // ST: Q goes to the internal Q register
            {
                s.regs[0x02] = lo;
                const uint32_t qBits = static_cast<uint32_t>(hi);
                std::memcpy(&s.q, &qBits, sizeof(qBits));
                break;
            }
            case 0x03: // UV
                shadowRaw(s, 0x03, (lo & 0x3FFFu) | (((lo >> 32) & 0x3FFFu) << 16));
                break;
            case 0x04: // XYZF2
            case 0x0C: // XYZF3
            case 0x0A: // FOG
                s.fog = static_cast<uint32_t>((hi >> 36) & 0xFFu);
                break;
            case 0x05: // XYZ2
            case 0x0D: // XYZ3
            case 0x0F: // NOP
                break;
            case 0x0E: // A+D
                shadowRaw(s, static_cast<uint8_t>(hi & 0xFFu), lo);
                break;
            default: // TEX0_1/2, CLAMP_1/2 and the reserved 0x0B take the low qword as is
                shadowRaw(s, desc, lo);
                break;
            }
        }

        void shadowGif(Shadow &s, uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes)
        {
            Shadow::Path &p = s.paths[pathIndex & 3u];
            for (uint32_t offset = 0u; offset + 16u <= sizeBytes; offset += 16u)
            {
                const uint64_t lo = load64(data + offset);
                const uint64_t hi = load64(data + offset + 8u);
                if (p.nloop == 0u)
                {
                    p.tagLo = lo;
                    p.nloop = static_cast<uint32_t>(lo & 0x7FFFu);
                    p.flg = static_cast<uint8_t>((lo >> 58) & 0x3u);
                    p.nreg = static_cast<uint32_t>((lo >> 60) & 0xFu);
                    if (p.nreg == 0u)
                        p.nreg = 16u;
                    p.reg = 0u;
                    for (uint32_t i = 0u; i < 16u; ++i)
                        p.regs[i] = static_cast<uint8_t>((hi >> (i * 4u)) & 0xFu);
                    if (p.nloop != 0u)
                    {
                        s.q = 1.0f;
                        if (((lo >> 46) & 1u) != 0u && p.flg == 0u)
                            shadowRaw(s, 0x00, (lo >> 47) & 0x7FFu);
                    }
                    continue;
                }

                if (p.flg == 0u) // PACKED
                {
                    shadowPacked(s, p.regs[p.reg], lo, hi);
                    if (++p.reg == p.nreg)
                    {
                        p.reg = 0u;
                        --p.nloop;
                    }
                }
                else if (p.flg == 1u) // REGLIST: two raw registers per qword
                {
                    const uint64_t words[2] = {lo, hi};
                    for (uint64_t word : words)
                    {
                        if (p.nloop == 0u)
                            break;
                        const uint8_t addr = p.regs[p.reg];
                        if (addr != 0x0Eu && addr != 0x0Fu)
                            shadowRaw(s, addr, word);
                        if (++p.reg == p.nreg)
                        {
                            p.reg = 0u;
                            --p.nloop;
                        }
                    }
                }
                else // IMAGE
                {
                    --p.nloop;
                }
            }
        }

        void put(Writer &w, const void *data, size_t size)
        {
            if (!w.file || size == 0u)
                return;
            if (std::fwrite(data, 1, size, w.file) != size)
            {
                std::fprintf(stderr, "[gs-dump] write failed: %s\n", w.path.c_str());
                std::fclose(w.file);
                w.file = nullptr;
                w.phase = Phase::Done;
                s_active.store(false, std::memory_order_release);
                return;
            }
            w.bytesWritten += size;
        }

        template <typename T>
        void putValue(Writer &w, const T &value)
        {
            put(w, &value, sizeof(value));
        }

        void putTransfer(Writer &w, uint8_t path, const uint8_t *data, uint32_t sizeBytes)
        {
            if (sizeBytes == 0u)
                return;
            putValue<uint8_t>(w, 0u);
            putValue<uint8_t>(w, path);
            putValue<uint32_t>(w, sizeBytes);
            put(w, data, sizeBytes);
        }

        void buildPrivRegs(const GSRegisters *priv, const GSPresentationRequest &request, uint8_t (&out)[kPrivRegsSize])
        {
            std::memset(out, 0, sizeof(out));
            auto set = [&](uint32_t offset, uint64_t value)
            { std::memcpy(out + offset, &value, sizeof(value)); };
            set(0x000, request.pmode);
            set(0x010, request.smode1);
            set(0x020, request.smode2);
            set(0x060, request.syncv);
            set(0x070, request.dispfb1);
            set(0x080, request.display1);
            set(0x090, request.dispfb2);
            set(0x0A0, request.display2);
            set(0x0E0, request.bgcolor);
            if (priv)
            {
                set(0x030, priv->srfsh);
                set(0x040, priv->synch1);
                set(0x050, priv->synch2);
                set(0x0B0, priv->extbuf);
                set(0x0C0, priv->extdata);
                set(0x0D0, priv->extwrite);
                set(0x1000, priv->csr.load(std::memory_order_relaxed));
                set(0x1010, priv->imr);
                set(0x1040, priv->busdir);
                set(0x1080, priv->siglblid);
            }
        }

        bool begin(Writer &w, const GSRegisters *priv, const GSPresentationRequest &request, GSRasterBackend *backend)
        {
            std::vector<uint8_t> vram;
            if (backend)
            {
                backend->Flush();
                backend->Sync(GSSyncReason::DebugReadback);
                backend->SnapshotVram(vram);
            }
            vram.resize(kVramSize, 0u);

            w.file = std::fopen(w.path.c_str(), "wb");
            if (!w.file)
            {
                std::fprintf(stderr, "[gs-dump] cannot open %s\n", w.path.c_str());
                return false;
            }
            w.fileBuffer.resize(8u << 20);
            std::setvbuf(w.file, w.fileBuffer.data(), _IOFBF, w.fileBuffer.size());

            const Shadow &s = w.shadow;
            std::vector<uint8_t> state;
            auto add = [&](const void *p, size_t n)
            {
                const auto *bytes = static_cast<const uint8_t *>(p);
                state.insert(state.end(), bytes, bytes + n);
            };
            auto add64 = [&](uint64_t v)
            { add(&v, sizeof(v)); };
            auto add32 = [&](uint32_t v)
            { add(&v, sizeof(v)); };

            add32(kStateVersion);
            for (uint8_t reg : {0x00, 0x1A, 0x1C, 0x22, 0x3B, 0x3D, 0x44, 0x45, 0x46, 0x49, 0x50, 0x53, 0x51, 0x52, 0x52})
                add64(s.regs[reg]);
            for (uint8_t ctx = 0; ctx < 2; ++ctx)
            {
                // XYOFFSET, TEX0, TEX1, CLAMP, MIPTBP1, MIPTBP2, SCISSOR, ALPHA, TEST, FBA, FRAME, ZBUF
                for (uint8_t reg : {0x18, 0x06, 0x14, 0x08, 0x34, 0x36, 0x40, 0x42, 0x47, 0x4A, 0x4C, 0x4E})
                    add64(s.regs[reg + ctx]);
            }
            uint64_t rgbaq = s.regs[0x01];
            if ((rgbaq >> 32) == 0u)
                rgbaq |= static_cast<uint64_t>(floatBits(1.0f)) << 32;
            add64(rgbaq);
            add64(s.regs[0x02]);
            add32(s.uv);
            add32(s.fog);
            add64(0u);          // XYZ
            add64(0u);          // obsolete GIFReg
            add32(0u);          // m_tr.x
            add32(0u);          // m_tr.y
            add(vram.data(), kVramSize);
            for (int i = 0; i < 4; ++i)
            {
                const uint64_t tag[2] = {0u, 0u}; // whole packets only: every path is idle
                add(tag, sizeof(tag));
                add32(0u);
            }
            add(&s.q, sizeof(float));

            uint32_t header[9] = {};
            header[0] = kStateVersion;
            header[1] = static_cast<uint32_t>(state.size());
            header[2] = sizeof(header);
            header[3] = static_cast<uint32_t>(w.serial.size());
            header[4] = 0u; // crc
            header[7] = header[2] + header[3];
            putValue<uint32_t>(w, 0xFFFFFFFFu);
            putValue<uint32_t>(w, static_cast<uint32_t>(sizeof(header) + w.serial.size()));
            put(w, header, sizeof(header));
            put(w, w.serial.data(), w.serial.size());
            put(w, state.data(), state.size());

            uint8_t privRegs[kPrivRegsSize];
            buildPrivRegs(priv, request, privRegs);
            put(w, privRegs, sizeof(privRegs));
            return w.file != nullptr;
        }

        void finish(Writer &w)
        {
            if (w.file)
            {
                std::fclose(w.file);
                w.file = nullptr;
            }
            w.phase = Phase::Done;
            s_active.store(false, std::memory_order_release);
            std::fprintf(stderr, "[gs-dump] wrote %u frame(s), %llu bytes to %s\n", w.framesWritten,
                         static_cast<unsigned long long>(w.bytesWritten), w.path.c_str());
        }

        // Emits a PATH3 A+D packet and applies it to the shadow.
        void putAd(Writer &w, const std::vector<std::pair<uint8_t, uint64_t>> &writes)
        {
            std::vector<uint64_t> qwords;
            qwords.reserve(2u + writes.size() * 2u);
            qwords.push_back(static_cast<uint64_t>(writes.size()) | (1ull << 15) | (1ull << 60)); // NLOOP, EOP, NREG=1, PACKED
            qwords.push_back(0x0Eu);                                                               // A+D
            for (const auto &[addr, value] : writes)
            {
                qwords.push_back(value);
                qwords.push_back(addr);
                shadowRaw(w.shadow, addr, value);
            }
            if (w.phase == Phase::Recording)
                putTransfer(w, 3u, reinterpret_cast<const uint8_t *>(qwords.data()), static_cast<uint32_t>(qwords.size() * 8u));
        }
    }

    bool active()
    {
        static const bool initialized = (writer(), true);
        (void)initialized;
        return s_active.load(std::memory_order_acquire);
    }

    void transfer(uint32_t path, const uint8_t *data, uint32_t sizeBytes)
    {
        if (!active() || !data || sizeBytes < 16u)
            return;
        Writer &w = writer();
        std::lock_guard<std::mutex> lock(w.mutex);
        sizeBytes &= ~15u;
        shadowGif(w.shadow, path, data, sizeBytes);
        if (w.phase == Phase::Recording)
            putTransfer(w, static_cast<uint8_t>(path & 3u), data, sizeBytes);
    }

    void registerWrite(uint8_t addr, uint64_t value)
    {
        if (!active())
            return;
        Writer &w = writer();
        std::lock_guard<std::mutex> lock(w.mutex);
        putAd(w, {{addr, value}});
    }

    void imageUpload(uint64_t bitbltbuf, uint64_t trxpos, uint64_t trxreg, uint64_t trxdir,
                     const uint8_t *data, uint32_t sizeBytes)
    {
        if (!active() || !data || sizeBytes == 0u)
            return;
        Writer &w = writer();
        std::lock_guard<std::mutex> lock(w.mutex);
        putAd(w, {{0x50, bitbltbuf}, {0x51, trxpos}, {0x52, trxreg}, {0x53, trxdir}});
        if (w.phase != Phase::Recording)
            return;

        const uint32_t qwc = (sizeBytes + 15u) / 16u;
        std::vector<uint8_t> packet;
        packet.reserve(static_cast<size_t>(qwc) * 16u + ((qwc + 0x7FFEu) / 0x7FFFu) * 16u);
        for (uint32_t done = 0u; done < qwc;)
        {
            const uint32_t n = std::min<uint32_t>(qwc - done, 0x7FFFu);
            const bool last = done + n == qwc;
            const uint64_t tag[2] = {n | (last ? (1ull << 15) : 0ull) | (2ull << 58), 0u};
            const auto *tagBytes = reinterpret_cast<const uint8_t *>(tag);
            packet.insert(packet.end(), tagBytes, tagBytes + 16u);
            const size_t begin = static_cast<size_t>(done) * 16u;
            const size_t end = std::min<size_t>(begin + static_cast<size_t>(n) * 16u, sizeBytes);
            packet.insert(packet.end(), data + begin, data + end);
            packet.resize(packet.size() + (static_cast<size_t>(n) * 16u - (end - begin)), 0u);
            done += n;
        }
        putTransfer(w, 3u, packet.data(), static_cast<uint32_t>(packet.size()));
    }

    void clearFramebuffer(int contextIndex, uint32_t rgba)
    {
        if (!active())
            return;
        Writer &w = writer();
        std::lock_guard<std::mutex> lock(w.mutex);
        const Shadow &s = w.shadow;
        const uint32_t ctx = contextIndex < 0 ? static_cast<uint32_t>((s.regs[0x00] >> 9) & 1u) : (contextIndex ? 1u : 0u);
        const uint64_t prim = s.regs[0x00];
        const uint64_t rgbaq = s.regs[0x01];
        const uint64_t test = s.regs[0x47 + ctx];
        const uint64_t zbuf = s.regs[0x4E + ctx];
        const uint64_t xyoffset = s.regs[0x18 + ctx];
        const uint64_t scissor = s.regs[0x40 + ctx];
        const uint64_t ofx = xyoffset & 0xFFFFu;
        const uint64_t ofy = (xyoffset >> 32) & 0xFFFFu;
        const uint64_t x0 = scissor & 0x7FFu, x1 = ((scissor >> 16) & 0x7FFu) + 1u;
        const uint64_t y0 = (scissor >> 32) & 0x7FFu, y1 = ((scissor >> 48) & 0x7FFu) + 1u;
        auto xyz = [&](uint64_t x, uint64_t y)
        { return ((ofx + (x << 4)) & 0xFFFFu) | (((ofy + (y << 4)) & 0xFFFFu) << 16); };
        // Sprite over the scissor rectangle: Z test ALWAYS, Z writes masked, then restore.
        putAd(w, {{0x00, 6u | (static_cast<uint64_t>(ctx) << 9)},
                  {0x01, rgba | (static_cast<uint64_t>(floatBits(1.0f)) << 32)},
                  {static_cast<uint8_t>(0x47 + ctx), (1ull << 16) | (1ull << 17)},
                  {static_cast<uint8_t>(0x4E + ctx), zbuf | (1ull << 32)},
                  {0x05, xyz(x0, y0)},
                  {0x05, xyz(x1, y1)},
                  {0x00, prim},
                  {0x01, rgbaq},
                  {static_cast<uint8_t>(0x47 + ctx), test},
                  {static_cast<uint8_t>(0x4E + ctx), zbuf}});
    }

    void readFifo(uint32_t bytes)
    {
        if (!active() || bytes == 0u)
            return;
        Writer &w = writer();
        std::lock_guard<std::mutex> lock(w.mutex);
        if (w.phase != Phase::Recording)
            return;
        // The EE drains the FIFO in arbitrary byte counts; PCSX2 reads whole qwords.
        w.fifoBytes += bytes;
        const uint32_t qwc = w.fifoBytes / 16u;
        if (qwc == 0u)
            return;
        w.fifoBytes -= qwc * 16u;
        putValue<uint8_t>(w, 2u);
        putValue<uint32_t>(w, qwc);
    }

    void vsync(const GSRegisters *priv, const GSPresentationRequest &request, GSRasterBackend *backend)
    {
        if (!active())
            return;
        Writer &w = writer();
        std::lock_guard<std::mutex> lock(w.mutex);
        if (w.phase == Phase::Waiting)
        {
            const uint32_t frame = hostFrame();
            ++s_presentations;
            if (frame < w.startFrame)
                return;
            if (!begin(w, priv, request, backend))
            {
                finish(w);
                return;
            }
            w.phase = Phase::Recording;
            std::fprintf(stderr, "[gs-dump] recording %u frame(s) from host frame %u to %s\n",
                         w.frameCount, frame, w.path.c_str());
            return;
        }
        if (w.phase != Phase::Recording)
            return;

        uint8_t privRegs[kPrivRegsSize];
        buildPrivRegs(priv, request, privRegs);
        putValue<uint8_t>(w, 3u);
        put(w, privRegs, sizeof(privRegs));
        putValue<uint8_t>(w, 1u);
        putValue<uint8_t>(w, static_cast<uint8_t>(request.field & 1u));
        if (++w.framesWritten >= w.frameCount && w.phase == Phase::Recording)
            finish(w);
    }
}
