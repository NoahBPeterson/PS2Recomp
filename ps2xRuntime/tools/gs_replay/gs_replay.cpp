// ps2x_gs_replay: replays a PCSX2 GS dump (.gs, raw) through the runtime's GS frontend and
// a raster backend, headless, and writes each presented frame as a PNG.
// Format: tickets/notes/gs-dump-format.md.
//
//   ps2x_gs_replay <dump.gs> [--backend cpu|parallel] [--out <dir>] [--frames N] [--quiet]
//                  [--ref <dir> [--tol T] [--max-bad N]]
// --ref compares every frame_NNNN.png with the same file in <dir>: a pixel is bad when any RGB
// channel differs by more than T (default 0); a frame fails with more than N bad pixels
// (default 0). Exit status: 0 ok, 1 replay error, 2 usage, 3 reference mismatch.
//
// Strict (bit-exact) checks of the whole GS state, not only the presented frame:
//   --hash-out <file>   after every frame (and every --hash-every N packets) writes one line with
//                       a 64-bit hash of all 4 MiB of VRAM, the CSR and the presented pixels
//   --hash-ref <file>   compares those lines with a file written by --hash-out; exit 3 on the
//                       first difference (reported with its line)
// Hashing time is excluded from the reported replay time.
//
// The CPU backend honours the runtime's settings: PS2X_GS_THREADS=N raster workers (default 3
// with PS2X_GS_ASYNC=1, else 1 = rasterize on the replay thread), PS2X_GS_ASYNC=1 asynchronous command execution. Output
// must be bit-identical for every setting (check with --hash-ref).

#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/ps2_memory.h"
#if defined(PS2X_GS_PARALLEL)
#include "runtime/gs/gs_parallel_backend.h"
#endif

#include "raylib.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kVramSize = 4u * 1024u * 1024u;
    constexpr uint32_t kPrivRegsSize = 8192u;

    struct Dump
    {
        std::vector<uint8_t> bytes;
        std::string serial;
        uint32_t stateVersion = 0u;
        const uint8_t *state = nullptr;
        size_t stateSize = 0u;
        const uint8_t *privRegs = nullptr;
        size_t packetOffset = 0u;
    };

    struct Reader
    {
        const uint8_t *p;
        size_t size;
        size_t pos = 0u;
        bool ok = true;

        bool need(size_t n)
        {
            if (pos + n > size)
                ok = false;
            return ok;
        }
        template <typename T>
        T get()
        {
            T v{};
            if (need(sizeof(T)))
                std::memcpy(&v, p + pos, sizeof(T));
            pos += sizeof(T);
            return v;
        }
        const uint8_t *take(size_t n)
        {
            if (!need(n))
                return nullptr;
            const uint8_t *r = p + pos;
            pos += n;
            return r;
        }
    };

    bool loadDump(const char *path, Dump &dump)
    {
        FILE *f = std::fopen(path, "rb");
        if (!f)
        {
            std::fprintf(stderr, "cannot open %s\n", path);
            return false;
        }
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        dump.bytes.resize(static_cast<size_t>(size));
        const size_t got = std::fread(dump.bytes.data(), 1, dump.bytes.size(), f);
        std::fclose(f);
        if (got != dump.bytes.size())
            return false;
        if (dump.bytes.size() >= 6 && (std::memcmp(dump.bytes.data(), "\xFD" "7zXZ", 6) == 0 ||
                                       std::memcmp(dump.bytes.data(), "\x28\xB5\x2F\xFD", 4) == 0))
        {
            std::fprintf(stderr, "%s is xz/zstd compressed; decompress it first (xz -dk / zstd -d)\n", path);
            return false;
        }

        Reader r{dump.bytes.data(), dump.bytes.size()};
        const uint32_t magic = r.get<uint32_t>();
        const uint32_t headerSize = r.get<uint32_t>();
        if (magic == 0xFFFFFFFFu)
        {
            const uint8_t *header = r.take(headerSize);
            if (!header || headerSize < 36u)
                return false;
            uint32_t h[9];
            std::memcpy(h, header, sizeof(h));
            if (h[3] && h[2] + h[3] <= headerSize)
                dump.serial.assign(reinterpret_cast<const char *>(header + h[2]), h[3]);
            dump.stateSize = h[1];
        }
        else
        {
            dump.stateSize = headerSize; // old style: crc, state size, state
        }
        dump.state = r.take(dump.stateSize);
        dump.privRegs = r.take(kPrivRegsSize);
        dump.packetOffset = r.pos;
        if (!r.ok || dump.stateSize < 4u)
            return false;
        std::memcpy(&dump.stateVersion, dump.state, 4);
        return true;
    }

    uint64_t privReg(const uint8_t *regs, uint32_t offset)
    {
        uint64_t v;
        std::memcpy(&v, regs + offset, sizeof(v));
        return v;
    }

    void applyPrivRegs(GSRegisters &priv, const uint8_t *regs)
    {
        priv.pmode = privReg(regs, 0x000);
        priv.smode1 = privReg(regs, 0x010);
        priv.smode2 = privReg(regs, 0x020);
        priv.srfsh = privReg(regs, 0x030);
        priv.synch1 = privReg(regs, 0x040);
        priv.synch2 = privReg(regs, 0x050);
        priv.syncv = privReg(regs, 0x060);
        priv.dispfb1 = privReg(regs, 0x070);
        priv.display1 = privReg(regs, 0x080);
        priv.dispfb2 = privReg(regs, 0x090);
        priv.display2 = privReg(regs, 0x0A0);
        priv.extbuf = privReg(regs, 0x0B0);
        priv.extdata = privReg(regs, 0x0C0);
        priv.extwrite = privReg(regs, 0x0D0);
        priv.bgcolor = privReg(regs, 0x0E0);
        priv.csr.store(privReg(regs, 0x1000));
        priv.imr = privReg(regs, 0x1010);
        priv.busdir = privReg(regs, 0x1040);
        priv.siglblid = privReg(regs, 0x1080);
    }

    // The GSState::Freeze blob (versions 7..9).
    struct FrozenState
    {
        uint64_t env[15]{};
        uint64_t ctx[2][12]{};
        uint64_t rgbaq = 0u;
        uint64_t st = 0u;
        uint32_t uv = 0u;
        uint32_t fog = 0u;
        const uint8_t *vram = nullptr;
        float q = 1.0f;
    };

    bool parseState(const Dump &dump, FrozenState &out)
    {
        Reader r{dump.state, dump.stateSize};
        r.get<uint32_t>(); // version
        for (uint64_t &v : out.env)
            v = r.get<uint64_t>();
        for (auto &c : out.ctx)
            for (uint64_t &v : c)
                v = r.get<uint64_t>();
        out.rgbaq = r.get<uint64_t>();
        out.st = r.get<uint64_t>();
        out.uv = r.get<uint32_t>();
        out.fog = r.get<uint32_t>();
        r.get<uint64_t>(); // XYZ
        r.get<uint64_t>(); // obsolete GIFReg
        r.get<uint32_t>(); // transfer x
        r.get<uint32_t>(); // transfer y
        if (dump.stateVersion >= 9u)
            r.take(61u); // in-flight transfer state
        out.vram = r.take(kVramSize);
        r.take(4u * 20u); // GIF paths: whole-packet dumps leave them idle
        out.q = r.get<float>();
        return r.ok && out.vram && dump.stateVersion >= 7u;
    }

    // Seeds the frozen GS registers through an A+D packet (TRXDIR excluded: writing it would
    // start a transfer), then a packed ST that also restores the internal Q.
    void writeSeedPacket(GS &gs, const FrozenState &state)
    {
        const auto &env = state.env;
        const auto &ctx = state.ctx;
        const uint64_t rgbaq = state.rgbaq, st = state.st;
        const uint32_t uv = state.uv, fog = state.fog;
        const float q = state.q;

        std::vector<std::pair<uint8_t, uint64_t>> writes;
        // env order in the state: PRIM, PRMODECONT, TEXCLUT, SCANMSK, TEXA, FOGCOL, DIMX, DTHE,
        // COLCLAMP, PABE, BITBLTBUF, TRXDIR, TRXPOS, TRXREG, TRXREG.
        const uint8_t envRegs[15] = {0x00, 0x1A, 0x1C, 0x22, 0x3B, 0x3D, 0x44, 0x45, 0x46, 0x49, 0x50, 0x53, 0x51, 0x52, 0x52};
        for (int i = 1; i < 14; ++i)
            if (envRegs[i] != 0x53)
                writes.emplace_back(envRegs[i], env[i]);
        const uint32_t currentCtx = static_cast<uint32_t>((env[0] >> 9) & 1u);
        const uint8_t ctxRegs[12] = {0x18, 0x06, 0x14, 0x08, 0x34, 0x36, 0x40, 0x42, 0x47, 0x4A, 0x4C, 0x4E};
        for (uint32_t c = 0; c < 2; ++c)
            for (int i = 0; i < 12; ++i)
            {
                uint64_t value = ctx[c][i];
                if (ctxRegs[i] == 0x06 && c != currentCtx)
                    value &= ~(7ull << 61); // like Defrost, only the current context reloads the CLUT
                if (ctxRegs[i] == 0x18)
                    value &= 0x0000FFFF0000FFFFull;
                writes.emplace_back(static_cast<uint8_t>(ctxRegs[i] + c), value);
            }
        writes.emplace_back(0x00, env[0]);
        writes.emplace_back(0x01, rgbaq);
        writes.emplace_back(0x03, uv);
        writes.emplace_back(0x0A, static_cast<uint64_t>(fog & 0xFFu) << 56);

        std::vector<uint64_t> packet;
        packet.push_back(writes.size() | (1ull << 60));
        packet.push_back(0x0Eu);
        for (const auto &[addr, value] : writes)
        {
            packet.push_back(value);
            packet.push_back(addr);
        }
        // Packed ST sets ST and the internal Q (the tag first resets Q to 1.0).
        uint32_t qBits;
        std::memcpy(&qBits, &q, 4);
        packet.push_back(1u | (1ull << 15) | (1ull << 60));
        packet.push_back(0x02u);
        packet.push_back(st);
        packet.push_back(qBits);
        gs.processGIFPacket(reinterpret_cast<const uint8_t *>(packet.data()), static_cast<uint32_t>(packet.size() * 8u), 3u);
    }

    bool writePng(const std::string &path, std::vector<uint8_t> &rgba, uint32_t width, uint32_t height)
    {
        for (size_t i = 3; i < rgba.size(); i += 4)
            rgba[i] = 0xFF;
        Image image{};
        image.data = rgba.data();
        image.width = static_cast<int>(width);
        image.height = static_cast<int>(height);
        image.mipmaps = 1;
        image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
        return ExportImage(image, path.c_str());
    }

    // Returns the number of pixels whose RGB differs by more than tol, or -1 when the reference
    // is missing or has another size.
    int64_t compareWithReference(const std::string &refPath, const std::vector<uint8_t> &rgba,
                                 uint32_t width, uint32_t height, int tol)
    {
        if (!FileExists(refPath.c_str()))
            return -1;
        Image ref = LoadImage(refPath.c_str());
        if (!ref.data)
            return -1;
        ImageFormat(&ref, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
        int64_t bad = -1;
        if (static_cast<uint32_t>(ref.width) == width && static_cast<uint32_t>(ref.height) == height)
        {
            bad = 0;
            const auto *r = static_cast<const uint8_t *>(ref.data);
            for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i)
                for (int c = 0; c < 3; ++c)
                    if (std::abs(int(rgba[i * 4 + c]) - int(r[i * 4 + c])) > tol)
                    {
                        ++bad;
                        break;
                    }
        }
        UnloadImage(ref);
        return bad;
    }

    // 64-bit hash over 8-byte words (4 independent lanes, then mixed): fast enough to run on
    // 4 MiB after every frame without dominating the replay time.
    uint64_t hashBytes(const uint8_t *data, size_t size)
    {
        constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull;
        uint64_t h[4] = {0x243F6A8885A308D3ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull, 0x082EFA98EC4E6C89ull};
        size_t i = 0;
        for (; i + 32u <= size; i += 32u)
            for (int l = 0; l < 4; ++l)
            {
                uint64_t w;
                std::memcpy(&w, data + i + l * 8, 8);
                h[l] = (h[l] ^ w) * kMul;
                h[l] ^= h[l] >> 29;
            }
        uint64_t out = size;
        for (; i < size; ++i)
            out = (out ^ data[i]) * kMul;
        for (uint64_t v : h)
        {
            out = (out ^ v) * kMul;
            out ^= out >> 31;
        }
        return out;
    }

    void usage()
    {
        std::fprintf(stderr, "usage: ps2x_gs_replay <dump.gs> [--backend cpu|parallel] [--out <dir>] [--frames N] [--quiet]\n"
                             "                      [--ref <dir> [--tol T] [--max-bad N]] [--vram-at <packet> <file>] [--vram-frame <N> <file>]\n"
                             "                      [--hash-out <file>] [--hash-ref <file>] [--hash-every <packets>]\n");
    }
}

int main(int argc, char **argv)
{
    std::string dumpPath, outDir = ".", backendName = "cpu", refDir;
    uint32_t maxFrames = UINT32_MAX;
    int tol = 0;
    int64_t maxBad = 0;
    bool quiet = false;
    uint64_t vramAtPacket = UINT64_MAX; // --vram-at N file: raw 4 MiB VRAM before packet N
    std::string vramAtPath;
    uint32_t vramFrame = UINT32_MAX; // --vram-frame N file: VRAM (any backend) at vsync N, before presenting it
    std::string vramFramePath;
    std::string hashOutPath, hashRefPath;
    uint64_t hashEvery = 0u;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc)
            backendName = argv[++i];
        else if (arg == "--out" && i + 1 < argc)
            outDir = argv[++i];
        else if (arg == "--frames" && i + 1 < argc)
            maxFrames = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (arg == "--ref" && i + 1 < argc)
            refDir = argv[++i];
        else if (arg == "--tol" && i + 1 < argc)
            tol = std::atoi(argv[++i]);
        else if (arg == "--max-bad" && i + 1 < argc)
            maxBad = std::atoll(argv[++i]);
        else if (arg == "--vram-at" && i + 2 < argc)
        {
            vramAtPacket = std::strtoull(argv[++i], nullptr, 10);
            vramAtPath = argv[++i];
        }
        else if (arg == "--vram-frame" && i + 2 < argc)
        {
            vramFrame = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            vramFramePath = argv[++i];
        }
        else if (arg == "--hash-out" && i + 1 < argc)
            hashOutPath = argv[++i];
        else if (arg == "--hash-ref" && i + 1 < argc)
            hashRefPath = argv[++i];
        else if (arg == "--hash-every" && i + 1 < argc)
            hashEvery = std::strtoull(argv[++i], nullptr, 10);
        else if (arg == "--quiet")
            quiet = true;
        else if (!arg.empty() && arg[0] != '-' && dumpPath.empty())
            dumpPath = arg;
        else
        {
            usage();
            return 2;
        }
    }
    if (dumpPath.empty())
    {
        usage();
        return 2;
    }
    SetTraceLogLevel(LOG_WARNING);

    const auto t0 = std::chrono::steady_clock::now();
    Dump dump;
    if (!loadDump(dumpPath.c_str(), dump))
    {
        std::fprintf(stderr, "%s: not a readable GS dump\n", dumpPath.c_str());
        return 1;
    }

    std::vector<uint8_t> vram(kVramSize, 0u);
    FrozenState state;
    if (!parseState(dump, state))
    {
        std::fprintf(stderr, "%s: unsupported state (version %u, %zu bytes)\n", dumpPath.c_str(), dump.stateVersion, dump.stateSize);
        return 1;
    }
    std::memcpy(vram.data(), state.vram, kVramSize);
    GS gs;
    GSRegisters priv{};
    applyPrivRegs(priv, dump.privRegs);
    gs.init(vram.data(), kVramSize, &priv);
    GSRasterBackend *backend = nullptr; // owned by gs; used for the state hashes
    if (backendName == "parallel")
    {
#if defined(PS2X_GS_PARALLEL)
        auto parallel = CreateParallelGSBackend(priv);
        backend = parallel.get();
        gs.setRasterBackend(std::move(parallel));
#else
        std::fprintf(stderr, "built without PS2X_GS_PARALLEL\n");
        return 1;
#endif
    }
    else if (backendName == "cpu")
    {
        auto cpu = std::make_unique<GSCpuBackend>(GSCpuBackend::DefaultRasterThreads(), GSCpuBackend::DefaultAsyncCommands());
        backend = cpu.get();
        gs.setRasterBackend(std::move(cpu));
    }
    else
    {
        usage();
        return 2;
    }
    writeSeedPacket(gs, state);

    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);
    const auto t1 = std::chrono::steady_clock::now();

    const bool hashing = !hashOutPath.empty() || !hashRefPath.empty();
    FILE *hashOut = hashOutPath.empty() ? nullptr : std::fopen(hashOutPath.c_str(), "w");
    if (!hashOutPath.empty() && !hashOut)
    {
        std::fprintf(stderr, "cannot write %s\n", hashOutPath.c_str());
        return 1;
    }
    std::vector<std::string> hashRef;
    if (!hashRefPath.empty())
    {
        FILE *hf = std::fopen(hashRefPath.c_str(), "r");
        if (!hf)
        {
            std::fprintf(stderr, "cannot read %s\n", hashRefPath.c_str());
            return 1;
        }
        char line[512];
        while (std::fgets(line, sizeof(line), hf))
            hashRef.emplace_back(line);
        std::fclose(hf);
    }
    size_t hashLines = 0u, hashMismatches = 0u;
    double hashMs = 0.0;
    // Records one state line. VRAM is read through the backend's Sync + SnapshotVram, so an
    // asynchronous backend is hashed only after it has finished.
    std::vector<uint8_t> snap;
    auto recordHash = [&](const char *what, uint64_t index, const std::vector<uint8_t> *frame)
    {
        const auto h0 = std::chrono::steady_clock::now();
        backend->Sync(GSSyncReason::DebugReadback);
        backend->SnapshotVram(snap);
        const uint64_t vramHash = hashBytes(snap.data(), snap.size());
        const uint64_t frameHash = frame ? hashBytes(frame->data(), frame->size()) : 0u;
        char line[256];
        std::snprintf(line, sizeof(line), "%s %llu vram=%016llx csr=%016llx frame=%016llx\n", what,
                      static_cast<unsigned long long>(index), static_cast<unsigned long long>(vramHash),
                      static_cast<unsigned long long>(priv.csr.load()), static_cast<unsigned long long>(frameHash));
        if (hashOut)
            std::fputs(line, hashOut);
        if (!hashRefPath.empty())
        {
            if (hashLines >= hashRef.size() || hashRef[hashLines] != line)
            {
                if (hashMismatches++ == 0u)
                    std::printf("HASH MISMATCH at line %zu: got %s  expected %s", hashLines + 1u, line,
                                hashLines < hashRef.size() ? hashRef[hashLines].c_str() : "(end of file)\n");
            }
        }
        ++hashLines;
        hashMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    };

    Reader r{dump.bytes.data(), dump.bytes.size(), dump.packetOffset};
    uint32_t frames = 0u, transfers = 0u, fifoReads = 0u, mismatches = 0u;
    uint64_t gifBytes = 0u;
    std::vector<uint8_t> fifo, pixels;
    uint64_t packetIndex = 0u;
    while (r.pos < r.size && frames < maxFrames)
    {
        if (packetIndex++ == vramAtPacket)
        {
            gs.shutdownBackend();
            if (FILE *vf = std::fopen(vramAtPath.c_str(), "wb"))
            {
                std::fwrite(vram.data(), 1, vram.size(), vf);
                std::fclose(vf);
            }
            std::printf("wrote VRAM before packet %llu to %s\n", static_cast<unsigned long long>(vramAtPacket), vramAtPath.c_str());
            return 0;
        }
        if (hashing && hashEvery && packetIndex % hashEvery == 0u)
            recordHash("packet", packetIndex, nullptr);
        const uint8_t type = r.get<uint8_t>();
        if (type == 0u)
        {
            const uint8_t path = r.get<uint8_t>();
            const uint32_t size = r.get<uint32_t>();
            const uint8_t *data = r.take(size);
            if (!data)
                break;
            gs.processGIFPacket(data, size, path == 0u ? 1u : path);
            ++transfers;
            gifBytes += size;
        }
        else if (type == 1u)
        {
            const uint8_t field = r.get<uint8_t>();
            if (!r.ok)
                break;
            uint64_t csr = priv.csr.load();
            csr = (csr & ~(1ull << 13)) | (static_cast<uint64_t>(field & 1u) << 13);
            priv.csr.store(csr);
            priv.vsyncTick.store(frames + 1u);
            if (frames == vramFrame)
            {
                gs.refreshDisplaySnapshot();
                uint32_t size = 0u;
                const uint8_t *snap = gs.lockDisplaySnapshot(size);
                if (FILE *vf = snap ? std::fopen(vramFramePath.c_str(), "wb") : nullptr)
                {
                    std::fwrite(snap, 1, size, vf);
                    std::fclose(vf);
                }
                gs.unlockDisplaySnapshot();
                std::printf("wrote VRAM at vsync %u to %s\n", vramFrame, vramFramePath.c_str());
            }
            backend->Sync(GSSyncReason::DebugReadback); // an asynchronous backend presents this vsync's VRAM
            gs.latchHostPresentationFrame();
            uint32_t width = 0u, height = 0u;
            const std::string name = "frame_" + std::to_string(10000 + frames).substr(1) + ".png";
            const std::string png = outDir + "/" + name;
            if (gs.copyLatchedHostPresentationFrame(pixels, width, height) && width && height)
            {
                if (!writePng(png, pixels, width, height))
                    std::fprintf(stderr, "failed to write %s\n", png.c_str());
                else if (!quiet)
                    std::printf("%s %ux%u\n", png.c_str(), width, height);
                if (hashing)
                    recordHash("frame", frames, &pixels);
                if (!refDir.empty())
                {
                    const int64_t bad = compareWithReference(refDir + "/" + name, pixels, width, height, tol);
                    if (bad < 0 || bad > maxBad)
                        ++mismatches;
                    if (bad < 0)
                        std::printf("MISMATCH %s: reference missing or size differs (%ux%u)\n", name.c_str(), width, height);
                    else
                        std::printf("%s %s: %lld pixel(s) differ by more than %d (%.3f%%)\n", bad > maxBad ? "MISMATCH" : "match",
                                    name.c_str(), static_cast<long long>(bad), tol, 100.0 * double(bad) / (double(width) * height));
                }
            }
            else
            {
                if (hashing)
                    recordHash("frame", frames, nullptr);
                std::fprintf(stderr, "frame %u: no presentation frame\n", frames);
                if (!refDir.empty())
                    ++mismatches;
            }
            ++frames;
        }
        else if (type == 2u)
        {
            const uint32_t qwc = r.get<uint32_t>();
            if (!r.ok)
                break;
            fifo.resize(static_cast<size_t>(qwc) * 16u);
            gs.consumeLocalToHostBytes(fifo.data(), static_cast<uint32_t>(fifo.size()));
            ++fifoReads;
        }
        else if (type == 3u)
        {
            const uint8_t *regs = r.take(kPrivRegsSize);
            if (!regs)
                break;
            applyPrivRegs(priv, regs);
        }
        else
        {
            std::fprintf(stderr, "unknown packet type %u at offset %zu\n", type, r.pos - 1u);
            return 1;
        }
    }
    const auto t2 = std::chrono::steady_clock::now();
    const double loadMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double replayMs = std::chrono::duration<double, std::milli>(t2 - t1).count() - hashMs;
    std::printf("replayed %u frame(s), %u transfers (%.1f MiB GIF), %u FIFO reads, serial '%s', state v%u: "
                "load %.0f ms, replay %.0f ms (%.1f ms/frame) [%s]\n",
                frames, transfers, double(gifBytes) / (1024.0 * 1024.0), fifoReads, dump.serial.c_str(), dump.stateVersion,
                loadMs, replayMs, frames ? replayMs / frames : 0.0, backendName.c_str());
    gs.shutdownBackend();
    if (hashOut)
        std::fclose(hashOut);
    if (!frames)
        return 1;
    if (!hashRefPath.empty())
    {
        if (hashLines != hashRef.size())
            ++hashMismatches;
        std::printf("%s: %zu/%zu state hash(es) match %s\n", hashMismatches ? "HASH FAIL" : "HASH PASS",
                    hashLines - std::min(hashLines, hashMismatches), hashRef.size(), hashRefPath.c_str());
        if (hashMismatches)
            return 3;
    }
    if (!refDir.empty())
    {
        std::printf("%s: %u/%u frame(s) match %s\n", mismatches ? "FAIL" : "PASS", frames - mismatches, frames, refDir.c_str());
        return mismatches ? 3 : 0;
    }
    return 0;
}
