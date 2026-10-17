// ps2x_vu1_replay: replays a VU1 run trace (runtime/ps2_vu1_trace.h, recorded with PS2X_VU1_TRACE)
// through the runtime's VU1 interpreter and checks every run bit-exactly against the recording:
// final VU1 state, data memory and the PATH1 (XGKICK) packets it sent. Also a benchmark.
//
//   ps2x_vu1_replay <trace> [--repeat N] [--examples N] [--quiet] [--repin <out trace>]
// --repeat replays the whole trace N times (timing only; results are checked every time).
// --repin writes the trace with every run's results (VU1 state, data hash, PATH1 packets) replaced by
// the interpreter's (blocks off), for an intended behaviour change. A resume whose recorded inputs
// were the previous run's results gets that run's new results as inputs.
// Exit status: 0 all runs match, 1 mismatch, 2 usage / bad trace.

#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu1_trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <time.h>
#endif

namespace
{
    // The run cycle cap the trace was recorded with (ps2_vu1_trace::runCycleCap).
    uint32_t g_runCycleCap = 65536u;
    using namespace ps2_vu1_trace;

    struct Record
    {
        Vu1TraceRecordHeader header{};
        const uint8_t *code = nullptr; // last included code at or before this record
        const uint8_t *data = nullptr;
        VU1State before{}, after{};
        uint64_t dataAfterHash = 0u, gifHash = 0u;
        uint32_t gifPackets = 0u, gifBytes = 0u;
        size_t beforeOffset = 0u; // in the file: before, after, the data hash, the GIF hash/packets/bytes follow
    };

    bool load(const char *path, std::vector<uint8_t> &bytes, std::vector<Record> &records)
    {
        FILE *f = std::fopen(path, "rb");
        if (!f)
            return false;
        std::fseek(f, 0, SEEK_END);
        bytes.resize(static_cast<size_t>(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        const bool read = std::fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
        std::fclose(f);
        if (!read || bytes.size() < 12u || std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0)
            return false;
        uint32_t version = 0u;
        std::memcpy(&version, bytes.data() + 8, 4);
        if (!readableVersion(version))
            return false;
        g_runCycleCap = runCycleCap(version);
        size_t pos = 12u;
        const uint8_t *code = nullptr;
        auto take = [&](void *dst, size_t n)
        {
            if (pos + n > bytes.size())
                return false;
            if (dst)
                std::memcpy(dst, bytes.data() + pos, n);
            pos += n;
            return true;
        };
        while (pos < bytes.size())
        {
            Record r;
            if (!take(&r.header, sizeof(r.header)) || r.header.stateSize != sizeof(VU1State))
                return false;
            if (r.header.codeIncluded)
            {
                code = bytes.data() + pos;
                if (!take(nullptr, kMemSize))
                    return false;
            }
            if (!code)
                return false;
            r.code = code;
            r.data = bytes.data() + pos;
            r.beforeOffset = pos + kMemSize;
            if (!take(nullptr, kMemSize) || !take(&r.before, sizeof(VU1State)) || !take(&r.after, sizeof(VU1State)) ||
                !take(&r.dataAfterHash, 8) || !take(&r.gifHash, 8) || !take(&r.gifPackets, 4) || !take(&r.gifBytes, 4))
                return false;
            records.push_back(r);
        }
        return true;
    }

    template <typename T>
    bool same(const T &a, const T &b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

    // Field by field (the raw structs carry padding).
    std::string stateDiff(const VU1State &want, const VU1State &got)
    {
        std::string out;
        char buf[160];
        auto note = [&](const char *what)
        {
            if (out.size() < 400)
                out += what;
        };
        for (int r = 0; r < 32; ++r)
            for (int c = 0; c < 4; ++c)
                if (!same(want.vf[r][c], got.vf[r][c]))
                {
                    uint32_t w, g;
                    std::memcpy(&w, &want.vf[r][c], 4);
                    std::memcpy(&g, &got.vf[r][c], 4);
                    std::snprintf(buf, sizeof(buf), " vf%02d.%c=%08x(want %08x)", r, "xyzw"[c], g, w);
                    note(buf);
                }
        for (int r = 0; r < 16; ++r)
            if (want.vi[r] != got.vi[r])
            {
                std::snprintf(buf, sizeof(buf), " vi%02d=%x(want %x)", r, got.vi[r], want.vi[r]);
                note(buf);
            }
        for (int c = 0; c < 4; ++c)
            if (!same(want.acc[c], got.acc[c]))
            {
                std::snprintf(buf, sizeof(buf), " acc.%c", "xyzw"[c]);
                note(buf);
            }
#define CHECK(field)                                                                   \
    if (!same(want.field, got.field))                                                  \
    {                                                                                  \
        std::snprintf(buf, sizeof(buf), " " #field "=%llx(want %llx)",                 \
                      (unsigned long long)toU64(got.field), (unsigned long long)toU64(want.field)); \
        note(buf);                                                                     \
    }
        auto toU64 = [](auto v)
        {
            uint64_t x = 0;
            std::memcpy(&x, &v, sizeof(v));
            return x;
        };
        CHECK(q) CHECK(p) CHECK(i) CHECK(r) CHECK(pc) CHECK(mac) CHECK(clip) CHECK(status) CHECK(cycles)
        CHECK(ebit) CHECK(haltAfterDelaySlot) CHECK(dBitEnabled) CHECK(tBitEnabled) CHECK(stoppedByD)
        CHECK(stoppedByT) CHECK(top) CHECK(itop) CHECK(branchPending) CHECK(branchTarget) CHECK(branchDelay)
#undef CHECK
        return out;
    }
}

namespace
{
    // CPU time of this thread: steadier than wall time on a loaded machine (preemption does not count).
    std::chrono::nanoseconds threadCpuNow()
    {
#if defined(_WIN32)
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch());
#else
        timespec ts{};
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
        return std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec);
#endif
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: ps2x_vu1_replay <trace> [--repeat N] [--examples N] [--quiet] [--repin <out>]\n");
        return 2;
    }
    int repeat = 1, examples = 5;
    bool quiet = false;
    const char *repinPath = nullptr;
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--repeat" && i + 1 < argc)
            repeat = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--examples" && i + 1 < argc)
            examples = std::atoi(argv[++i]);
        else if (arg == "--quiet")
            quiet = true;
        else if (arg == "--repin" && i + 1 < argc)
            repinPath = argv[++i];
        else
        {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }

    if (repinPath)
    {
        repeat = 1;
#if defined(_WIN32)
        _putenv_s("PS2X_VU1_NO_AOT", "1"); // the new results come from the interpreter
#else
        setenv("PS2X_VU1_NO_AOT", "1", 1);
#endif
    }
    std::vector<uint8_t> bytes;
    std::vector<Record> records;
    if (!load(argv[1], bytes, records) || records.empty())
    {
        std::fprintf(stderr, "%s: not a readable VU1 trace (v%u)\n", argv[1], kVersion);
        return 2;
    }
    static_assert(PS2_VU1_CODE_SIZE == kMemSize && PS2_VU1_DATA_SIZE == kMemSize);

    PS2Memory memory;
    if (!memory.initialize())
    {
        std::fprintf(stderr, "PS2Memory init failed\n");
        return 2;
    }
    uint64_t gifHash = 0u;
    uint32_t gifPackets = 0u, gifBytes = 0u;
    GifArbiter arbiter([&](GifPathId, const uint8_t *data, uint32_t size)
                       {
                           gifHash = fnv1a(data, size, gifHash);
                           ++gifPackets;
                           gifBytes += size;
                       });
    memory.setGifArbiter(&arbiter);
    GS gs;
    VU1Interpreter vu1(VU1Interpreter::Unit::VU1);
    vu1.reset();
    uint8_t *const code = memory.getVU1Code();
    uint8_t *const data = memory.getVU1Data();

    uint64_t mismatches = 0u, resumes = 0u, cycles = 0u, codeLoads = 0u;
    std::vector<VU1State> oldAfter;
    std::vector<uint64_t> oldDataHash;
    for (const Record &r : records)
    {
        oldAfter.push_back(r.after);
        oldDataHash.push_back(r.dataAfterHash);
    }
    std::chrono::nanoseconds vuTime{}; // thread CPU time inside execute/resume only (excludes replay bookkeeping)
    const auto t0 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < repeat; ++pass)
    {
        const uint8_t *loadedCode = nullptr;
        vu1.reset(); // every pass starts like the first: no pipeline state left from the last run
        for (size_t index = 0; index < records.size(); ++index)
        {
            Record &r = records[index];
            if (repinPath && index > 0u && r.header.kind == static_cast<uint32_t>(RunKind::Resume))
            {
                // Chained to the previous run (recorded inputs = its recorded results): take its
                // new results instead. (records[index - 1] already holds them; its old ones are in
                // the copy made before patching.)
                const Record &prev = records[index - 1u];
                if (same(r.before, oldAfter[index - 1u]))
                {
                    r.before = prev.after;
                    std::memcpy(bytes.data() + r.beforeOffset, &r.before, sizeof(VU1State));
                }
                if (fnv1a(r.data, kMemSize) == oldDataHash[index - 1u] && r.data != prev.data)
                    std::memcpy(const_cast<uint8_t *>(r.data), data, kMemSize); // data still holds prev's result
            }
            if (r.code != loadedCode)
            {
                std::memcpy(code, r.code, kMemSize);
                memory.markVU1CodeModified();
                loadedCode = r.code;
                ++codeLoads;
            }
            std::memcpy(data, r.data, kMemSize);
            gifHash = 0xcbf29ce484222325ull;
            gifPackets = gifBytes = 0u;
            if (r.header.kind == static_cast<uint32_t>(RunKind::Execute))
            {
                vu1.restoreState(r.before);
                const auto start = threadCpuNow();
                vu1.execute(code, kMemSize, data, kMemSize, gs, &memory, r.header.startPc, r.header.top, r.header.itop, g_runCycleCap);
                vuTime += threadCpuNow() - start;
            }
            else
            {
                // Continues from the previous run's pipelines (replayed just before; empty after a
                // program that ended). The recorded state equals the previous run's result when that
                // matched, and seeds the chain when the trace starts with a resume.
                ++resumes;
                vu1.restoreState(r.before);
                const auto start = threadCpuNow();
                vu1.resume(code, kMemSize, data, kMemSize, gs, &memory, r.header.top, r.header.itop, g_runCycleCap);
                vuTime += threadCpuNow() - start;
            }
            cycles += r.after.cycles - r.before.cycles;
            const std::string diff = stateDiff(r.after, vu1.state());
            const uint64_t dataHash = fnv1a(data, kMemSize);
            const bool ok = diff.empty() && dataHash == r.dataAfterHash && gifHash == r.gifHash &&
                            gifPackets == r.gifPackets && gifBytes == r.gifBytes;
            if (repinPath)
            {
                r.after = vu1.state();
                r.dataAfterHash = dataHash;
                r.gifHash = gifHash;
                r.gifPackets = gifPackets;
                r.gifBytes = gifBytes;
                uint8_t *const at = bytes.data() + r.beforeOffset + sizeof(VU1State);
                std::memcpy(at, &r.after, sizeof(VU1State));
                std::memcpy(at + sizeof(VU1State), &dataHash, 8);
                std::memcpy(at + sizeof(VU1State) + 8, &gifHash, 8);
                std::memcpy(at + sizeof(VU1State) + 16, &gifPackets, 4);
                std::memcpy(at + sizeof(VU1State) + 20, &gifBytes, 4);
            }
            if (!ok)
            {
                if (static_cast<int64_t>(mismatches) < examples)
                    std::printf("MISMATCH run %zu (%s pc=0x%x):%s%s%s\n", index,
                                r.header.kind ? "resume" : "execute", r.header.startPc, diff.c_str(),
                                dataHash != r.dataAfterHash ? " data" : "",
                                (gifHash != r.gifHash || gifPackets != r.gifPackets || gifBytes != r.gifBytes)
                                    ? (" gif(" + std::to_string(gifPackets) + " pkts/" + std::to_string(gifBytes) + " B, want " +
                                       std::to_string(r.gifPackets) + "/" + std::to_string(r.gifBytes) + ")").c_str()
                                    : "");
                ++mismatches;
            }
        }
    }
    const double totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double ms = std::chrono::duration<double, std::milli>(vuTime).count();
    const uint64_t runs = records.size() * static_cast<uint64_t>(repeat);
    std::printf("%s: %llu runs (%zu per pass x %d, %llu resumes), %llu mismatches; VU1 %.1f ms (replay %.0f ms), %.2f us/run, %.1f VU1 Mcycles/s\n",
                mismatches ? "FAIL" : "PASS", (unsigned long long)runs, records.size(), repeat,
                (unsigned long long)resumes, (unsigned long long)mismatches, ms, totalMs, ms * 1000.0 / double(runs),
                double(cycles) / (ms * 1000.0));
    (void)quiet;
    (void)codeLoads;
    if (repinPath)
    {
        FILE *f = std::fopen(repinPath, "wb");
        if (!f || std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size() || std::fclose(f) != 0)
        {
            std::fprintf(stderr, "%s: cannot write\n", repinPath);
            return 2;
        }
        std::printf("repinned %zu runs (%llu changed) -> %s\n", records.size(), (unsigned long long)mismatches, repinPath);
        return 0;
    }
    return mismatches ? 1 : 0;
}
