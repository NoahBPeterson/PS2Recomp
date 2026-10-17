// ps2x_vu1_pcsx2_diff: differential test of the runtime's VU1 interpreter against PCSX2 2.8.2's VU1
// interpreter (pcsx2_ref.cpp), driven by recorded VU1 run traces (runtime/ps2_vu1_trace.h).
// Every run's recorded inputs (micro memory, data memory, VU1 state, start pc, TOP/ITOP, D/T enable)
// go into PCSX2's VURegs, PCSX2 runs the program to its end, and the result is compared with the
// recorded result (which the runtime's interpreter produced): VF, VI (16 bits), ACC, Q, P, I, R,
// MAC/STATUS/CLIP flags, pc, data memory and the PATH1 (XGKICK) bytes. The runtime's interpreter
// (translated blocks off) replays alongside, as a check that it still reproduces the trace and to
// locate data / PATH1 differences byte by byte. Not compared: cycle counts and pipeline timing (see
// pcsx2_ref.cpp for what of PCSX2 is excluded).
//
// Field values print as name=PCSX2(want=recorded/runtime). A run the recording stopped on its cycle
// budget (resumed by the next record) is compared mid-program after the same number of pairs; the
// runtime still has results in flight there, so such runs are flagged and counted separately.
//
//   ps2x_vu1_pcsx2_diff <trace> [--max-runs N] [--examples N] [--step RUN] [--classify] [--context N]
// --step RUN    single-steps both interpreters in lockstep on that run (the runs before it replay
//               normally) and prints the first pair after which VF/ACC/VI differ (and the first
//               flag/Q/P difference, the first control-flow difference), with disassembly.
// --classify    does that for every mismatching run and prints histograms of the first diverging
//               instruction pair and field.
// --context N   with --step: also print the N pairs before the divergence.
// Exit status: 0 all runs match, 1 mismatch, 2 usage / bad trace.

#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu1_trace.h"

#include "vu1_opnames.inc"
#include "vu1_ref.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{
    // The run cycle cap the trace was recorded with (ps2_vu1_trace::runCycleCap).
    uint32_t g_runCycleCap = 65536u;
    using namespace ps2_vu1_trace;
    using vu1ref::ArchState;

    struct Record
    {
        Vu1TraceRecordHeader header{};
        const uint8_t *code = nullptr; // last included code at or before this record
        const uint8_t *data = nullptr;
        VU1State before{}, after{};
        uint64_t dataAfterHash = 0u, gifHash = 0u;
        uint32_t gifPackets = 0u, gifBytes = 0u;
    };

    // Same reader as ps2x_vu1_replay.
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
            if (!take(nullptr, kMemSize) || !take(&r.before, sizeof(VU1State)) || !take(&r.after, sizeof(VU1State)) ||
                !take(&r.dataAfterHash, 8) || !take(&r.gifHash, 8) || !take(&r.gifPackets, 4) || !take(&r.gifBytes, 4))
                return false;
            records.push_back(r);
        }
        return true;
    }

    template <typename T>
    uint32_t bits(T v)
    {
        static_assert(sizeof(T) == 4);
        uint32_t x;
        std::memcpy(&x, &v, 4);
        return x;
    }

    ArchState fromRuntime(const VU1State &s)
    {
        ArchState a{};
        std::memcpy(a.vf, s.vf, sizeof(a.vf));
        for (int r = 0; r < 16; ++r)
            a.vi[r] = static_cast<uint16_t>(s.vi[r]);
        std::memcpy(a.acc, s.acc, sizeof(a.acc));
        a.q = bits(s.q);
        a.p = bits(s.p);
        a.i = bits(s.i);
        a.r = s.r;
        a.mac = s.mac;
        a.status = s.status;
        a.clip = s.clip;
        a.pc = s.pc;
        return a;
    }

    // ---- disassembly (mnemonic tables from PCSX2's opcode layout, vu1_opnames.inc) ----
    std::string destMask(uint32_t w)
    {
        std::string d;
        for (int c = 0; c < 4; ++c)
            if (w & (1u << (24 - c)))
                d += "xyzw"[c];
        return d;
    }

    const char *upperName(uint32_t w)
    {
        const char *n = kUpper[w & 0x3f];
        if (n[0] != '@')
            return n;
        const uint32_t sub = (w >> 6) & 0x1f;
        switch (w & 3)
        {
        case 0: return kUpperFd00[sub];
        case 1: return kUpperFd01[sub];
        case 2: return kUpperFd10[sub];
        default: return kUpperFd11[sub];
        }
    }

    const char *lowerName(uint32_t w)
    {
        const char *n = kLower[w >> 25];
        if (n[0] != '@')
            return n;
        n = kLowerOp[w & 0x3f];
        if (n[0] != '@')
            return n;
        const uint32_t sub = (w >> 6) & 0x1f;
        switch (w & 3)
        {
        case 0: return kLowerT3_00[sub];
        case 1: return kLowerT3_01[sub];
        case 2: return kLowerT3_10[sub];
        default: return kLowerT3_11[sub];
        }
    }

    std::string disUpper(uint32_t w)
    {
        char buf[96];
        const uint32_t ft = (w >> 16) & 31, fs = (w >> 11) & 31, fd = (w >> 6) & 31;
        const std::string name = upperName(w);
        const bool special = (w & 0x3c) == 0x3c;
        if (name == "NOP")
            std::snprintf(buf, sizeof(buf), "NOP");
        else if (special)
            std::snprintf(buf, sizeof(buf), "%s.%s vf%u, vf%u", name.c_str(), destMask(w).c_str(), ft, fs);
        else
            std::snprintf(buf, sizeof(buf), "%s.%s vf%u, vf%u, vf%u", name.c_str(), destMask(w).c_str(), fd, fs, ft);
        std::string s = buf;
        if (w & 0x80000000u) s += " [I]";
        if (w & 0x40000000u) s += " [E]";
        if (w & 0x20000000u) s += " [M]";
        if (w & 0x10000000u) s += " [D]";
        if (w & 0x08000000u) s += " [T]";
        return s;
    }

    std::string disLower(uint32_t w)
    {
        char buf[96];
        const uint32_t it = (w >> 16) & 31, is = (w >> 11) & 31, id = (w >> 6) & 31;
        const std::string name = lowerName(w);
        if (w >> 25 == 0x40)
            std::snprintf(buf, sizeof(buf), "%s.%s t=%u s=%u d=%u", name.c_str(), destMask(w).c_str(), it, is, id);
        else
            std::snprintf(buf, sizeof(buf), "%s.%s t=%u s=%u imm11=0x%x", name.c_str(), destMask(w).c_str(), it, is, w & 0x7ff);
        return buf;
    }

    std::string disPair(const uint8_t *code, uint32_t pc, bool *iBit = nullptr)
    {
        uint32_t lower, upper;
        std::memcpy(&lower, code + (pc & 0x3ff8), 4);
        std::memcpy(&upper, code + (pc & 0x3ff8) + 4, 4);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "pc=0x%04x [%08x %08x] ", pc, upper, lower);
        if (iBit)
            *iBit = (upper & 0x80000000u) != 0;
        std::string s = buf + disUpper(upper) + " | ";
        if (upper & 0x80000000u)
        {
            std::snprintf(buf, sizeof(buf), "LOI 0x%08x", lower);
            s += buf;
        }
        else
            s += disLower(lower);
        return s;
    }

    // ---- field comparison ----
    enum Field
    {
        FVf, FVi, FAcc, FQ, FP, FI, FR, FMac, FStatus, FClip, FPc, FData, FGif, FCount
    };
    const char *const kFieldNames[FCount] = {"VF", "VI", "ACC", "Q", "P", "I", "R", "MAC", "STATUS", "CLIP", "pc", "data", "PATH1"};

    // Appends " name=got(want)" for differing fields; sets mask bits (1 << Field).
    uint32_t diffState(const ArchState &want, const ArchState &got, std::string *out, bool withFlags = true,
                       bool withPc = true)
    {
        uint32_t mask = 0;
        char buf[128];
        auto note = [&](const char *s)
        {
            if (out && out->size() < 600)
                *out += s;
        };
        for (int r = 0; r < 32; ++r)
            for (int c = 0; c < 4; ++c)
                if (want.vf[r][c] != got.vf[r][c])
                {
                    mask |= 1u << FVf;
                    std::snprintf(buf, sizeof(buf), " vf%02d.%c=%08x(want %08x)", r, "xyzw"[c], got.vf[r][c], want.vf[r][c]);
                    note(buf);
                }
        for (int r = 0; r < 16; ++r)
            if (want.vi[r] != got.vi[r])
            {
                mask |= 1u << FVi;
                std::snprintf(buf, sizeof(buf), " vi%02d=%04x(want %04x)", r, got.vi[r], want.vi[r]);
                note(buf);
            }
        for (int c = 0; c < 4; ++c)
            if (want.acc[c] != got.acc[c])
            {
                mask |= 1u << FAcc;
                std::snprintf(buf, sizeof(buf), " acc.%c=%08x(want %08x)", "xyzw"[c], got.acc[c], want.acc[c]);
                note(buf);
            }
        auto scalar = [&](Field f, const char *name, uint32_t w, uint32_t g)
        {
            if (w != g)
            {
                mask |= 1u << f;
                std::snprintf(buf, sizeof(buf), " %s=%x(want %x)", name, g, w);
                note(buf);
            }
        };
        if (withFlags)
        {
            scalar(FQ, "q", want.q, got.q);
            scalar(FP, "p", want.p, got.p);
            scalar(FMac, "mac", want.mac, got.mac);
            scalar(FStatus, "status", want.status, got.status);
            scalar(FClip, "clip", want.clip, got.clip);
        }
        scalar(FI, "i", want.i, got.i);
        scalar(FR, "r", want.r, got.r);
        if (withPc)
            scalar(FPc, "pc", want.pc, got.pc);
        return mask;
    }

    std::string maskNames(uint32_t mask)
    {
        std::string s;
        for (int f = 0; f < FCount; ++f)
            if (mask & (1u << f))
                s += (s.empty() ? "" : "+") + std::string(kFieldNames[f]);
        return s;
    }

    // ---- the runtime's interpreter with per-pair snapshots (VU1Interpreter::s_pcHook) ----
    VU1Interpreter *g_ours = nullptr;
    std::vector<ArchState> *g_ourSteps = nullptr;
    uint64_t g_ourPairs = 0;
    void ourHook(const uint8_t *, uint32_t pc)
    {
        ++g_ourPairs;
        if (g_ourSteps)
        {
            ArchState s = fromRuntime(g_ours->state());
            s.pc = pc;
            g_ourSteps->push_back(s);
        }
    }

    struct Divergence
    {
        bool found = false;
        size_t step = 0;            // first snapshot that differs
        uint32_t pairPc = 0;        // the pair executed just before it
        std::string what;           // fields
        uint32_t mask = 0;
        bool controlFlow = false;   // pc sequence differs
    };

    // ArchState as numbered elements: vf 0..127, vi 128..143, acc 144..147, i, r, q, p, mac, status, clip.
    constexpr int kElements = 155;
    constexpr size_t kPersist = 6; // pairs a difference must last to count (longer than any VI/flag latency)
    uint32_t &element(ArchState &s, int e)
    {
        if (e < 144)
            return s.vi[e - 128];
        if (e < 148)
            return s.acc[e - 144];
        uint32_t *const scalars[] = {&s.i, &s.r, &s.q, &s.p, &s.mac, &s.status, &s.clip};
        return *scalars[e - 148];
    }
    uint32_t element(const ArchState &s, int e) { return element(const_cast<ArchState &>(s), e); }

    // First step (snapshot index) at which the two step sequences differ. `primary`: VF/ACC/VI/I/R
    // and the pc sequence; otherwise MAC/STATUS/CLIP/Q/P only (their timing differs between the two
    // pipeline models, so they are reported separately).
    Divergence firstDivergence(const std::vector<ArchState> &ours, const std::vector<ArchState> &ref, bool primary)
    {
        Divergence d;
        const size_t n = std::min(ours.size(), ref.size());
        for (size_t k = 0; k < n; ++k)
        {
            const bool last = k + 1 == ours.size() || k + 1 == ref.size();
            if (primary && !last && ours[k].pc != ref[k].pc)
            {
                d.found = d.controlFlow = true;
                d.step = k;
                d.pairPc = k ? ours[k - 1].pc : ours[k].pc;
                char buf[96];
                std::snprintf(buf, sizeof(buf), " next pc 0x%x (PCSX2 0x%x)", ours[k].pc, ref[k].pc);
                d.what = buf;
                d.mask = 1u << FPc;
                return d;
            }
            // Only differences that persist count: the runtime models result latencies (a VI load
            // lands cycles after issue) where PCSX2 writes at issue, so a value can lag a few pairs.
            ArchState a = ours[k], b = ref[k];
            for (int e = 0; e < kElements; ++e)
            {
                if (element(a, e) == element(b, e))
                    continue;
                for (size_t j = k + 1; j <= k + kPersist && j < n; ++j)
                    if (element(ours[j], e) == element(ref[j], e))
                    {
                        element(b, e) = element(a, e);
                        break;
                    }
            }
            std::string what;
            uint32_t mask = diffState(a, b, &what, true, false);
            const uint32_t primaryMask = (1u << FVf) | (1u << FVi) | (1u << FAcc) | (1u << FI) | (1u << FR);
            mask &= primary ? primaryMask : ~primaryMask;
            if (mask)
            {
                if (!primary)
                { // rebuild the text with only the secondary fields
                    what.clear();
                    std::memcpy(b.vf, a.vf, sizeof(a.vf));
                    std::memcpy(b.vi, a.vi, sizeof(a.vi));
                    std::memcpy(b.acc, a.acc, sizeof(a.acc));
                    b.i = a.i;
                    b.r = a.r;
                    diffState(a, b, &what, true, false);
                }
                d.found = true;
                d.step = k;
                d.pairPc = k ? (k - 1 < ours.size() ? ours[k - 1].pc : 0) : ours[k].pc;
                d.what = what;
                d.mask = mask;
                return d;
            }
        }
        if (primary && ours.size() != ref.size())
        {
            d.found = d.controlFlow = true;
            d.step = n;
            d.pairPc = n ? ours[n - 1].pc : 0;
            d.what = " pair count " + std::to_string(ours.size() - 1) + " (PCSX2 " + std::to_string(ref.size() - 1) + ")";
            d.mask = 1u << FPc;
        }
        return d;
    }

    // The lowest differing byte offset of two buffers, or -1.
    long firstByteDiff(const uint8_t *a, size_t an, const uint8_t *b, size_t bn)
    {
        const size_t n = std::min(an, bn);
        for (size_t k = 0; k < n; ++k)
            if (a[k] != b[k])
                return static_cast<long>(k);
        return an == bn ? -1 : static_cast<long>(n);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: ps2x_vu1_pcsx2_diff <trace> [--max-runs N] [--examples N] [--step RUN] [--classify] [--context N]\n");
        return 2;
    }
    long maxRuns = -1, stepRun = -1;
    int examples = 10, context = 0;
    bool classify = false;
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--max-runs" && i + 1 < argc)
            maxRuns = std::atol(argv[++i]);
        else if (arg == "--examples" && i + 1 < argc)
            examples = std::atoi(argv[++i]);
        else if (arg == "--step" && i + 1 < argc)
            stepRun = std::atol(argv[++i]);
        else if (arg == "--context" && i + 1 < argc)
            context = std::atoi(argv[++i]);
        else if (arg == "--classify")
            classify = true;
        else
        {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }

    std::vector<uint8_t> bytes;
    std::vector<Record> records;
    if (!load(argv[1], bytes, records) || records.empty())
    {
        std::fprintf(stderr, "%s: not a readable VU1 trace (v%u)\n", argv[1], kVersion);
        return 2;
    }
    size_t runCount = records.size();
    if (maxRuns >= 0)
        runCount = std::min(runCount, static_cast<size_t>(maxRuns));
    if (stepRun >= 0)
        runCount = std::min(runCount, static_cast<size_t>(stepRun) + 1);

    // The runtime's interpreter, set up like ps2x_vu1_replay (translated blocks off: the hook needs
    // every pair, and the trace was pinned by the interpreter).
    PS2Memory memory;
    if (!memory.initialize())
    {
        std::fprintf(stderr, "PS2Memory init failed\n");
        return 2;
    }
    std::vector<uint8_t> ourGif;
    uint32_t ourGifPackets = 0u;
    GifArbiter arbiter([&](GifPathId, const uint8_t *data, uint32_t size)
                       {
                           ourGif.insert(ourGif.end(), data, data + size);
                           ++ourGifPackets;
                       });
    memory.setGifArbiter(&arbiter);
    GS gs;
    VU1Interpreter vu1(VU1Interpreter::Unit::VU1);
    vu1.setCompiledBlocksEnabled(false);
    vu1.reset();
    g_ours = &vu1;
    VU1Interpreter::s_pcHook = ourHook;
    uint8_t *const code = memory.getVU1Code();
    uint8_t *const data = memory.getVU1Data();

    vu1ref::init();

    uint64_t matches = 0, budgetMismatches = 0, ourMismatches = 0, budgetRuns = 0, refNoEnd = 0, resumes = 0, interrupts = 0;
    uint64_t fieldRuns[FCount] = {};
    int shown = 0;
    std::map<std::string, uint64_t> histPair, histField, histFlag;
    std::vector<ArchState> ourSteps, refSteps;
    const uint8_t *loadedCode = nullptr;
    for (size_t index = 0; index < runCount; ++index)
    {
        const Record &r = records[index];
        const bool resume = r.header.kind == static_cast<uint32_t>(RunKind::Resume);
        resumes += resume;
        const bool capture = classify || static_cast<long>(index) == stepRun;
        if (r.code != loadedCode)
        {
            std::memcpy(code, r.code, kMemSize);
            memory.markVU1CodeModified();
            vu1ref::loadCode(r.code);
            loadedCode = r.code;
        }

        // Ours.
        std::memcpy(data, r.data, kMemSize);
        ourGif.clear();
        ourGifPackets = 0u;
        ourSteps.clear();
        g_ourSteps = capture ? &ourSteps : nullptr;
        g_ourPairs = 0;
        vu1.restoreState(r.before);
        if (!resume)
            vu1.execute(code, kMemSize, data, kMemSize, gs, &memory, r.header.startPc, r.header.top, r.header.itop, g_runCycleCap);
        else
            vu1.resume(code, kMemSize, data, kMemSize, gs, &memory, r.header.top, r.header.itop, g_runCycleCap);
        g_ourSteps = nullptr;
        const VU1State ourAfter = vu1.state();
        if (capture)
            ourSteps.push_back(fromRuntime(ourAfter));
        const uint64_t ourPairs = g_ourPairs;
        const bool ourBudgetStop = ourAfter.cycles - r.before.cycles >= g_runCycleCap && !ourAfter.stoppedByD && !ourAfter.stoppedByT;
        budgetRuns += ourBudgetStop;
        {
            std::string d;
            const uint32_t m = diffState(fromRuntime(r.after), fromRuntime(ourAfter), &d);
            const bool ok = !m && fnv1a(data, kMemSize) == r.dataAfterHash && fnv1a(ourGif.data(), ourGif.size()) == r.gifHash &&
                            ourGif.size() == r.gifBytes && ourGifPackets == r.gifPackets;
            if (!ok)
            {
                if (ourMismatches < 3)
                    std::printf("NOTE run %zu: the runtime interpreter no longer reproduces the trace:%s\n", index, d.c_str());
                ++ourMismatches;
            }
        }

        // PCSX2.
        vu1ref::RunSetup setup;
        setup.resume = resume;
        setup.startPc = r.header.startPc;
        setup.top = r.header.top;
        setup.itop = r.header.itop;
        setup.dBit = r.header.dBitEnabled != 0;
        setup.tBit = r.header.tBitEnabled != 0;
        setup.cycle = r.before.cycles;
        setup.regs = fromRuntime(r.before);
        vu1ref::loadData(r.data);
        refSteps.clear();
        // A run the recording stopped on its cycle budget: the same number of pairs.
        const uint64_t limit = ourBudgetStop ? ourPairs : 4000000u;
        const vu1ref::RunResult rr = vu1ref::run(setup, limit, capture ? &refSteps : nullptr);
        interrupts += rr.interrupted;
        if (!rr.ended && !ourBudgetStop)
            ++refNoEnd;
        const ArchState refAfter = vu1ref::state();

        std::string diff;
        uint32_t mask = diffState(fromRuntime(r.after), refAfter, &diff);
        const uint8_t *refData = vu1ref::data();
        const std::vector<uint8_t> &refGif = vu1ref::path1();
        if (fnv1a(refData, kMemSize) != r.dataAfterHash)
        {
            mask |= 1u << FData;
            const long at = firstByteDiff(refData, kMemSize, data, kMemSize);
            char buf[96];
            if (at >= 0)
            {
                uint32_t g, w;
                std::memcpy(&g, refData + (at & ~3l), 4);
                std::memcpy(&w, data + (at & ~3l), 4);
                std::snprintf(buf, sizeof(buf), " data@0x%04lx=%08x(want %08x)", at & ~3l, g, w);
            }
            else
                std::snprintf(buf, sizeof(buf), " data(hash)");
            diff += buf;
        }
        if (fnv1a(refGif.data(), refGif.size()) != r.gifHash || refGif.size() != r.gifBytes ||
            vu1ref::path1Packets() != r.gifPackets)
        {
            mask |= 1u << FGif;
            const long at = firstByteDiff(refGif.data(), refGif.size(), ourGif.data(), ourGif.size());
            char buf[128];
            std::snprintf(buf, sizeof(buf), " PATH1(%u pkts/%zu B, want %u/%u; first diff @%ld)", vu1ref::path1Packets(),
                          refGif.size(), r.gifPackets, r.gifBytes, at);
            diff += buf;
        }
        if (!rr.ended && !ourBudgetStop)
            diff += " (PCSX2 did not end)";

        if (ourBudgetStop)
            diff += " (stopped on the cycle budget: pipelines in flight, compared mid-program)";
        if (!mask)
            ++matches;
        else
        {
            budgetMismatches += ourBudgetStop;
            for (int f = 0; f < FCount; ++f)
                if (mask & (1u << f))
                    ++fieldRuns[f];
            if (shown < examples)
            {
                std::printf("MISMATCH run %zu (%s pc=0x%x, %llu pairs): [%s]%s\n", index, resume ? "resume" : "execute",
                            r.header.startPc, (unsigned long long)ourPairs, maskNames(mask).c_str(), diff.c_str());
                ++shown;
            }
        }

        if (capture && (mask || static_cast<long>(index) == stepRun))
        {
            const Divergence d = firstDivergence(ourSteps, refSteps, true);
            const Divergence f = firstDivergence(ourSteps, refSteps, false);
            if (static_cast<long>(index) == stepRun)
            {
                std::printf("step run %zu: ours %zu pairs, PCSX2 %zu pairs\n", index, ourSteps.size() - 1, refSteps.size() - 1);
                if (d.found)
                {
                    for (int c = context; c > 0; --c)
                        if (d.step >= static_cast<size_t>(c) + 1)
                            std::printf("  step %zu: %s\n", d.step - c - 1, disPair(r.code, ourSteps[d.step - c - 1].pc).c_str());
                    std::printf("first VF/VI/ACC/I/R %s after step %zu: %s\n  differs (field=PCSX2(want=ours)):%s\n",
                                d.controlFlow ? "control-flow divergence" : "divergence", d.step ? d.step - 1 : 0,
                                disPair(r.code, d.pairPc).c_str(), d.what.c_str());
                    for (size_t k = d.step >= 3 ? d.step - 3 : 0; k <= d.step && k < ourSteps.size() && k < refSteps.size(); ++k)
                    {
                        std::printf("  VI before step %zu (pc 0x%04x / PCSX2 0x%04x):", k, ourSteps[k].pc, refSteps[k].pc);
                        for (int v = 1; v < 16; ++v)
                            if (ourSteps[k].vi[v] != refSteps[k].vi[v])
                                std::printf(" vi%d=%04x/PCSX2 %04x", v, ourSteps[k].vi[v], refSteps[k].vi[v]);
                            else
                                std::printf(" vi%d=%04x", v, ourSteps[k].vi[v]);
                        std::printf("\n");
                    }
                }
                else
                    std::printf("no VF/VI/ACC/I/R or control-flow divergence between pairs\n");
                if (f.found)
                    std::printf("first MAC/STATUS/CLIP/Q/P difference after step %zu: %s\n  differs (field=PCSX2(want=ours)):%s\n",
                                f.step ? f.step - 1 : 0, disPair(r.code, f.pairPc).c_str(), f.what.c_str());
                else
                    std::printf("no MAC/STATUS/CLIP/Q/P difference between pairs\n");
                if (f.found)
                    for (size_t k = f.step >= 6 ? f.step - 6 : 0; k < f.step + 10 && k < ourSteps.size() && k < refSteps.size(); ++k)
                        std::printf("  before step %zu %s: mac %04x/PCSX2 %04x status %03x/PCSX2 %03x acc %08x %08x %08x %08x\n", k,
                                    disPair(r.code, ourSteps[k].pc).c_str(), ourSteps[k].mac, refSteps[k].mac, ourSteps[k].status,
                                    refSteps[k].status, ourSteps[k].acc[0], ourSteps[k].acc[1], ourSteps[k].acc[2], ourSteps[k].acc[3]);
            }
            if (classify && mask)
            {
                auto key = [&](const Divergence &dv)
                {
                    uint32_t lower, upper;
                    std::memcpy(&lower, r.code + (dv.pairPc & 0x3ff8), 4);
                    std::memcpy(&upper, r.code + (dv.pairPc & 0x3ff8) + 4, 4);
                    char at[16];
                    std::snprintf(at, sizeof(at), "@0x%04x ", dv.pairPc);
                    std::string k = at + std::string(upperName(upper)) + " | " + ((upper & 0x80000000u) ? "LOI" : lowerName(lower));
                    return k;
                };
                if (d.found)
                {
                    ++histPair[(d.controlFlow ? "[control flow] " : "") + key(d)];
                    ++histField[maskNames(d.mask)];
                }
                else
                    ++histPair["(no step divergence: final/flush only) " + maskNames(mask)];
                if (f.found)
                    ++histFlag[key(f) + "  -> " + maskNames(f.mask)];
            }
        }
    }

    if (classify)
    {
        auto dump = [](const char *title, const std::map<std::string, uint64_t> &h)
        {
            std::vector<std::pair<uint64_t, std::string>> v;
            for (const auto &[k, n] : h)
                v.push_back({n, k});
            std::sort(v.rbegin(), v.rend());
            std::printf("%s:\n", title);
            for (size_t k = 0; k < v.size() && k < 25; ++k)
                std::printf("  %6llu  %s\n", (unsigned long long)v[k].first, v[k].second.c_str());
        };
        dump("first diverging pair (VF/VI/ACC/I/R/control flow), mismatching runs", histPair);
        dump("fields at that divergence", histField);
        dump("first MAC/STATUS/CLIP/Q/P divergence (pair -> fields)", histFlag);
    }
    std::printf("fields differing (runs):");
    for (int f = 0; f < FCount; ++f)
        if (fieldRuns[f])
            std::printf(" %s=%llu", kFieldNames[f], (unsigned long long)fieldRuns[f]);
    std::printf("\n");
    const uint64_t mismatches = runCount - matches;
    // A run the recording stopped on its cycle budget is compared mid-program, where results still in
    // flight in our pipeline differ from PCSX2's issue-time writes: reported, not a failure.
    const bool fail = mismatches > budgetMismatches || ourMismatches != 0;
    std::printf("%s: %zu runs (%llu resumes, %llu stopped on the cycle budget, %llu D/T interrupts), %llu match PCSX2, "
                "%llu mismatch (%llu of them stopped on the budget); PCSX2 did not end %llu; runtime interpreter vs trace: %llu mismatches\n",
                fail ? "FAIL" : "PASS", runCount, (unsigned long long)resumes, (unsigned long long)budgetRuns,
                (unsigned long long)interrupts, (unsigned long long)matches, (unsigned long long)mismatches, (unsigned long long)budgetMismatches,
                (unsigned long long)refNoEnd, (unsigned long long)ourMismatches);
    return fail ? 1 : 0;
}
