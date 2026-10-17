// ps2x_vu1_rec_gen: VU1 microprogram recompiler (T-0121). Generates C++ for the VU1 code in recorded
// run traces (runtime/ps2_vu1_trace.h):
//   1. replays the traces through the interpreter (translation off) to find the code that runs and
//      where control enters it (program starts, branch/jump destinations);
//   2. cuts it into blocks (ending at branches + delay slots, E-bit + delay slot, XGKICK);
//   3. emits, per block, a VU1Interpreter::runCompiledBlock specialization that does what the
//      interpreter would, pair by pair, with everything known at translation time folded away:
//      decode; the scoreboard (each pair issues at max(previous + 1, the ready cycles of what it
//      reads), with only the operands that can still stall checked); the pipeline commits, done
//      lazily where something depends on them (recCommitTo); stores and VF/VI/ACC writes landing
//      at issue, the scoreboard state stored at the exits; FMAC flag updates committed by the block
//      itself when nothing in it reads MAC/STATUS. A block declines (the interpreter runs the pair)
//      only for state it does not handle: a branch or halt pending, interpreter writes queued, too
//      little cycle budget left.
// The vu1 conformance tier replays the same traces through the result and checks every run.
//
//   ps2x_vu1_rec_gen <out dir> <trace.vu1trace>...   (tools/perf/vu1_rec_regen.sh)
//   ps2x_vu1_rec_gen --selftest-fmac                 SIMD FMAC code against the scalar code
//   ps2x_vu1_rec_gen --check-fmac-ref <file>         FMAC values against ps2-float (tools/vu1/fmac_ref)
//   ps2x_vu1_rec_gen --fuzz-blocks <cases>           every translated block against the interpreter
// Output is disc-derived (game code): not for git.

#include "../../src/lib/vu/ps2_vu1_issue.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1_trace.h"

#include <algorithm>
#include <iterator>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ps2_vu1_trace;

struct Vu1RecGenerator
{
    // computeFmacOp (SIMD) against computeFmacOpScalar on random and edge-case operands.
    template <VU1Interpreter::FmacKind K, VU1Interpreter::FmacOperand O>
    static uint64_t fuzzFmac(VU1Interpreter &vu, uint64_t &seed, int rounds)
    {
        static const uint32_t kEdges[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x00800000u, 0x80800000u,
                                          0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x3F800000u,
                                          0xBF800000u, 0x7F000000u, 0x00FFFFFFu, 0x01000000u, 0x0D800000u, 0x3F7FFFFFu};
        auto next = [&]()
        {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            return seed;
        };
        auto value = [&]() -> float
        {
            const uint64_t pick = next();
            uint32_t bits;
            switch (pick % 4u)
            {
            case 0: bits = kEdges[(pick >> 8) % (sizeof(kEdges) / 4u)]; break;
            case 1: bits = static_cast<uint32_t>(pick >> 16); break;                                    // anything
            case 2: bits = (static_cast<uint32_t>(pick >> 16) & 0x807FFFFFu) | (((pick >> 48) % 40u + 107u) << 23); break; // near 1
            default: bits = (static_cast<uint32_t>(pick >> 16) & 0x807FFFFFu) | ((((pick >> 48) % 12u) + ((pick >> 60) & 1u ? 0u : 242u)) << 23); break; // near limits
            }
            float f;
            std::memcpy(&f, &bits, 4);
            return f;
        };
        uint64_t bad = 0;
        for (int round = 0; round < rounds; ++round)
        {
            for (int reg = 1; reg < 4; ++reg)
                for (int c = 0; c < 4; ++c)
                    vu.m_state.vf[reg][c] = value();
            for (int c = 0; c < 4; ++c)
                vu.m_state.acc[c] = value();
            vu.m_state.q = value();
            vu.m_state.i = value();
            const uint8_t dest = static_cast<uint8_t>(next() & 0xFu);
            const uint32_t instr = (static_cast<uint32_t>(dest) << 21) | (2u << 16) | (1u << 11) | (3u << 6) | static_cast<uint32_t>(next() & 3u);
            float a[4]{}, b[4]{};
            uint8_t laneFlags[4]{};
            uint32_t sticky = 0u;
            vu.computeFmacOpScalar<K, O>(instr, dest, a, laneFlags, sticky);
            const auto simd = vu.computeFmacOp<K, O>(instr, dest, b);
            uint32_t mac = 0u, status = 0u;
            for (uint32_t c = 0; c < 4u; ++c)
            {
                const uint8_t lane = static_cast<uint8_t>(1u << (3u - c));
                if ((dest & lane) == 0u)
                    continue;
                const uint32_t f = laneFlags[c];
                mac |= ((f & 1u) ? lane : 0u) | ((f & 2u) ? lane << 4 : 0u) | ((f & 4u) ? lane << 8 : 0u) | ((f & 8u) ? lane << 12 : 0u);
                status |= f;
                if (std::memcmp(&a[c], &b[c], 4) != 0)
                    ++bad;
            }
            if (mac != simd.mac || status != simd.status || (sticky & 0xFu) != simd.productSticky)
                ++bad;
        }
        return bad;
    }

    static int selfTestFmac()
    {
        using K = VU1Interpreter::FmacKind;
        using O = VU1Interpreter::FmacOperand;
        auto vu = std::make_unique<VU1Interpreter>(VU1Interpreter::Unit::VU1);
        vu->reset();
        if (std::fesetround(FE_TOWARDZERO) != 0) // as VU1Interpreter::run() executes microcode
        {
            std::printf("fmac self-test: cannot round toward zero\n");
            return 1;
        }
        uint64_t seed = 0x9E3779B97F4A7C15ull, bad = 0;
        const int rounds = 200000;
#define PS2X_FUZZ(k, o)                                                                    \
    {                                                                                      \
        const uint64_t b = fuzzFmac<K::k, O::o>(*vu, seed, rounds);                        \
        std::printf("fmac %-6s %-9s %s (%llu bad)\n", #k, #o, b ? "FAIL" : "ok", (unsigned long long)b); \
        bad += b;                                                                          \
    }
#define PS2X_FUZZ_ALL(k) PS2X_FUZZ(k, Broadcast) PS2X_FUZZ(k, Q) PS2X_FUZZ(k, I) PS2X_FUZZ(k, Vector)
        PS2X_FUZZ_ALL(Add)
        PS2X_FUZZ_ALL(Sub)
        PS2X_FUZZ_ALL(Madd)
        PS2X_FUZZ_ALL(Msub)
        PS2X_FUZZ_ALL(Mul)
        PS2X_FUZZ(Opmsub, Vector)
        PS2X_FUZZ(Opmula, Vector)
#undef PS2X_FUZZ_ALL
#undef PS2X_FUZZ
        std::printf("fmac self-test: %s\n", bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    }

    // FMAC results against reference results from ps2-float's VU model (PCSX2's arithmetic; file
    // from tools/vu1/fmac_ref): ADD SUB MUL MADD MSUB, through the generic (execFmac), scalar and
    // SIMD code. Values only (the model has no flags).
    static int checkFmacRef(const char *path)
    {
        using K = VU1Interpreter::FmacKind;
        using O = VU1Interpreter::FmacOperand;
        std::ifstream in(path, std::ios::binary);
        std::vector<uint32_t> words;
        if (in)
        {
            in.seekg(0, std::ios::end);
            words.resize(static_cast<size_t>(in.tellg()) / 4u);
            in.seekg(0);
            in.read(reinterpret_cast<char *>(words.data()), static_cast<std::streamsize>(words.size() * 4u));
        }
        const size_t cases = words.size() / 8u;
        if (cases < 4u)
        {
            std::printf("fmac ref: %s: no cases\n", path);
            return 1;
        }
        auto vu = std::make_unique<VU1Interpreter>(VU1Interpreter::Unit::VU1);
        vu->reset();
        if (std::fesetround(FE_TOWARDZERO) != 0) // as VU1Interpreter::run() executes microcode
        {
            std::printf("fmac ref: cannot round toward zero\n");
            return 1;
        }
        static const char *kNames[5] = {"ADD", "SUB", "MUL", "MADD", "MSUB"};
        static const K kKinds[5] = {K::Add, K::Sub, K::Mul, K::Madd, K::Msub};
        uint64_t bad[5][3]{}; // [op][generic, scalar, SIMD]
        uint64_t fusedWouldDiffer = 0, shown = 0;
        constexpr uint32_t instr = (0xFu << 21) | (2u << 16) | (1u << 11) | (3u << 6); // .xyzw vf3, vf1, vf2
        for (size_t base = 0; base + 4u <= cases; base += 4u)
        {
            for (uint32_t c = 0; c < 4u; ++c)
            {
                const uint32_t *w = &words[(base + c) * 8u];
                std::memcpy(&vu->m_state.acc[c], &w[0], 4);
                std::memcpy(&vu->m_state.vf[1][c], &w[1], 4);
                std::memcpy(&vu->m_state.vf[2][c], &w[2], 4);
            }
            float vs[4], vt[4], acc[4];
            for (uint32_t c = 0; c < 4u; ++c)
            {
                vs[c] = vu->normalizeOperand(vu->m_state.vf[1][c]);
                vt[c] = vu->normalizeOperand(vu->m_state.vf[2][c]);
                acc[c] = vu->normalizeOperand(vu->m_state.acc[c]);
                const float fused = std::fma(vs[c], vt[c], acc[c]); // what one rounding would give
                uint32_t fusedBits, ref = words[(base + c) * 8u + 6u];
                std::memcpy(&fusedBits, &fused, 4);
                if ((fusedBits & 0x7F800000u) != 0u && (fusedBits & 0x7F800000u) != 0x7F800000u && fusedBits != ref)
                    ++fusedWouldDiffer;
            }
            for (int op = 0; op < 5; ++op)
            {
                vu->reset(); // (execFmac queues a flag update per instruction; cycles never advance here)
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    const uint32_t *w = &words[(base + c) * 8u];
                    std::memcpy(&vu->m_state.acc[c], &w[0], 4);
                    std::memcpy(&vu->m_state.vf[1][c], &w[1], 4);
                    std::memcpy(&vu->m_state.vf[2][c], &w[2], 4);
                }
                float out[3][4]{};
                vu->execFmac(kKinds[op], O::Vector, 0u, false, out[0], 0xFu, vs, vt, acc, 0.0f, 0.0f);
                uint8_t laneFlags[4]{};
                uint32_t sticky = 0u;
                switch (op)
                {
                case 0: vu->computeFmacOpScalar<K::Add, O::Vector>(instr, 0xFu, out[1], laneFlags, sticky); vu->computeFmacOp<K::Add, O::Vector>(instr, 0xFu, out[2]); break;
                case 1: vu->computeFmacOpScalar<K::Sub, O::Vector>(instr, 0xFu, out[1], laneFlags, sticky); vu->computeFmacOp<K::Sub, O::Vector>(instr, 0xFu, out[2]); break;
                case 2: vu->computeFmacOpScalar<K::Mul, O::Vector>(instr, 0xFu, out[1], laneFlags, sticky); vu->computeFmacOp<K::Mul, O::Vector>(instr, 0xFu, out[2]); break;
                case 3: vu->computeFmacOpScalar<K::Madd, O::Vector>(instr, 0xFu, out[1], laneFlags, sticky); vu->computeFmacOp<K::Madd, O::Vector>(instr, 0xFu, out[2]); break;
                default: vu->computeFmacOpScalar<K::Msub, O::Vector>(instr, 0xFu, out[1], laneFlags, sticky); vu->computeFmacOp<K::Msub, O::Vector>(instr, 0xFu, out[2]); break;
                }
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    const uint32_t *w = &words[(base + c) * 8u];
                    for (int path = 0; path < 3; ++path)
                    {
                        uint32_t got;
                        std::memcpy(&got, &out[path][c], 4);
                        if (got == w[3 + op])
                            continue;
                        ++bad[op][path];
                        if (shown++ < 8u)
                            std::printf("  %s path %d: acc %08x s %08x r %08x: got %08x, model %08x\n", kNames[op], path,
                                        w[0], w[1], w[2], got, w[3 + op]);
                    }
                }
            }
        }
        uint64_t total = 0;
        for (int op = 0; op < 5; ++op)
        {
            std::printf("fmac ref %-4s generic %llu scalar %llu simd %llu bad of %zu\n", kNames[op],
                        (unsigned long long)bad[op][0], (unsigned long long)bad[op][1], (unsigned long long)bad[op][2],
                        cases / 4u * 4u);
            total += bad[op][0] + bad[op][1] + bad[op][2];
        }
        std::printf("fmac ref: %s, %zu cases x 5 ops x 3 paths bit-exact vs ps2-float (a fused MADD would differ in %llu)\n",
                    total ? "FAIL" : "PASS", cases / 4u * 4u, (unsigned long long)fusedWouldDiffer);
        return total ? 1 : 0;
    }

    // Every registered translated block against the interpreter: per case a random VU1 state and
    // data memory, micro memory holding only the block's code, one run from the
    // block's pc with translated blocks and one interpreted, compared in full (VU1 state, data
    // memory, PATH1 packets). Micro memory outside the block holds NOP[E] pairs. Covers blocks no trace runs (static coverage from the ELF). A case in
    // which the block declined to run does not count; every block must run in at least one.
    static bool sameState(const VU1State &a, const VU1State &b)
    {
        return std::memcmp(a.vf, b.vf, sizeof(a.vf)) == 0 && std::memcmp(a.vi, b.vi, sizeof(a.vi)) == 0 &&
               std::memcmp(a.acc, b.acc, sizeof(a.acc)) == 0 && std::memcmp(&a.q, &b.q, 4) == 0 &&
               std::memcmp(&a.p, &b.p, 4) == 0 && std::memcmp(&a.i, &b.i, 4) == 0 && a.r == b.r && a.pc == b.pc &&
               a.mac == b.mac && a.clip == b.clip && a.status == b.status && a.cycles == b.cycles && a.ebit == b.ebit &&
               a.haltAfterDelaySlot == b.haltAfterDelaySlot && a.stoppedByD == b.stoppedByD && a.stoppedByT == b.stoppedByT &&
               a.branchPending == b.branchPending && a.branchTarget == b.branchTarget && a.branchDelay == b.branchDelay;
    }

    static int fuzzBlocks(int cases)
    {
        const auto &list = VU1Interpreter::compiledBlockList();
        if (list.empty())
        {
            std::printf("fuzz blocks: no translated blocks registered\n");
            return 1;
        }
        struct Side
        {
            PS2Memory memory;
            std::unique_ptr<GifArbiter> arbiter;
            GS gs;
            VU1Interpreter vu{VU1Interpreter::Unit::VU1};
            uint64_t gifHash = 0u;
            uint32_t gifPackets = 0u;
        };
        auto a = std::make_unique<Side>(), b = std::make_unique<Side>();
        for (Side *side : {a.get(), b.get()})
        {
            side->memory.initialize();
            side->arbiter = std::make_unique<GifArbiter>([side](GifPathId, const uint8_t *data, uint32_t size)
                                                         {
                                                             side->gifHash = fnv1a(data, size, side->gifHash);
                                                             ++side->gifPackets;
                                                         });
            side->memory.setGifArbiter(side->arbiter.get());
        }
        b->vu.setCompiledBlocksEnabled(false);
        uint64_t seed = 0xD1B54A32D192ED03ull;
        auto next = [&]()
        {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            return seed;
        };
        auto randomFloat = [&]() -> float
        {
            const uint64_t pick = next();
            uint32_t bits = static_cast<uint32_t>(pick >> 16);
            switch (pick % 8u)
            {
            case 0: bits = 0u; break;
            case 1: bits = (bits & 0x807FFFFFu) | (((pick >> 48) % 8u + 123u) << 23); break; // near 1
            case 2: bits = (bits & 0x807FFFFFu) | (((pick >> 48) % 60u + 100u) << 23); break;
            case 3: break; // anything, specials included
            default: bits = (bits & 0x807FFFFFu) | (((pick >> 48) % 30u + 112u) << 23); break;
            }
            float f;
            std::memcpy(&f, &bits, 4);
            return f;
        };
        // With blocks on: the block declined at entry if its first pair is interpreted at the entry
        // cycle; interpreted later (a loop back to its start) it declined a re-entry.
        static uint32_t s_watchPc = UINT32_MAX;
        static const VU1Interpreter *s_watchVu = nullptr;
        static bool s_entryDeclined = false;
        static uint64_t s_reentryDeclines = 0;
        VU1Interpreter::s_pcHook = [](const uint8_t *, uint32_t pc)
        {
            if (!s_watchVu || pc != s_watchPc)
                return;
            if (s_watchVu->m_cycle == 0u)
                s_entryDeclined = true;
            else
            {
                ++s_reentryDeclines;
                if (std::getenv("PS2X_FUZZ_WHY"))
                {
                    const VU1Interpreter &v = *s_watchVu;
                    std::fprintf(stderr, "reentry-decline cycle=%llu pc=%04x branchPending=%d halt=%d ebit=%d store=%x vf=%x vi=%x acc=%x\n", (unsigned long long)v.m_cycle, pc,
                                 v.m_state.branchPending, v.m_state.haltAfterDelaySlot, v.m_state.ebit, unsigned(v.m_storeMask),
                                 unsigned(v.m_vfWriteMask), unsigned(v.m_viWriteMask), unsigned(v.m_accWriteMask));
                }
            }
        };
        std::vector<uint8_t> data(kMemSize);
        uint64_t mismatches = 0, ran = 0, neverRan = 0, shown = 0, errorStops = 0;
        const uint64_t showLimit = std::getenv("PS2X_FUZZ_SHOW") ? std::strtoull(std::getenv("PS2X_FUZZ_SHOW"), nullptr, 10) : 10u;
        const char *onlyEnv = std::getenv("PS2X_FUZZ_ONLY"); // one block pc (hex)
        const char *caseEnv = std::getenv("PS2X_FUZZ_CASE");
        for (const auto &info : list)
        {
            if (onlyEnv)
            {
                // "pc" or "from-to" (hex)
                char *dash = nullptr;
                const unsigned long from = std::strtoul(onlyEnv, &dash, 16);
                const unsigned long to = (dash && *dash == '-') ? std::strtoul(dash + 1, nullptr, 16) : from;
                if (info.pc < from || info.pc > to)
                    continue;
            }
            bool blockRan = false;
            for (int c = 0; c < cases; ++c)
            {
                const bool skipCase = caseEnv && c != std::atoi(caseEnv);
                if (skipCase)
                    continue;
                seed = fnv1a(info.code, info.size, 0x9E3779B97F4A7C15ull ^ (uint64_t(info.pc) << 32) ^ uint64_t(c)) | 1u;
                VU1State st{};
                for (auto &reg : st.vf)
                    for (float &lane : reg)
                        lane = randomFloat();
                st.vf[0][0] = st.vf[0][1] = st.vf[0][2] = 0.0f;
                st.vf[0][3] = 1.0f;
                for (int r = 1; r < 16; ++r)
                    st.vi[r] = static_cast<int32_t>((next() % 4u) ? next() % 0x400u : next() & 0xFFFFu);
                for (float &lane : st.acc)
                    lane = randomFloat();
                st.q = randomFloat();
                st.p = randomFloat();
                st.i = randomFloat();
                st.r = 0x3F800000u | static_cast<uint32_t>(next() & 0x7FFFFFu);
                st.mac = static_cast<uint32_t>(next() & 0xFFFFu);
                st.status = static_cast<uint32_t>(next() & 0xFFFu);
                st.clip = static_cast<uint32_t>(next() & 0xFFFFFFu);
                for (size_t off = 0; off < data.size(); off += 8u)
                {
                    const uint64_t w = next();
                    std::memcpy(data.data() + off, &w, 8);
                }
                for (size_t off = 0; off < data.size(); off += 4u)
                    if (next() % 2u)
                    {
                        const float f = randomFloat();
                        std::memcpy(data.data() + off, &f, 4);
                    }
                // Any qword XGKICK may start at reads as a short GIF packet (NLOOP <= 7, EOP): random
                // tags overflow the transfer buffer, which stops the program as an error with
                // writes in flight. Only the low mantissa bits of each x lane change.
                for (size_t off = 0; off < data.size(); off += 16u)
                {
                    uint32_t x;
                    std::memcpy(&x, data.data() + off, 4);
                    x = (x & ~0xFFFFu) | 0x8000u | static_cast<uint32_t>(next() & 7u);
                    std::memcpy(data.data() + off, &x, 4);
                }
                const uint32_t top = static_cast<uint32_t>(next() % 0x400u), itop = static_cast<uint32_t>(next() % 0x400u);

                VU1State out[2];
                bool errorStop = false;
                for (int k = 0; k < 2; ++k)
                {
                    Side &side = k == 0 ? *a : *b;
                    // Outside the block: NOP[E] pairs, so wherever control leaves it the program ends
                    // normally (pipelines drained) instead of stopping on a reserved instruction.
                    for (uint32_t off = 0; off < kMemSize; off += 8u)
                    {
                        const uint32_t lower = 0x8000033Cu, upper = 0x400002FFu;
                        std::memcpy(side.memory.getVU1Code() + off, &lower, 4);
                        std::memcpy(side.memory.getVU1Code() + off + 4u, &upper, 4);
                    }
                    std::memcpy(side.memory.getVU1Code() + info.pc, info.code, info.size);
                    side.memory.markVU1CodeModified();
                    std::memcpy(side.memory.getVU1Data(), data.data(), kMemSize);
                    side.gifHash = 0xcbf29ce484222325ull;
                    side.gifPackets = 0u;
                    side.vu.reset();
                    side.vu.restoreState(st);
                    s_watchVu = k == 0 ? &side.vu : nullptr;
                    s_watchPc = info.pc;
                    s_entryDeclined = false;
                    side.vu.execute(side.memory.getVU1Code(), kMemSize, side.memory.getVU1Data(), kMemSize, side.gs, &side.memory,
                                    info.pc, top, itop, 65536);
                    if (k == 0 && !s_entryDeclined)
                        blockRan = true, ++ran;
                    out[k] = side.vu.state();
                    errorStop = errorStop || side.vu.m_stopRequested;
                }
                if (errorStop)
                {
                    // Stopped on an error (a reserved instruction, an XGKICK packet overflowing its
                    // buffer after a store wrote over its tag), writes possibly in flight: the
                    // states are not comparable. Counted, not judged.
                    ++errorStops;
                    continue;
                }
                const bool ok = sameState(out[0], out[1]) &&
                                std::memcmp(a->memory.getVU1Data(), b->memory.getVU1Data(), kMemSize) == 0 &&
                                a->gifHash == b->gifHash && a->gifPackets == b->gifPackets;
                if (!ok)
                {
                    ++mismatches;
                    if (shown++ < showLimit)
                    {
                        std::string what;
                        const VU1State &x = out[0], &y = out[1];
                        for (int r = 0; r < 32; ++r)
                            for (int l = 0; l < 4; ++l)
                                if (std::memcmp(&x.vf[r][l], &y.vf[r][l], 4) != 0)
                                {
                                    uint32_t xb, yb;
                                    std::memcpy(&xb, &x.vf[r][l], 4);
                                    std::memcpy(&yb, &y.vf[r][l], 4);
                                    what += " vf" + std::to_string(r) + "xyzw"[l] + "=" + hex(xb, 8) + "/" + hex(yb, 8);
                                }
                        for (int r = 0; r < 16; ++r)
                            if (x.vi[r] != y.vi[r])
                                what += " vi" + std::to_string(r);
                        if (std::memcmp(x.acc, y.acc, sizeof(x.acc)) != 0) what += " acc";
                        if (std::memcmp(&x.q, &y.q, 4) != 0) what += " q";
                        if (std::memcmp(&x.p, &y.p, 4) != 0) what += " p";
                        if (std::memcmp(&x.i, &y.i, 4) != 0) what += " i";
                        if (x.r != y.r) what += " r";
                        if (x.mac != y.mac) what += " mac";
                        if (x.status != y.status) what += " status(" + hex(x.status) + "/" + hex(y.status) + ")";
                        if (x.clip != y.clip) what += " clip";
                        if (x.ebit != y.ebit || x.branchPending != y.branchPending || x.haltAfterDelaySlot != y.haltAfterDelaySlot)
                            what += " control";
                        if (std::memcmp(a->memory.getVU1Data(), b->memory.getVU1Data(), kMemSize) != 0) what += " data";
                        if (a->gifHash != b->gifHash) what += " gif";
                        std::printf("  MISMATCH block pc 0x%04x (%u bytes) case %d: pc %x/%x cycles %llu/%llu:%s\n", info.pc,
                                    info.size, c, x.pc, y.pc, (unsigned long long)x.cycles, (unsigned long long)y.cycles,
                                    what.c_str());
                    }
                }
            }
            if (!blockRan)
            {
                ++neverRan;
                if (shown++ < showLimit)
                    std::printf("  block pc 0x%04x (%u bytes) declined in every case\n", info.pc, info.size);
            }
        }
        VU1Interpreter::s_pcHook = nullptr;
        const bool pass = mismatches == 0u && neverRan == 0u;
        std::printf("fuzz blocks: %s, %zu blocks x %d cases, %llu runs through the block, %llu mismatches, %llu blocks never "
                    "ran, %llu cases stopped on an error (not compared), %llu loop re-entries declined\n",
                    pass ? "PASS" : "FAIL", list.size(), cases, (unsigned long long)ran, (unsigned long long)mismatches,
                    (unsigned long long)neverRan, (unsigned long long)errorStops, (unsigned long long)s_reentryDeclines);
        return pass ? 0 : 1;
    }

    using Pair = VU1Interpreter::DecodedInstructionPair;
    using Pipeline = VU1Interpreter::Pipeline;

    struct Image
    {
        std::vector<uint8_t> code;
        std::set<uint32_t> executed;
        std::set<uint32_t> entries; // program starts and jump/branch destinations seen
    };

    static Pair decode(const std::vector<uint8_t> &code, uint32_t pc)
    {
        uint32_t lower = 0, upper = 0;
        std::memcpy(&lower, code.data() + pc, 4);
        std::memcpy(&upper, code.data() + pc + 4, 4);
        return VU1Interpreter::decodePair(lower, upper, VU1Interpreter::Unit::VU1);
    }

    static bool isNop(uint32_t lower) { return lower == 0u || lower == 0x8000033Cu; }
    static bool haltBits(const Pair &p) { return p.dBit || p.tBit; }
    static bool reserved(const Pair &p) { return p.upperUsage.reserved || p.lowerUsage.reserved; }
    static bool isBranch(const Pair &p) { return !p.iBit && p.lowerUsage.pipeline == VU1Interpreter::PipelineBranch; }
    static bool isXgkick(const Pair &p) { return !p.iBit && p.lowerUsage.pipeline == VU1Interpreter::PipelineXgkick; }
    static uint32_t opHi(uint32_t lower) { return (lower >> 25) & 0x7Fu; }
    static uint32_t special(uint32_t lower) { return (lower & 3u) | ((lower >> 4) & 0x7Cu); }

    // Pair needs every commit due by its issue cycle first: it allocates or reads flags, or uses Q/P.
    // Upper ops without an FMAC handler that read neither Q nor flags and queue nothing: MAX/MINI,
    // ITOF/FTOI, ABS (and NOP).
    static bool upperPlain(uint32_t upper)
    {
        const uint32_t op = upper & 0x3Fu;
        if (op < 0x3Cu)
            return (op >= 0x10u && op <= 0x17u) || op == 0x1Du || op == 0x1Fu || op == 0x2Bu || op == 0x2Fu;
        const uint32_t code = (upper & 3u) | ((upper >> 4) & 0x7Cu);
        return (code >= 0x10u && code <= 0x17u) || code == 0x1Du || code == 0x2Fu || code == 0x30u;
    }
    static bool upperClip(uint32_t upper) { return (upper & 0x3Fu) >= 0x3Cu && ((upper & 3u) | ((upper >> 4) & 0x7Cu)) == 0x1Fu; }

    // When a pair needs the commits due by its issue cycle first, as a C++ condition: "" never,
    // "true" always, else only while that is pending (what it reads or replaces).
    static std::string commitGuard(const Pair &p)
    {
        bool q = false, pResult = false;
        // Upper: FMACs with the Q operand read Q; anything not known to be plain might (CLIP only
        // needs a free flag ring slot, like an FMAC: see the lazy commit).
        if (p.upperHandler == nullptr && !upperPlain(p.upper) && !upperClip(p.upper))
            return "true";
        if (p.upperHandler != nullptr && VU1Interpreter::fmacSpecFor(p.upper).operand == VU1Interpreter::FmacOperand::Q)
            q = true;
        if (!p.iBit && !isNop(p.lower))
        {
            const auto pipe = p.lowerUsage.pipeline;
            const uint32_t hi = opHi(p.lower);
            if (hi >= 0x10u && hi <= 0x1Cu) // FCEQ..FCGET: clip/status/MAC reads, FCSET/FSSET
                return "true";
            if (pipe == VU1Interpreter::PipelineFdiv || p.lowerUsage.waitQ)
                q = true; // replaces or waits for the DIV unit's result
            if (pipe == VU1Interpreter::PipelineEfu || p.lowerUsage.waitP ||
                (hi == 0x40u && (p.lower & 0x3Fu) >= 0x3Cu && special(p.lower) == 0x64u)) // MFP
                pResult = true;
        }
        if (q && pResult)
            return "m_fdiv.valid || m_efu[0].valid || m_efu[1].valid";
        if (q)
            return "m_fdiv.valid";
        if (pResult)
            return "m_efu[0].valid || m_efu[1].valid";
        return "";
    }
    static bool needsCommit(const Pair &p) { return !commitGuard(p).empty(); }

    static bool isStore(const Pair &p)
    {
        if (p.iBit)
            return false;
        const uint32_t hi = opHi(p.lower);
        if (hi == 0x01u || hi == 0x05u) // SQ, ISW
            return true;
        if (hi == 0x40u && (p.lower & 0x3Fu) >= 0x3Cu)
        {
            const uint32_t op = special(p.lower);
            return op == 0x35u || op == 0x37u || op == 0x3Fu; // SQI, SQD, ISWR
        }
        return false;
    }

    // Reads or writes MAC/STATUS besides FMAC updates: FSEQ..FMOR.
    static bool macStatusAccess(const Pair &p) { return !p.iBit && opHi(p.lower) >= 0x14u && opHi(p.lower) <= 0x1Bu; }

    static uint8_t upperDest(const Pair &p) { return static_cast<uint8_t>((p.upper >> 21) & 0xFu); }
    static bool upperNop(const Pair &p) { return (p.upper & 0x7FFu) == 0x2FFu; }
    // Queues a MAC/STATUS update (an execFmacOp FMAC with a non-empty dest mask).
    static bool flagWriter(const Pair &p) { return p.upperHandler != nullptr && upperDest(p) != 0u; }
    // Might read MAC/STATUS (FC*/FS*/FM* lower ops) or touch them some other way (any upper op that
    // is not a plain FMAC or NOP: treated as a reader to stay on the safe side).
    static bool flagBarrier(const Pair &p)
    {
        if (!p.iBit && opHi(p.lower) >= 0x10u && opHi(p.lower) <= 0x1Cu)
            return true;
        return p.upperHandler == nullptr && !upperNop(p);
    }

    // Static code from the game's ELF: the main VU1 program (VIF MPG packets in .vutext) alone and
    // with each overlay for the 0x3800 slot (raw code in vu_code, listed in .DVP.ovlytab). Every pair
    // of loaded code counts as executed (all-zero padding pairs excepted); entries are the code
    // starts, branch targets, the pairs after a branch's or E-bit's delay slot (return points,
    // following programs). A pc reached any other way (JR to a computed address) is interpreted up
    // to the next block start.
    static uint32_t branchTarget(uint32_t pc, uint32_t lower)
    {
        int32_t imm = static_cast<int32_t>(lower & 0x7FFu);
        if (imm & 0x400)
            imm -= 0x800;
        return static_cast<uint32_t>(static_cast<int32_t>(pc) + 8 + imm * 8) & (kMemSize - 1u);
    }

    static void analyseStatic(Image &img, const std::vector<std::pair<uint32_t, uint32_t>> &ranges)
    {
        for (const auto &[from, to] : ranges)
        {
            img.entries.insert(from);
            for (uint32_t pc = from; pc + 8u <= to; pc += 8u)
            {
                uint64_t words = 0;
                std::memcpy(&words, img.code.data() + pc, 8);
                if (words == 0u)
                    continue;
                img.executed.insert(pc);
                const Pair d = decode(img.code, pc);
                if (reserved(d))
                    continue;
                const uint32_t hi = opHi(d.lower);
                if (isBranch(d))
                {
                    if (hi != 0x24u && hi != 0x25u) // JR/JALR: computed target
                        img.entries.insert(branchTarget(pc, d.lower));
                    img.entries.insert((pc + 16u) & (kMemSize - 1u));
                }
                if (d.eBit)
                    img.entries.insert((pc + 16u) & (kMemSize - 1u));
            }
        }
    }

    static bool loadElfImages(const char *path, std::map<uint64_t, Image> &images)
    {
        std::ifstream in(path, std::ios::binary);
        std::vector<uint8_t> elf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto u32 = [&](size_t off) -> uint32_t
        {
            uint32_t v = 0;
            if (off + 4u <= elf.size())
                std::memcpy(&v, elf.data() + off, 4);
            return v;
        };
        auto u16 = [&](size_t off) -> uint32_t { return u32(off) & 0xFFFFu; };
        if (elf.size() < 52u || std::memcmp(elf.data(), "\x7f" "ELF", 4) != 0)
            return false;
        const uint32_t shoff = u32(32), shnum = u16(48), shstrndx = u16(50);
        struct Section
        {
            std::string name;
            uint32_t addr, offset, size;
        };
        std::vector<Section> sections;
        const uint32_t strOff = u32(shoff + 40u * shstrndx + 16u);
        for (uint32_t i = 0; i < shnum; ++i)
        {
            const size_t h = shoff + 40u * i;
            const char *name = reinterpret_cast<const char *>(elf.data() + std::min<size_t>(strOff + u32(h), elf.size() - 1u));
            sections.push_back({name, u32(h + 12u), u32(h + 16u), u32(h + 20u)});
        }
        auto find = [&](const char *name) -> const Section *
        {
            for (const Section &sec : sections)
                if (sec.name == name)
                    return &sec;
            return nullptr;
        };
        const Section *vutext = find(".vutext"), *vucode = find("vu_code"), *table = find(".DVP.ovlytab");
        if (!vutext)
            return false;
        // Main program: MPG packets (VIFcode 0x4A: NUM pairs at ADDR*8, data 64-bit aligned).
        std::vector<uint8_t> main(kMemSize, 0u);
        std::vector<std::pair<uint32_t, uint32_t>> mainRanges;
        for (size_t pos = vutext->offset; pos + 4u <= size_t(vutext->offset) + vutext->size;)
        {
            const uint32_t code = u32(pos);
            pos += 4u;
            if (((code >> 24) & 0x7Fu) != 0x4Au)
                continue;
            const uint32_t num = ((code >> 16) & 0xFFu) ? ((code >> 16) & 0xFFu) : 256u;
            const uint32_t addr = (code & 0xFFFFu) * 8u;
            pos = (pos + 7u) & ~size_t(7u);
            if (addr + num * 8u > kMemSize || pos + num * 8u > elf.size())
                return false;
            std::memcpy(main.data() + addr, elf.data() + pos, num * 8u);
            mainRanges.push_back({addr, addr + num * 8u});
            pos += num * 8u;
        }
        auto add = [&](std::vector<uint8_t> code, const std::vector<std::pair<uint32_t, uint32_t>> &ranges)
        {
            Image &img = images[fnv1a(code.data(), kMemSize)];
            if (img.code.empty())
                img.code = std::move(code);
            analyseStatic(img, ranges);
        };
        add(main, mainRanges);
        if (!vucode || !table)
            return true;
        // Overlays for the 0x3800 slot: from their lma to the next overlay's (or the section end).
        std::vector<std::pair<uint32_t, uint32_t>> slot; // (lma, vma)
        for (uint32_t off = 0; off + 12u <= table->size; off += 12u)
        {
            const uint32_t lma = u32(table->offset + off + 4u), vma = u32(table->offset + off + 8u);
            if (lma >= vucode->addr && lma < vucode->addr + vucode->size)
                slot.push_back({lma, vma});
        }
        std::sort(slot.begin(), slot.end());
        for (size_t i = 0; i < slot.size(); ++i)
        {
            const auto [lma, vma] = slot[i];
            const uint32_t end = i + 1u < slot.size() ? slot[i + 1u].first : vucode->addr + vucode->size;
            const uint32_t size = std::min<uint32_t>(end - lma, kMemSize - vma) & ~7u;
            if (vma != 0x3800u || size == 0u)
                continue;
            std::vector<uint8_t> code = main;
            std::memcpy(code.data() + vma, elf.data() + vucode->offset + (lma - vucode->addr), size);
            auto ranges = mainRanges;
            ranges.push_back({vma, vma + size});
            add(std::move(code), ranges);
        }
        return true;
    }

    struct PairPlan
    {
        uint32_t pc = 0;
        Pair d{};
    };

    struct Block
    {
        uint32_t start = 0;
        std::vector<PairPlan> pairs;
        enum class End
        {
            Fallthrough, // next pc static
            Branch,      // last two pairs: branch + delay slot
            Ebit,        // last two pairs: E-bit pair + delay slot
            Xgkick       // last pair is XGKICK
        } end = End::Fallthrough;
        uint32_t nextPc = 0;
        uint64_t total = 0; // cycles from entry to exit with nothing in flight at entry
    };

    static const char *kindName(VU1Interpreter::FmacKind k)
    {
        static const char *names[] = {"Add", "Sub", "Madd", "Msub", "Mul", "Opmsub", "Opmula"};
        return names[static_cast<int>(k)];
    }
    static const char *operandName(VU1Interpreter::FmacOperand o)
    {
        static const char *names[] = {"Broadcast", "Q", "I", "Vector"};
        return names[static_cast<int>(o)];
    }

    // Longest any pipeline result can still be outstanding at block entry (the longest VU latency
    // is 54 cycles, EFU), i.e. the most the entry state can delay a block.
    static constexpr uint64_t kMaxEntryStall = 64u;

    static bool buildBlock(const Image &img, uint32_t start, Block &block)
    {
        block.start = start;
        uint32_t pc = start;
        while (pc + 8u <= kMemSize && img.executed.count(pc) && block.pairs.size() < 512u)
        {
            const Pair d = decode(img.code, pc);
            if (reserved(d) || haltBits(d))
                break;
            const bool branch = isBranch(d);
            if (branch || d.eBit)
            {
                const uint32_t slot = pc + 8u;
                if (slot + 8u > kMemSize)
                    break;
                const Pair s = decode(img.code, slot);
                if (reserved(s) || haltBits(s) || isBranch(s) || s.eBit || isXgkick(s) || (branch && d.eBit))
                    break; // leave this pair to the interpreter
                block.pairs.push_back({pc, d});
                block.pairs.push_back({slot, s});
                block.end = branch ? Block::End::Branch : Block::End::Ebit;
                block.nextPc = slot + 8u;
                return true;
            }
            block.pairs.push_back({pc, d});
            if (isXgkick(d))
            {
                block.end = Block::End::Xgkick;
                block.nextPc = pc + 8u;
                return true;
            }
            pc += 8u;
        }
        block.end = Block::End::Fallthrough;
        block.nextPc = pc;
        return !block.pairs.empty();
    }

    // Block length with nothing in flight at entry, by the interpreter's own scoreboard. With the
    // entry state delaying it by at most kMaxEntryStall, this bounds the block's cycles for the
    // budget check.
    static void schedule(Block &block)
    {
        auto sim = std::make_unique<VU1Interpreter>(VU1Interpreter::Unit::VU1);
        sim->reset();
        sim->m_cycle = 0;
        for (PairPlan &plan : block.pairs)
        {
            const Pair &d = plan.d;
            if (sim->m_fdiv.valid && sim->m_fdiv.readyCycle <= sim->m_cycle)
                sim->m_fdiv = {};
            for (auto &entry : sim->m_efu)
                if (entry.valid && entry.readyCycle <= sim->m_cycle)
                    entry = {};
            const uint64_t ready = std::max(sim->m_cycle, sim->calculatePairReadyCycle(d));
            sim->m_cycle = ready;
            sim->markPairWrites(d);
            const auto pipe = d.lowerUsage.pipeline;
            if (pipe == VU1Interpreter::PipelineFdiv && d.lowerUsage.latency != 0u)
            {
                sim->m_fdiv.valid = true;
                sim->m_fdiv.readyCycle = ready + d.lowerUsage.latency;
            }
            if (pipe == VU1Interpreter::PipelineEfu && d.lowerUsage.latency != 0u)
            {
                for (auto &entry : sim->m_efu)
                    if (!entry.valid)
                    {
                        entry.valid = true;
                        entry.readyCycle = ready + d.lowerUsage.latency;
                        break;
                    }
                sim->m_efuResourceReady = ready + d.lowerUsage.latency - 1u;
            }
            sim->m_cycle = ready + 1u;
        }
        block.total = sim->m_cycle;
    }

    static std::string hex(uint64_t v, int width = 0)
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%0*llx", width, static_cast<unsigned long long>(v));
        return buf;
    }

    static std::string vfReady(uint8_t lane) { return "m_vfReady[" + std::to_string(lane >> 2) + "][" + std::to_string(lane & 3) + "]"; }

    // One block. Each pair issues at t = max(previous issue + 1, its operands' ready cycles), the
    // interpreter's calculatePairReadyCycle with the pair folded in, so stalls on anything in
    // flight, from before the block or inside it, come out as in the interpreter.
    static void emitBlock(std::ostream &o, uint64_t codeHash, const Block &b)
    {
        o << "template <>\nVU1Interpreter::IssueResult VU1Interpreter::runCompiledBlock<" << hex(codeHash) << "ull, "
          << hex(b.start, 4) << "u>(const BlockArgs &a)\n{\n";
        o << "    const uint64_t e = m_cycle;\n";
        o << "    if (m_state.branchPending || m_state.haltAfterDelaySlot || m_state.ebit) return IssueDeclined;\n";
        o << "    if ((m_storeMask | m_vfWriteMask | m_viWriteMask | m_accWriteMask) != 0u) return IssueDeclined;\n";
        o << "    if (e + " << (b.total + kMaxEntryStall) << "u + kImmediateWriteMargin >= a.budgetEnd) return IssueDeclined;\n";
        // Per FMAC pair: does its MAC/STATUS update have to be queued? Not when the next one in the
        // block, j, overwrites it before anything can read it: pairs issue at least a cycle apart and
        // updates commit kFmacLatency (4) cycles after issue, in issue order, so only pairs up to
        // j + 3 can see the earlier value; j + 3 must be inside the block and none of those pairs may
        // read the flags. Its sticky bits still reach STATUS, with j's update.
        const size_t n = b.pairs.size();
        // Direct flags: with nothing in the block reading or writing MAC/STATUS but FMAC updates,
        // the block commits those itself at its exits (flushFlags) instead of queueing each one.
        bool direct = true;
        for (const PairPlan &plan : b.pairs)
            direct = direct && !macStatusAccess(plan.d);
        std::vector<size_t> writers;
        for (size_t k = 0; k < n; ++k)
            if (flagWriter(b.pairs[k].d))
                writers.push_back(k);
        if (writers.empty())
            direct = false;
        std::vector<bool> live(n, true);
        static_assert(VU1Interpreter::kFmacLatency == 4u);
        bool anyDead = false;
        for (size_t k = 0; k < n; ++k)
        {
            if (!flagWriter(b.pairs[k].d))
                continue;
            size_t j = k + 1;
            while (j < n && !flagWriter(b.pairs[j].d))
                ++j;
            if (j >= n || j + 3u >= n)
                continue;
            bool read = false;
            for (size_t m = k + 1; m <= j + 3u; ++m)
                read = read || flagBarrier(b.pairs[m].d);
            if (!read)
            {
                live[k] = false;
                anyDead = true;
            }
        }
        if (direct)
            anyDead = false;
        o << "    m_storesImmediate = true;\n    m_recCommitted = e;\n    uint64_t t = e;\n";
        if (direct)
            o << "    FmacFlagBits fl[" << n << "]; // FMAC flag updates, committed at the exits\n";
        if (anyDead)
            o << "    uint32_t sticky = 0u; // sticky STATUS bits of FMAC updates that are overwritten unseen\n";

        // Scoreboard folding. Last in-block write per VF lane / VI reg / ACC lane: (pair, latency);
        // every write's (pair, latency) for the pending-until marks; external lanes already checked.
        struct Write
        {
            size_t pair;
            uint32_t latency;
        };
        std::map<uint8_t, Write> vfLast, viLast, accLast;
        std::vector<Write> vfAll, viAll, accAll;
        std::set<uint8_t> vfChecked, viChecked, accChecked;
        o << "    uint64_t tk[" << b.pairs.size() << "]; // issue cycle per pair\n";
        // The scoreboard state the interpreter would have after the pairs so far.
        auto flushScoreboard = [&](const std::string &indent)
        {
            for (const auto &[lane, w] : vfLast)
                o << indent << vfReady(lane) << " = tk[" << w.pair << "] + " << w.latency << "u;\n";
            for (const auto &[reg, w] : viLast)
                o << indent << "m_viReady[" << int(reg) << "] = tk[" << w.pair << "] + " << w.latency << "u;\n";
            for (const auto &[lane, w] : accLast)
                o << indent << "m_accReady[" << int(lane) << "] = tk[" << w.pair << "] + kAccForwardLatency;\n";
            auto pending = [&](const char *name, const std::vector<Write> &all)
            {
                std::map<uint32_t, size_t> lastByLatency; // issue cycles only grow: the last write per latency
                for (const Write &w : all)
                    lastByLatency[w.latency] = w.pair;
                for (const auto &[latency, pair] : lastByLatency)
                    o << indent << name << " = std::max<uint64_t>(" << name << ", tk[" << pair << "] + " << latency << "u);\n";
            };
            pending("m_vfPendingUntil", vfAll);
            pending("m_viPendingUntil", viAll);
            pending("m_accPendingUntil", accAll);
        };
        // Commit the FMAC flag updates of the first `executed` pairs due by exit cycle `exitCycle`
        // (after the flag pipeline, whose entries are all older) and queue the rest. Pair `exitPair`
        // issued at exitCycle - exitPlus or later, so pairs 4 - exitPlus before it are certainly due.
        auto flushFlags = [&](const std::string &indent, const std::string &exitCycle, size_t executed, size_t exitPair, size_t exitPlus)
        {
            if (!direct)
                return;
            std::vector<size_t> due, maybe;
            for (size_t k : writers)
                if (k < executed)
                    (exitPair - k + exitPlus >= VU1Interpreter::kFmacLatency ? due : maybe).push_back(k);
            if (due.empty() && maybe.empty())
                return;
            o << indent << "{\n";
            if (!due.empty())
            {
                o << indent << "    uint32_t mac = fl[" << due.back() << "].mac, current = fl[" << due.back() << "].status, sticky = 0u";
                for (size_t k : due)
                    o << " | fl[" << k << "].status | fl[" << k << "].productSticky";
                o << ";\n" << indent << "    bool any = true;\n";
            }
            else
                o << indent << "    uint32_t mac = 0u, current = 0u, sticky = 0u;\n" << indent << "    bool any = false;\n";
            if (!maybe.empty())
            {
                o << indent << "    const uint64_t saved = m_cycle;\n";
                for (size_t k : maybe)
                    o << indent << "    if (tk[" << k << "] + kFmacLatency <= " << exitCycle << ") { mac = fl[" << k << "].mac; current = fl[" << k
                      << "].status; sticky |= fl[" << k << "].status | fl[" << k << "].productSticky; any = true; }\n"
                      << indent << "    else { m_cycle = tk[" << k << "]; queueFmacFlags(fl[" << k << "].mac, fl[" << k << "].status, fl[" << k
                      << "].productSticky); }\n";
                o << indent << "    m_cycle = saved;\n";
            }
            o << indent << "    if (any)\n" << indent << "    {\n" << indent << "        m_state.mac = mac;\n"
              << indent << "        m_state.status = (m_state.status & 0xFF0u) | current | (sticky << 6);\n" << indent << "    }\n"
              << indent << "}\n";
        };
        auto readyTerm = [&](size_t k, std::map<uint8_t, Write> &last, std::set<uint8_t> &checked, uint8_t key,
                             const std::string &memory)
        {
            if (auto it = last.find(key); it != last.end())
            {
                // pairs issue at least a cycle apart: k - pair cycles have passed already
                if (k - it->second.pair < it->second.latency)
                    o << "        t = std::max<uint64_t>(t, tk[" << it->second.pair << "] + " << it->second.latency << "u);\n";
            }
            else if (checked.insert(key).second)
                o << "        t = std::max(t, " << memory << ");\n"; // later reads issue later still
        };

        for (size_t i = 0; i < b.pairs.size(); ++i)
        {
            const PairPlan &p = b.pairs[i];
            const Pair &d = p.d;
            const bool isDelaySlot = (b.end == Block::End::Branch || b.end == Block::End::Ebit) && i + 1 == b.pairs.size();
            const bool branchPair = b.end == Block::End::Branch && i + 2 == b.pairs.size();
            const uint8_t writtenVi = d.viWriteCount ? d.viWrites[0] : 0u;
            const bool records = writtenVi != 0u && d.lowerUsage.delaysNextBranchRead;
            o << "    // " << hex(p.pc, 4) << "  " << hex(d.lower, 8) << " " << hex(d.upper, 8) << "\n    {\n";
            if (i != 0)
                o << "        ++t;\n";
            for (uint32_t k = 0; k < d.vfReadLaneCount; ++k)
                readyTerm(i, vfLast, vfChecked, d.vfReadLanes[k], vfReady(d.vfReadLanes[k]));
            for (uint32_t k = 0; k < d.viReadCount; ++k)
                readyTerm(i, viLast, viChecked, d.viReads[k], "m_viReady[" + std::to_string(d.viReads[k]) + "]");
            for (uint32_t k = 0; k < d.accReadCount; ++k)
                readyTerm(i, accLast, accChecked, d.accReads[k], "m_accReady[" + std::to_string(d.accReads[k]) + "]");
            if (d.lowerUsage.pipeline == VU1Interpreter::PipelineFdiv || d.lowerUsage.waitQ)
                o << "        if (m_fdiv.valid) t = std::max(t, m_fdiv.readyCycle);\n";
            if (d.lowerUsage.pipeline == VU1Interpreter::PipelineEfu)
                o << "        t = std::max(t, m_efuResourceReady);\n";
            if (d.lowerUsage.waitP)
                o << "        for (const ScalarPipelineEntry &entry : m_efu) if (entry.valid) t = std::max(t, entry.readyCycle);\n";
            if (d.lowerUsage.pipeline == VU1Interpreter::PipelineXgkick)
            {
                // Waits for the previous PATH1 transfer, which moves at every cycle boundary: leave
                // that to the interpreter, from the state it would have at this pair.
                o << "        recCommitTo(t);\n        if (m_xgkick.active)\n        {\n";
                flushScoreboard("            ");
                flushFlags("            ", "t", i, i, 0u);
                o << "            m_storesImmediate = false;\n            m_cycle = t;\n            m_state.cycles = t;\n"
                  << "            m_state.pc = " << hex(p.pc, 4) << "u;\n            return IssueDeclined;\n        }\n";
            }
            o << "        m_cycle = t;\n        tk[" << i << "] = t;\n";
            if (const std::string guard = commitGuard(d); guard == "true")
                o << "        recCommitTo(t);\n";
            else if (!guard.empty())
                o << "        if (" << guard << ") recCommitTo(t);\n";
            else if ((flagWriter(d) && live[i] && !direct) || (d.upperHandler == nullptr && upperClip(d.upper)))
                o << "        if (m_recCommitted + kFmacLatency < t) recCommitTo(t - kFmacLatency); // its flag ring slot is free\n";
            else if (isStore(d))
                o << "        if (m_xgkick.active) recCommitTo(t); // PATH1 reads VU memory up to this cycle first\n";
            if (records)
                o << "        const int32_t oldVi = m_state.vi[" << int(writtenVi) << "];\n";
            if (isDelaySlot && records)
                o << "        const bool inTakenDelaySlot = m_state.branchPending && m_state.branchDelay == 0u;\n";
            if (branchPair)
                o << "        m_state.pc = " << hex(p.pc, 4) << "u;\n";
            std::string upper;
            if (upperNop(d))
                upper = "";
            else if (flagWriter(d) && direct)
            {
                const auto spec = VU1Interpreter::fmacSpecFor(d.upper);
                upper = "fl[" + std::to_string(i) + "] = execFmacDirect<FmacKind::" + kindName(spec.kind) + ", FmacOperand::" +
                        operandName(spec.operand) + ", " + (spec.toAcc ? "true" : "false") + ", " + std::to_string(upperDest(d)) + "u>(" +
                        hex(d.upper, 8) + "u);";
            }
            else if (flagWriter(d))
            {
                const auto spec = VU1Interpreter::fmacSpecFor(d.upper);
                upper = std::string("execFmacRec<FmacKind::") + kindName(spec.kind) + ", FmacOperand::" + operandName(spec.operand) + ", " +
                        (spec.toAcc ? "true" : "false") + ", " + std::to_string(upperDest(d)) + "u, " + (live[i] ? "true" : "false") + ">(" +
                        hex(d.upper, 8) + "u, " + (anyDead ? "sticky" : "noSticky") + ");";
                if (!anyDead)
                    upper = "{ uint32_t noSticky = 0u; " + upper + " }";
            }
            else if (d.upperHandler)
                upper = "{ static constexpr UpperHandler h = fmacHandlerFor(" + hex(d.upper, 8) + "u); (this->*h)(" + hex(d.upper, 8) + "u); }";
            else
                upper = "execUpperInline(" + hex(d.upper, 8) + "u);";
            const std::string lower = "execLowerInline(" + hex(d.lower, 8) + "u, a.vuData, a.dataSize, *a.gs, a.memory, " + hex(d.upper, 8) + "u);";
            if (d.iBit)
            {
                o << "        " << upper << "\n";
                o << "        { const uint32_t imm = " << hex(d.lower, 8) << "u; float f; std::memcpy(&f, &imm, 4); m_state.i = normalizeOperand(f); }\n";
            }
            else if (d.upperVfShadowReg != 0u)
            {
                const int r = d.upperVfShadowReg;
                o << "        float shadowOld[4], shadowUp[4];\n";
                o << "        std::memcpy(shadowOld, m_state.vf[" << r << "], 16);\n";
                o << "        " << upper << "\n";
                o << "        std::memcpy(shadowUp, m_state.vf[" << r << "], 16);\n";
                o << "        std::memcpy(m_state.vf[" << r << "], shadowOld, 16);\n";
                if (!isNop(d.lower))
                    o << "        " << lower << "\n";
                o << "        std::memcpy(m_state.vf[" << r << "], shadowUp, 16);\n";
            }
            else
            {
                o << "        " << upper << "\n";
                if (!isNop(d.lower))
                    o << "        " << lower << "\n";
            }
            o << "        m_viBranchBackupValid = false;\n";
            if (writtenVi != 0u)
                o << "        m_state.vi[" << int(writtenVi) << "] = static_cast<int16_t>(m_state.vi[" << int(writtenVi) << "]);\n";
            if (records)
            {
                if (isDelaySlot)
                    o << "        if (!inTakenDelaySlot)\n    ";
                o << "        recordViWriteForBranch(" << int(writtenVi) << ", oldVi);\n";
            }
            // markPairWrites, folded: tracked here, stored at the exits
            for (uint32_t k = 0; k < d.lowerVfWriteCount; ++k)
            {
                vfLast[d.lowerVfWriteLanes[k]] = {i, d.lowerVfWriteLatency};
                vfAll.push_back({i, d.lowerVfWriteLatency});
            }
            for (uint32_t k = 0; k < d.upperVfWriteCount; ++k)
            {
                vfLast[d.upperVfWriteLanes[k]] = {i, d.upperVfWriteLatency};
                vfAll.push_back({i, d.upperVfWriteLatency});
            }
            for (uint32_t k = 0; k < d.viWriteCount; ++k)
            {
                viLast[d.viWrites[k]] = {i, d.viWriteLatency};
                viAll.push_back({i, d.viWriteLatency});
            }
            for (uint32_t k = 0; k < d.accWriteCount; ++k)
            {
                accLast[d.accWrites[k]] = {i, VU1Interpreter::kAccForwardLatency};
                accAll.push_back({i, VU1Interpreter::kAccForwardLatency});
            }
            o << "        m_state.vf[0][0] = 0.0f; m_state.vf[0][1] = 0.0f; m_state.vf[0][2] = 0.0f; m_state.vf[0][3] = 1.0f; m_state.vi[0] = 0;\n";
            if (branchPair)
                o << "        if (m_state.branchPending) --m_state.branchDelay;\n";
            if (d.eBit)
                o << "        m_state.ebit = true;\n";
            o << "    }\n";
        }

        flushScoreboard("    ");
        o << "    m_storesImmediate = false;\n";
        switch (b.end)
        {
        case Block::End::Xgkick:
            // XGKICK starts PATH1; from here on the interpreter moves it at every cycle boundary.
            o << "    recCommitTo(t);\n";
            flushFlags("    ", "t", n, n - 1u, 0u);
            o << "    m_state.pc = " << hex(b.nextPc, 4) << "u;\n    advanceOneCycle();\n    return IssueContinue;\n";
            break;
        case Block::End::Branch:
            o << "    m_cycle = t + 1u;\n    recCommitTo(m_cycle);\n    m_state.cycles = m_cycle;\n";
            flushFlags("    ", "m_cycle", n, n - 1u, 1u);
            o << "    if (m_state.branchPending && m_state.branchDelay == 0u)\n    {\n"
                 "        m_state.pc = m_state.branchTarget & 0x3FFFu;\n        m_state.branchPending = false;\n    }\n"
                 "    else\n        m_state.pc = "
              << hex(b.nextPc & 0x3FFFu, 4) << "u;\n    return IssueContinue;\n";
            break;
        case Block::End::Ebit:
            o << "    m_cycle = t + 1u;\n    recCommitTo(m_cycle);\n    m_state.cycles = m_cycle;\n";
            flushFlags("    ", "m_cycle", n, n - 1u, 1u);
            o << "    m_state.pc = " << hex(b.nextPc & 0x3FFFu, 4) << "u;\n    return IssueEnded;\n";
            break;
        case Block::End::Fallthrough:
            o << "    m_cycle = t + 1u;\n    recCommitTo(m_cycle);\n    m_state.cycles = m_cycle;\n";
            flushFlags("    ", "m_cycle", n, n - 1u, 1u);
            o << "    m_state.pc = " << hex(b.nextPc & 0x3FFFu, 4) << "u;\n    return IssueContinue;\n";
            break;
        }
        o << "}\n\n";
    }
};

namespace
{
    // The run cycle cap of the trace being replayed (ps2_vu1_trace::runCycleCap).
    uint32_t g_runCycleCap = 65536u;
    struct Record
    {
        Vu1TraceRecordHeader header{};
        const uint8_t *code = nullptr;
        const uint8_t *data = nullptr;
        VU1State before{};
    };

    bool loadTrace(const char *path, std::vector<uint8_t> &bytes, std::vector<Record> &records)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            return false;
        bytes.assign(std::istreambuf_iterator<char>(f), {});
        if (bytes.size() < 12u || std::memcmp(bytes.data(), kMagic, 8) != 0)
            return false;
        uint32_t version = 0u;
        std::memcpy(&version, bytes.data() + 8, 4);
        if (!readableVersion(version))
            return false;
        g_runCycleCap = runCycleCap(version);
        size_t pos = 12u;
        const uint8_t *code = nullptr;
        while (pos + sizeof(Vu1TraceRecordHeader) <= bytes.size())
        {
            Record r;
            std::memcpy(&r.header, bytes.data() + pos, sizeof(r.header));
            pos += sizeof(r.header);
            if (r.header.stateSize != sizeof(VU1State))
                return false;
            if (r.header.codeIncluded)
            {
                code = bytes.data() + pos;
                pos += kMemSize;
            }
            r.code = code;
            r.data = bytes.data() + pos;
            pos += kMemSize;
            std::memcpy(&r.before, bytes.data() + pos, sizeof(VU1State));
            pos += 2 * sizeof(VU1State) + 8 + 8 + 4 + 4;
            if (pos > bytes.size() || !code)
                return false;
            records.push_back(r);
        }
        return true;
    }

    std::map<uint64_t, Vu1RecGenerator::Image> *g_images = nullptr;
    uint64_t g_currentImage = 0;
    uint32_t g_lastPc = 0xFFFFFFFFu;
    bool g_runStart = true;

    void onPc(const uint8_t *, uint32_t pc)
    {
        auto &img = (*g_images)[g_currentImage];
        img.executed.insert(pc);
        if (g_runStart || pc != ((g_lastPc + 8u) & (kMemSize - 1u)))
            img.entries.insert(pc);
        g_lastPc = pc;
        g_runStart = false;
    }
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--selftest-fmac") == 0)
        return Vu1RecGenerator::selfTestFmac();
    if (argc == 3 && std::strcmp(argv[1], "--check-fmac-ref") == 0)
        return Vu1RecGenerator::checkFmacRef(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--fuzz-blocks") == 0)
    {
        if (std::fesetround(FE_TOWARDZERO) != 0)
            return 1;
        return Vu1RecGenerator::fuzzBlocks(std::max(1, std::atoi(argv[2])));
    }
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: ps2x_vu1_rec_gen <out dir> [--elf <game ELF>] <trace.vu1trace>...\n");
        return 2;
    }
    setenv("PS2X_VU1_NO_AOT", "1", 1); // coverage must see every pair interpreted
    std::map<uint64_t, Vu1RecGenerator::Image> images;
    g_images = &images;

    PS2Memory memory;
    memory.initialize();
    GifArbiter arbiter([](GifPathId, const uint8_t *, uint32_t) {});
    memory.setGifArbiter(&arbiter);
    GS gs;
    VU1Interpreter vu1(VU1Interpreter::Unit::VU1);
    VU1Interpreter::s_pcHook = onPc;

    int firstTrace = 2;
    if (argc >= 4 && std::strcmp(argv[2], "--elf") == 0)
    {
        if (!Vu1RecGenerator::loadElfImages(argv[3], images))
        {
            std::fprintf(stderr, "%s: no VU1 code found (.vutext)\n", argv[3]);
            return 2;
        }
        std::printf("%s: %zu static image(s)\n", argv[3], images.size());
        firstTrace = 4;
    }
    for (int t = firstTrace; t < argc; ++t)
    {
        std::vector<uint8_t> bytes;
        std::vector<Record> records;
        if (!loadTrace(argv[t], bytes, records))
        {
            std::fprintf(stderr, "%s: not a readable VU1 trace\n", argv[t]);
            return 2;
        }
        vu1.reset();
        const uint8_t *loaded = nullptr;
        for (const Record &r : records)
        {
            if (r.code != loaded)
            {
                std::memcpy(memory.getVU1Code(), r.code, kMemSize);
                memory.markVU1CodeModified();
                loaded = r.code;
                g_currentImage = fnv1a(r.code, kMemSize);
                auto &img = images[g_currentImage];
                if (img.code.empty())
                    img.code.assign(r.code, r.code + kMemSize);
            }
            std::memcpy(memory.getVU1Data(), r.data, kMemSize);
            vu1.restoreState(r.before);
            g_runStart = true;
            if (r.header.kind == static_cast<uint32_t>(RunKind::Execute))
                vu1.execute(memory.getVU1Code(), kMemSize, memory.getVU1Data(), kMemSize, gs, &memory, r.header.startPc,
                            r.header.top, r.header.itop, g_runCycleCap);
            else
                vu1.resume(memory.getVU1Code(), kMemSize, memory.getVU1Data(), kMemSize, gs, &memory, r.header.top,
                           r.header.itop, g_runCycleCap);
        }
    }
    VU1Interpreter::s_pcHook = nullptr;

    const std::filesystem::path outDir = argv[1];
    std::filesystem::create_directories(outDir);
    for (const auto &entry : std::filesystem::directory_iterator(outDir))
        if (entry.path().filename().string().rfind("vu1_aot_", 0) == 0)
            std::filesystem::remove(entry.path());

    // Blocks from every image, deduplicated by (pc, code): a block depends only on its own words.
    struct Unique
    {
        Vu1RecGenerator::Block block;
        uint64_t hash;
        std::vector<uint8_t> code;
    };
    std::map<std::pair<uint32_t, uint64_t>, Unique> unique;
    for (const auto &[id, img] : images)
    {
        size_t pairs = 0, blockCount = 0, added = 0;
        // Blocks start at entries and wherever a block (or a pair left to the interpreter) falls through.
        std::set<uint32_t> starts(img.entries.begin(), img.entries.end()), done;
        while (!starts.empty())
        {
            const uint32_t start = *starts.begin();
            starts.erase(starts.begin());
            if (!done.insert(start).second)
                continue;
            Vu1RecGenerator::Block block;
            if (!Vu1RecGenerator::buildBlock(img, start, block))
            {
                if (img.executed.count(start + 8u))
                    starts.insert(start + 8u); // the interpreter runs this pair, then the next one is a start
                continue;
            }
            if (img.executed.count(block.nextPc))
                starts.insert(block.nextPc);
            if (block.end == Vu1RecGenerator::Block::End::Fallthrough && img.executed.count(block.nextPc + 8u))
                starts.insert(block.nextPc + 8u);
            pairs += block.pairs.size();
            ++blockCount;
            std::vector<uint8_t> code(img.code.begin() + block.start, img.code.begin() + block.start + block.pairs.size() * 8u);
            const uint64_t hash = fnv1a(code.data(), code.size());
            auto [it, inserted] = unique.try_emplace({block.start, hash});
            if (inserted)
            {
                Vu1RecGenerator::schedule(block);
                it->second = {std::move(block), hash, std::move(code)};
                ++added;
            }
        }
        std::printf("image %016llx: %zu executed pairs, %zu entries, %zu blocks (%zu new), %zu pairs translated\n",
                    static_cast<unsigned long long>(id), img.executed.size(), img.entries.size(), blockCount, added, pairs);
    }

    // Blocks go into several files so they compile in parallel; vu1_aot_table.cpp registers them.
    const std::string header = "// Generated by ps2x_vu1_rec_gen (T-0121) from VU1 traces and the game ELF. Do not edit; disc-derived.\n"
                               "#include \"../lib/vu/ps2_vu1_issue.h\"\n#include \"../lib/vu/ps2_vu1_lower.h\"\n#include \"../lib/vu/ps2_vu1_upper.h\"\n"
                               "\n";
    std::vector<const Unique *> list;
    for (const auto &[key, u] : unique)
        list.push_back(&u);
    constexpr size_t kBlocksPerFile = 24u;
    size_t totalPairs = 0;
    for (size_t first = 0, part = 0; first < list.size(); first += kBlocksPerFile, ++part)
    {
        char name[48];
        std::snprintf(name, sizeof(name), "vu1_aot_blocks_%03zu.cpp", part);
        std::ofstream bo(outDir / name);
        bo << header;
        for (size_t i = first; i < std::min(list.size(), first + kBlocksPerFile); ++i)
        {
            Vu1RecGenerator::emitBlock(bo, list[i]->hash, list[i]->block);
            totalPairs += list[i]->block.pairs.size();
        }
    }
    std::ofstream o(outDir / "vu1_aot_table.cpp");
    o << header;
    auto fnName = [](const Unique &u)
    {
        return "VU1Interpreter::runCompiledBlock<" + Vu1RecGenerator::hex(u.hash) + "ull, " + Vu1RecGenerator::hex(u.block.start, 4) + "u>";
    };
    for (const Unique *u : list)
        o << "template <>\nVU1Interpreter::IssueResult " << fnName(*u) << "(const BlockArgs &a);\n";
    o << "\nnamespace\n{\n";
    for (size_t i = 0; i < list.size(); ++i)
    {
        o << "    alignas(8) const uint8_t kCode" << i << "[] = {";
        for (size_t b = 0; b < list[i]->code.size(); ++b)
            o << (b % 32u == 0u ? "\n        " : "") << Vu1RecGenerator::hex(list[i]->code[b], 2) << ",";
        o << "};\n";
    }
    o << "    const VU1Interpreter::CompiledBlockInfo kBlocks[] = {\n";
    for (size_t i = 0; i < list.size(); ++i)
        o << "        {" << Vu1RecGenerator::hex(list[i]->block.start, 4) << "u, " << list[i]->code.size() << "u, kCode" << i
          << ", &" << fnName(*list[i]) << "},\n";
    o << "    };\n}\n\nvoid ps2xVu1AotRegister()\n{\n    VU1Interpreter::registerCompiledBlocks(kBlocks, " << list.size()
      << "u);\n}\n";
    std::printf("%zu unique blocks, %zu pairs translated\n", list.size(), totalPairs);
    return 0;
}
