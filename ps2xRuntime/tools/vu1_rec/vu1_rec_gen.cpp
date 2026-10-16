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
// Output is disc-derived (game code): not for git.

#include "../../src/lib/vu/ps2_vu1_issue.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1_trace.h"

#include <cfenv>
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
    static void emitBlock(std::ostream &o, uint64_t imageId, const Block &b)
    {
        o << "template <>\nVU1Interpreter::IssueResult VU1Interpreter::runCompiledBlock<" << hex(imageId) << "ull, "
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
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: ps2x_vu1_rec_gen <out dir> <trace.vu1trace>...\n");
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

    for (int t = 2; t < argc; ++t)
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
                            r.header.top, r.header.itop, 65536);
            else
                vu1.resume(memory.getVU1Code(), kMemSize, memory.getVU1Data(), kMemSize, gs, &memory, r.header.top,
                           r.header.itop, 65536);
        }
    }
    VU1Interpreter::s_pcHook = nullptr;

    const std::filesystem::path outDir = argv[1];
    std::filesystem::create_directories(outDir);
    for (const auto &entry : std::filesystem::directory_iterator(outDir))
        if (entry.path().filename().string().rfind("vu1_aot_", 0) == 0)
            std::filesystem::remove(entry.path());

    std::vector<std::string> registrars;
    for (const auto &[id, img] : images)
    {
        std::vector<Vu1RecGenerator::Block> blocks;
        size_t pairs = 0;
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
            Vu1RecGenerator::schedule(block);
            pairs += block.pairs.size();
            blocks.push_back(std::move(block));
        }
        char name[32];
        std::snprintf(name, sizeof(name), "%016llx", static_cast<unsigned long long>(id));
        // Blocks go into several files so they compile in parallel; the image file holds the table.
        const std::string header = "// Generated by ps2x_vu1_rec_gen (T-0121) from recorded VU1 traces. Do not edit; disc-derived.\n"
                                   "#include \"../lib/vu/ps2_vu1_issue.h\"\n#include \"../lib/vu/ps2_vu1_lower.h\"\n#include \"../lib/vu/ps2_vu1_upper.h\"\n"
                                   "\n";
        constexpr size_t kBlocksPerFile = 24u;
        for (size_t first = 0, part = 0; first < blocks.size(); first += kBlocksPerFile, ++part)
        {
            std::ofstream bo(outDir / ("vu1_aot_" + std::string(name) + "_" + std::to_string(part) + ".cpp"));
            bo << header;
            for (size_t i = first; i < std::min(blocks.size(), first + kBlocksPerFile); ++i)
                Vu1RecGenerator::emitBlock(bo, id, blocks[i]);
        }
        std::ofstream o(outDir / ("vu1_aot_" + std::string(name) + ".cpp"));
        o << header;
        for (const auto &block : blocks)
            o << "template <>\nVU1Interpreter::IssueResult VU1Interpreter::runCompiledBlock<" << Vu1RecGenerator::hex(id) << "ull, "
              << Vu1RecGenerator::hex(block.start, 4) << "u>(const BlockArgs &a);\n";
        o << "\nnamespace\n{\n    alignas(16) const uint8_t kImage[16384] = {\n";
        for (uint32_t off = 0; off < kMemSize; off += 32)
        {
            o << "        ";
            for (uint32_t i = 0; i < 32; ++i)
                o << Vu1RecGenerator::hex(img.code[off + i], 2) << ",";
            o << "\n";
        }
        o << "    };\n";
        o << "    const uint32_t kPcs[] = {";
        for (const auto &block : blocks)
            o << Vu1RecGenerator::hex(block.start, 4) << "u, ";
        o << "};\n    const VU1Interpreter::CompiledBlock kBlocks[] = {\n";
        for (const auto &block : blocks)
            o << "        &VU1Interpreter::runCompiledBlock<" << Vu1RecGenerator::hex(id) << "ull, " << Vu1RecGenerator::hex(block.start, 4) << "u>,\n";
        o << "    };\n}\n\nvoid ps2xVu1AotRegister_" << name << "()\n{\n    VU1Interpreter::registerCompiledImage({kImage, kPcs, kBlocks, "
          << blocks.size() << "u});\n}\n";
        registrars.push_back(name);
        std::printf("image %s: %zu executed pairs, %zu entries, %zu blocks, %zu pairs translated\n", name, img.executed.size(),
                    img.entries.size(), blocks.size(), pairs);
    }
    std::ofstream reg(outDir / "vu1_aot_register.cpp");
    reg << "// Generated by ps2x_vu1_rec_gen (T-0121). Do not edit.\n";
    for (const auto &n : registrars)
        reg << "void ps2xVu1AotRegister_" << n << "();\n";
    reg << "\nvoid ps2xVu1AotRegister()\n{\n";
    for (const auto &n : registrars)
        reg << "    ps2xVu1AotRegister_" << n << "();\n";
    reg << "}\n";
    return 0;
}
