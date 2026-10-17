// ps2x_vif1_pcsx2_diff: differential test of the runtime's VIF1 command interpreter
// (PS2Memory::processVIF1Data, src/lib/ps2_vif1_interpreter.cpp) against PCSX2 2.8.2's VIF1
// interpreter (pcsx2_vif_ref.cpp: Vif_Codes.cpp, Vif_Transfer.cpp, Vif_Unpack.cpp).
//
// Each case is a random VIF1 packet (reproducible from --seed and the case number) of NOP, STCYCL,
// OFFSET, BASE, ITOP, STMOD, MSKPATH3, MARK, FLUSHE/FLUSH/FLUSHA, MSCAL/MSCALF/MSCNT, STMASK, STROW,
// STCOL, MPG and UNPACK (every vn/vl, m, usn, flg, num 1..256, addresses near the 1K-qword wrap),
// from a random initial state (VU1 data and micro memory, VIF1 registers) loaded identically into
// both. VU1 does not run: both sides only record the starts (pc, TOP, ITOP). The whole packet goes
// in as one transfer. DIRECT/DIRECTHL (the GIF) and the i bit are not generated.
//
// Compared afterwards: VU1 data memory, VU1 micro memory, ROW0-3, COL0-3, MASK, CYCLE, MODE, NUM,
// CODE, TOP, TOPS (10 bits: PCSX2 keeps BASE+OFST unmasked), ITOP, ITOPS, BASE, OFST, MARK,
// STAT.DBF/MRK, and the VU1 start sequence (MSCAL pc compared as the runtime's start byte address
// & 0x3fff, which the runtime's VU1 masks it with, against PCSX2's TPC * 8).
//
//   ps2x_vif1_pcsx2_diff [--seed S] [--cases N] [--case K] [-v] [--defined] [--allow EDGE]
//                        [--ignore F,F..] [--minimize] [--examples N]
// --defined   only well-defined input (see Generator): no MODE 3, no CL/WL 0, filling writes and
//             V3 W lanes always masked by MASK (m bit set), where PCSX2's C unpackers, its dynarec
//             and the manual disagree or say "indeterminate".
// --allow     with --defined: turn one such edge back on (mode3, cyclezero, fill, v3w; repeatable).
// --ignore    leave fields out of the comparison (names as printed, e.g. num,vudata).
// --minimize  shrink each printed mismatching case (drop commands while the same fields differ).
// --case K -v dumps the packet and every difference of case K.
// Exit status: 0 all cases match, 1 mismatches, 2 usage error.

#include "runtime/ps2_memory.h"
#include "vif1_ref.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kMemSize = 0x4000u;

    struct Rng
    {
        uint64_t s;
        uint64_t next()
        {
            uint64_t z = (s += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }
        uint32_t u32() { return static_cast<uint32_t>(next() >> 32); }
        uint32_t below(uint32_t n) { return static_cast<uint32_t>((next() >> 32) % n); }
        bool pct(uint32_t p) { return below(100u) < p; }
    };

    // Bytes of source data per vector, [vn * 4 + vl] (PCSX2 nVifT; 0 = invalid).
    constexpr uint8_t kVecBytes[16] = {4, 2, 1, 0, 8, 4, 2, 0, 12, 6, 3, 0, 16, 8, 4, 2};
    const char *const kVnName[4] = {"S", "V2", "V3", "V4"};
    const char *const kVlName[4] = {"32", "16", "8", "5"};

    enum Op : uint8_t
    {
        NOP = 0x00, STCYCL = 0x01, OFFSET = 0x02, BASE = 0x03, ITOP = 0x04, STMOD = 0x05, MSKPATH3 = 0x06,
        MARK = 0x07, FLUSHE = 0x10, FLUSH = 0x11, FLUSHA = 0x13, MSCAL = 0x14, MSCALF = 0x15, MSCNT = 0x17,
        STMASK = 0x20, STROW = 0x30, STCOL = 0x31, MPG = 0x4A,
    };

    // One VIFcode with its payload pool; the payload length of an UNPACK depends on CYCLE at that
    // point, so packets are assembled from the command list (which the minimizer edits).
    struct Cmd
    {
        uint8_t op;   // cmd field (bits 30..24)
        uint8_t num;
        uint16_t imm;
        std::vector<uint32_t> pool; // payload words (UNPACK: the longest possible, cut on assembly)
    };

    struct Case
    {
        vif1ref::Regs init{};
        std::vector<uint8_t> data, micro;
        std::vector<Cmd> cmds;
    };

    bool isUnpack(uint8_t op) { return (op & 0x60u) == 0x60u; }

    // Payload words of an UNPACK under CYCLE (PCSX2 vifUnpackSetup; the oracle decides the length).
    uint32_t unpackWords(uint8_t op, uint8_t numField, uint32_t cycle)
    {
        const uint32_t gsize = kVecBytes[op & 0x0Fu];
        const uint32_t num = numField ? numField : 256u;
        const uint32_t cl = cycle & 0xFFu;
        const uint32_t wl = ((cycle >> 8) & 0xFFu) ? ((cycle >> 8) & 0xFFu) : 256u;
        if (wl <= cl)
            return (num * gsize + 3u) / 4u;
        const uint32_t n = cl * (num / wl) + std::min(num % wl, cl);
        return (n * gsize + 3u) >> 2;
    }

    // Packet words (plus 4 zero words of slack: V3 unpacks read one element past their data).
    std::vector<uint32_t> assemble(const Case &c, std::vector<size_t> *starts = nullptr)
    {
        std::vector<uint32_t> w;
        uint32_t cycle = c.init.cycle;
        for (const Cmd &cmd : c.cmds)
        {
            if (starts)
                starts->push_back(w.size());
            w.push_back((static_cast<uint32_t>(cmd.op) << 24) | (static_cast<uint32_t>(cmd.num) << 16) | cmd.imm);
            size_t n = cmd.pool.size();
            if (isUnpack(cmd.op))
                n = std::min<size_t>(n, unpackWords(cmd.op, cmd.num, cycle));
            w.insert(w.end(), cmd.pool.begin(), cmd.pool.begin() + static_cast<ptrdiff_t>(n));
            if (cmd.op == STCYCL)
                cycle = cmd.imm;
        }
        return w;
    }

    std::string describe(const Cmd &c)
    {
        char buf[160];
        const uint32_t num = c.num ? c.num : 256u;
        if (isUnpack(c.op))
        {
            std::snprintf(buf, sizeof(buf), "UNPACK %s-%s%s%s%s addr=0x%03x num=%u (imm=0x%04x)",
                          kVnName[(c.op >> 2) & 3u], kVlName[c.op & 3u], (c.op & 0x10u) ? " m" : "",
                          (c.imm & 0x4000u) ? " usn" : "", (c.imm & 0x8000u) ? " flg" : "", c.imm & 0x3FFu, num, c.imm);
            return buf;
        }
        switch (c.op)
        {
        case NOP: return "NOP";
        case STCYCL: std::snprintf(buf, sizeof(buf), "STCYCL cl=%u wl=%u", c.imm & 0xFFu, c.imm >> 8); return buf;
        case OFFSET: std::snprintf(buf, sizeof(buf), "OFFSET 0x%04x", c.imm); return buf;
        case BASE: std::snprintf(buf, sizeof(buf), "BASE 0x%04x", c.imm); return buf;
        case ITOP: std::snprintf(buf, sizeof(buf), "ITOP 0x%04x", c.imm); return buf;
        case STMOD: std::snprintf(buf, sizeof(buf), "STMOD 0x%04x", c.imm); return buf;
        case MSKPATH3: std::snprintf(buf, sizeof(buf), "MSKPATH3 0x%04x", c.imm); return buf;
        case MARK: std::snprintf(buf, sizeof(buf), "MARK 0x%04x", c.imm); return buf;
        case FLUSHE: return "FLUSHE";
        case FLUSH: return "FLUSH";
        case FLUSHA: return "FLUSHA";
        case MSCAL: std::snprintf(buf, sizeof(buf), "MSCAL 0x%04x", c.imm); return buf;
        case MSCALF: std::snprintf(buf, sizeof(buf), "MSCALF 0x%04x", c.imm); return buf;
        case MSCNT: return "MSCNT";
        case STMASK: std::snprintf(buf, sizeof(buf), "STMASK 0x%08x", c.pool[0]); return buf;
        case STROW:
        case STCOL:
            std::snprintf(buf, sizeof(buf), "%s %08x %08x %08x %08x", c.op == STROW ? "STROW" : "STCOL", c.pool[0],
                          c.pool[1], c.pool[2], c.pool[3]);
            return buf;
        case MPG: std::snprintf(buf, sizeof(buf), "MPG addr=0x%04x num=%u", c.imm, num); return buf;
        default: std::snprintf(buf, sizeof(buf), "cmd 0x%02x", c.op); return buf;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Generator
    // ---------------------------------------------------------------------------------------
    // Input areas where PCSX2's C unpackers, its dynarec and the EE manual disagree or say
    // "indeterminate"; all on by default, all off with --defined, one back on with --allow NAME.
    struct Edges
    {
        bool mode3 = true;     // STMOD 3
        bool cycleZero = true; // STCYCL with CL = 0 or WL = 0
        bool fill = true;      // filling write (CL < WL) with the filled cycles not fully masked
        bool v3w = true;       // V3 unpacks with the W lane not masked
    };
    struct Generator
    {
        Rng rng;
        Edges e;
        uint32_t cycle = 0, mask = 0;

        uint32_t randCycle()
        {
            uint32_t cl, wl;
            switch (rng.below(e.cycleZero ? 5u : 4u))
            {
            case 0: cl = wl = 1u + rng.below(4u); break;                  // plain
            case 1: wl = 1u + rng.below(4u); cl = wl + 1u + rng.below(4u); break; // skipping
            case 2:                                                         // filling
                cl = 1u + rng.below(e.fill ? 4u : 3u);
                wl = cl + 1u + rng.below(4u);
                break;
            case 3: cl = 1u + rng.below(8u); wl = 1u + rng.below(8u); break;
            default: // the CL/WL = 0 edges (WL 0 = 256 in PCSX2)
                cl = rng.below(5u);
                wl = rng.pct(70) ? 0u : rng.below(5u);
                break;
            }
            if (!e.fill && cl < wl && cl > 3u)
                cl = 3u;
            return cl | (wl << 8);
        }

        uint16_t junkImm(uint16_t used, uint16_t value)
        {
            return rng.pct(15) ? static_cast<uint16_t>(value | (rng.u32() & ~used & 0xFFFFu)) : value;
        }
        uint8_t ignoredNum() { return rng.pct(20) ? static_cast<uint8_t>(rng.u32()) : 0u; }

        std::vector<uint32_t> words(size_t n)
        {
            std::vector<uint32_t> w(n);
            for (uint32_t &x : w)
                x = rng.u32();
            return w;
        }

        // Edges off: make MASK cover what the hardware/PCSX2 leave open for this UNPACK.
        void defineMask(Case &c, uint8_t op)
        {
            const uint32_t cl = cycle & 0xFFu, wl = (cycle >> 8) & 0xFFu;
            const bool fill = cl < wl;
            const bool v3 = ((op >> 2) & 3u) == 2u;
            uint32_t m = mask;
            for (uint32_t row = 0; row < 4u; ++row)
                for (uint32_t f = 0; f < 4u; ++f)
                {
                    const bool open = (!e.fill && fill && row >= cl) || (!e.v3w && v3 && f == 3u);
                    const uint32_t sh = (row * 4u + f) * 2u;
                    if (open && ((m >> sh) & 3u) == 0u)
                        m |= (1u + rng.below(3u)) << sh;
                }
            if (m != mask)
            {
                c.cmds.push_back({STMASK, 0u, 0u, {m}});
                mask = m;
            }
        }

        void unpack(Case &c)
        {
            static const uint8_t kTypes[13] = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15};
            uint8_t op = static_cast<uint8_t>(0x60u | kTypes[rng.below(13u)]);
            const bool v3 = ((op >> 2) & 3u) == 2u;
            const bool fill = (cycle & 0xFFu) < ((cycle >> 8) & 0xFFu);
            if (rng.pct(40) || (!e.v3w && v3) || (!e.fill && fill))
                op |= 0x10u;
            if ((!e.v3w || !e.fill) && (op & 0x10u))
                defineMask(c, op);
            uint16_t addr;
            const uint32_t r = rng.below(100u);
            if (r < 40)
                addr = static_cast<uint16_t>(0x3FFu - rng.below(32u)); // near the 1K-qword wrap
            else if (r < 50)
                addr = static_cast<uint16_t>(rng.below(16u));
            else
                addr = static_cast<uint16_t>(rng.below(0x400u));
            uint16_t imm = addr;
            if (rng.pct(50))
                imm |= 0x4000u; // usn
            if (rng.pct(35))
                imm |= 0x8000u; // flg
            if (rng.pct(10))
                imm |= static_cast<uint16_t>(rng.below(16u) << 10); // unused bits 13..10
            uint8_t num;
            const uint32_t q = rng.below(100u);
            if (q < 5)
                num = 0u;
            else if (q < 70)
                num = static_cast<uint8_t>(1u + rng.below(16u));
            else
                num = static_cast<uint8_t>(1u + rng.below(255u));
            c.cmds.push_back({op, num, imm, words(unpackWords(op, num, cycle))});
        }

        Case make()
        {
            Case c;
            c.data.resize(kMemSize);
            c.micro.resize(kMemSize);
            for (uint8_t &b : c.data)
                b = static_cast<uint8_t>(rng.u32());
            for (uint8_t &b : c.micro)
                b = static_cast<uint8_t>(rng.u32());
            vif1ref::Regs &r = c.init;
            r.stat = rng.pct(50) ? 0x80u : 0u; // DBF
            r.mark = rng.u32() & 0xFFFFu;
            r.cycle = cycle = randCycle();
            r.mode = rng.below(e.mode3 ? 4u : 3u);
            r.mask = mask = rng.u32();
            r.itops = rng.below(0x400u);
            r.base = rng.below(0x400u);
            r.ofst = rng.below(0x400u);
            r.tops = rng.below(0x400u);
            r.itop = rng.below(0x400u);
            r.top = rng.below(0x400u);
            for (uint32_t i = 0; i < 4u; ++i)
            {
                r.row[i] = rng.u32();
                r.col[i] = rng.u32();
            }

            const uint32_t count = 1u + rng.below(12u);
            for (uint32_t i = 0; i < count; ++i)
            {
                const uint32_t w = rng.below(100u);
                if (w < 45)
                {
                    unpack(c);
                    continue;
                }
                Cmd cmd{NOP, ignoredNum(), 0u, {}};
                if (w < 55)
                {
                    cmd.op = STCYCL;
                    cmd.imm = static_cast<uint16_t>(cycle = randCycle());
                }
                else if (w < 61)
                {
                    cmd.op = STMASK;
                    cmd.pool = {mask = rng.u32()};
                }
                else if (w < 65)
                    cmd = {STROW, ignoredNum(), 0u, words(4)};
                else if (w < 69)
                    cmd = {STCOL, ignoredNum(), 0u, words(4)};
                else if (w < 74)
                {
                    cmd.op = STMOD;
                    cmd.imm = junkImm(3u, static_cast<uint16_t>(rng.below(e.mode3 ? 4u : 3u)));
                }
                else if (w < 77)
                {
                    cmd.op = OFFSET;
                    cmd.imm = junkImm(0x3FFu, static_cast<uint16_t>(rng.below(0x400u)));
                }
                else if (w < 80)
                {
                    cmd.op = BASE;
                    cmd.imm = junkImm(0x3FFu, static_cast<uint16_t>(rng.below(0x400u)));
                }
                else if (w < 83)
                {
                    cmd.op = ITOP;
                    cmd.imm = junkImm(0x3FFu, static_cast<uint16_t>(rng.below(0x400u)));
                }
                else if (w < 90)
                {
                    static const uint8_t kMs[3] = {MSCAL, MSCALF, MSCNT};
                    cmd.op = kMs[rng.below(3u)];
                    if (cmd.op != MSCNT)
                        cmd.imm = rng.pct(85) ? static_cast<uint16_t>(rng.below(0x800u)) : static_cast<uint16_t>(rng.u32());
                }
                else if (w < 93)
                {
                    cmd.op = MPG;
                    const uint32_t q = rng.below(100u);
                    cmd.num = q < 5 ? 0u : static_cast<uint8_t>(1u + rng.below(q < 80 ? 32u : 255u));
                    const uint32_t a = rng.below(100u);
                    if (a < 70)
                        cmd.imm = static_cast<uint16_t>(rng.below(0x800u));
                    else if (a < 90)
                        cmd.imm = static_cast<uint16_t>(0x7FFu - rng.below(16u)); // wraps past 16 KB
                    else
                        cmd.imm = static_cast<uint16_t>(rng.u32());
                    cmd.pool = words((cmd.num ? cmd.num : 256u) * 2u);
                }
                else if (w < 95)
                {
                    cmd.op = MARK;
                    cmd.imm = static_cast<uint16_t>(rng.u32());
                }
                else if (w < 97)
                {
                    static const uint8_t kFl[3] = {FLUSHE, FLUSH, FLUSHA};
                    cmd.op = kFl[rng.below(3u)];
                }
                else if (w < 98)
                {
                    cmd.op = MSKPATH3;
                    cmd.imm = rng.pct(50) ? 0x8000u : 0u;
                }
                c.cmds.push_back(std::move(cmd));
            }
            return c;
        }
    };

    Case makeCase(uint64_t seed, uint64_t index, const Edges &edges)
    {
        Generator g{Rng{seed * 0x100000001B3ull ^ (index + 1u) * 0x9E3779B97F4A7C15ull}, edges};
        g.rng.next();
        return g.make();
    }

    // ---------------------------------------------------------------------------------------
    // Running both sides
    // ---------------------------------------------------------------------------------------
    struct Outcome
    {
        vif1ref::Regs regs{};
        std::vector<uint8_t> data, micro;
        std::vector<vif1ref::Start> starts;
        vif1ref::Result ref{}; // PCSX2 only
    };

    std::vector<vif1ref::Start> g_ourStarts;

    void runOurs(PS2Memory &mem, const Case &c, const std::vector<uint32_t> &words, Outcome &out)
    {
        VIFRegisters &v = mem.vif1_regs;
        const vif1ref::Regs &r = c.init;
        v = VIFRegisters{};
        v.stat = r.stat;
        v.err = r.err;
        v.mark = r.mark;
        v.cycle = r.cycle;
        v.mode = r.mode;
        v.num = r.num;
        v.mask = r.mask;
        v.code = r.code;
        v.itops = r.itops;
        v.base = r.base;
        v.ofst = r.ofst;
        v.tops = r.tops;
        v.itop = r.itop;
        v.top = r.top;
        std::memcpy(v.row, r.row, 16);
        std::memcpy(v.col, r.col, 16);
        std::memcpy(mem.getVU1Data(), c.data.data(), kMemSize);
        std::memcpy(mem.getVU1Code(), c.micro.data(), kMemSize);
        mem.m_path3Masked = false;
        mem.m_path3MaskedFifo.clear();
        // A desynchronized earlier case can leave a stall (payload read as an i-bit VIFcode) or a
        // DIRECT continuation behind: every case starts from an idle VIF1.
        mem.m_vif1IrqStallPending = false;
        mem.m_vif1Stalled = false;
        mem.m_vif1DmaHeld = false;
        mem.m_vif1StalledData.clear();
        mem.m_vif1PendingDirectQwc = 0u;
        mem.m_vif1PendingDirectHl = false;
        mem.m_vif1PendingPath2ImageQwc = 0u;
        mem.m_vif1PendingPath2DirectHl = false;
        g_ourStarts.clear();
        std::vector<uint32_t> buf = words; // processVIF1Data reads only [0, size)
        mem.processVIF1Data(reinterpret_cast<const uint8_t *>(buf.data()), static_cast<uint32_t>(buf.size() * 4u));
        out.regs = {v.stat, v.err, v.mark, v.cycle, v.mode, v.num, v.mask, v.code, v.itops, v.base, v.ofst,
                    v.tops, v.itop, v.top, {}, {}};
        std::memcpy(out.regs.row, v.row, 16);
        std::memcpy(out.regs.col, v.col, 16);
        out.data.assign(mem.getVU1Data(), mem.getVU1Data() + kMemSize);
        out.micro.assign(mem.getVU1Code(), mem.getVU1Code() + kMemSize);
        out.starts = g_ourStarts;
    }

    void runRef(const Case &c, const std::vector<uint32_t> &words, Outcome &out)
    {
        vif1ref::reset(c.init, c.data.data(), c.micro.data());
        std::vector<uint32_t> buf = words;
        buf.resize(words.size() + 4u, 0u); // slack read by V3 unpacks (the next element)
        out.ref = vif1ref::transfer(buf.data(), words.size());
        out.regs = vif1ref::regs();
        out.data.assign(vif1ref::vuData(), vif1ref::vuData() + kMemSize);
        out.micro.assign(vif1ref::vuMicro(), vif1ref::vuMicro() + kMemSize);
        out.starts = vif1ref::starts();
    }

    // ---------------------------------------------------------------------------------------
    // Comparison
    // ---------------------------------------------------------------------------------------
    struct Diff
    {
        std::string field;
        std::string detail;
    };

    std::set<std::string> g_ignore;

    std::vector<Diff> compare(const Outcome &ours, const Outcome &ref, bool all)
    {
        std::vector<Diff> d;
        char buf[256];
        auto add = [&](const char *field, const std::string &detail)
        {
            if (!g_ignore.count(field))
                d.push_back({field, detail});
        };
        auto reg = [&](const char *name, uint32_t o, uint32_t p)
        {
            if (o != p)
            {
                std::snprintf(buf, sizeof(buf), "%s=0x%x (PCSX2 0x%x)", name, o, p);
                add(name, buf);
            }
        };
        const vif1ref::Regs &o = ours.regs, &p = ref.regs;
        if (ref.ref.stalled || ref.ref.cmdOpen || ref.ref.wordsLeft || ref.ref.dmaStalls)
        {
            std::snprintf(buf, sizeof(buf), "PCSX2 stalled=%d open=%d left=%u dmastalls=%u", ref.ref.stalled,
                          ref.ref.cmdOpen, ref.ref.wordsLeft, ref.ref.dmaStalls);
            add("refstate", buf);
        }
        // VU1 data memory: first differing qword (all of them with `all`).
        int shown = 0;
        uint32_t qwDiffs = 0;
        for (uint32_t q = 0; q < kMemSize / 16u; ++q)
        {
            if (std::memcmp(&ours.data[q * 16u], &ref.data[q * 16u], 16) == 0)
                continue;
            ++qwDiffs;
            if (shown < (all ? 1024 : 1))
            {
                uint32_t a[4], b[4];
                std::memcpy(a, &ours.data[q * 16u], 16);
                std::memcpy(b, &ref.data[q * 16u], 16);
                std::string lanes;
                for (int l = 0; l < 4; ++l)
                    if (a[l] != b[l])
                    {
                        std::snprintf(buf, sizeof(buf), " %c=%08x(PCSX2 %08x)", "xyzw"[l], a[l], b[l]);
                        lanes += buf;
                    }
                std::snprintf(buf, sizeof(buf), "vudata qw 0x%03x:", q);
                add("vudata", buf + lanes);
                ++shown;
            }
        }
        if (qwDiffs > 1 && !all && !g_ignore.count("vudata"))
            d.back().detail += " (+" + std::to_string(qwDiffs - 1) + " more qw)";
        shown = 0;
        for (uint32_t i = 0; i < kMemSize; i += 8u)
            if (std::memcmp(&ours.micro[i], &ref.micro[i], 8) != 0 && shown++ < (all ? 64 : 1))
            {
                uint64_t a, b;
                std::memcpy(&a, &ours.micro[i], 8);
                std::memcpy(&b, &ref.micro[i], 8);
                std::snprintf(buf, sizeof(buf), "micro 0x%04x: %016llx (PCSX2 %016llx)", i,
                              static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
                add("micro", buf);
            }
        for (int i = 0; i < 4; ++i)
        {
            const std::string rn = "row" + std::to_string(i), cn = "col" + std::to_string(i);
            reg(rn.c_str(), o.row[i], p.row[i]);
            reg(cn.c_str(), o.col[i], p.col[i]);
        }
        reg("mask", o.mask, p.mask);
        reg("cycle", o.cycle & 0xFFFFu, p.cycle & 0xFFFFu);
        reg("mode", o.mode, p.mode);
        reg("num", o.num, p.num);
        reg("code", o.code, p.code);
        reg("top", o.top, p.top);
        reg("tops", o.tops & 0x3FFu, p.tops & 0x3FFu);
        reg("itop", o.itop, p.itop);
        reg("itops", o.itops, p.itops);
        reg("base", o.base, p.base);
        reg("ofst", o.ofst, p.ofst);
        reg("mark", o.mark, p.mark);
        reg("stat.dbf", (o.stat >> 7) & 1u, (p.stat >> 7) & 1u);
        reg("stat.mrk", (o.stat >> 6) & 1u, (p.stat >> 6) & 1u);
        // VU1 starts.
        const size_t n = std::max(ours.starts.size(), ref.starts.size());
        for (size_t i = 0; i < n; ++i)
        {
            auto fmt = [&](const std::vector<vif1ref::Start> &s, bool isOurs) -> std::string
            {
                if (i >= s.size())
                    return "none";
                char b[96];
                const vif1ref::Start &x = s[i];
                if (x.cont)
                    std::snprintf(b, sizeof(b), "MSCNT top=0x%x itop=0x%x", x.top, x.itop);
                else
                    std::snprintf(b, sizeof(b), "MSCAL pc=0x%x top=0x%x itop=0x%x",
                                  isOurs ? (x.pc & 0x3FFFu) : x.pc * 8u, x.top, x.itop);
                return b;
            };
            const std::string a = fmt(ours.starts, true), b = fmt(ref.starts, false);
            if (a != b)
            {
                add("vu1start", "start #" + std::to_string(i) + ": " + a + " (PCSX2 " + b + ")");
                if (!all)
                    break;
            }
        }
        return d;
    }

    std::set<std::string> fieldsOf(const std::vector<Diff> &d)
    {
        std::set<std::string> s;
        for (const Diff &x : d)
            s.insert(x.field);
        return s;
    }

    std::vector<Diff> runCase(PS2Memory &mem, const Case &c, bool all)
    {
        const std::vector<uint32_t> words = assemble(c);
        Outcome ours, ref;
        runOurs(mem, c, words, ours);
        runRef(c, words, ref);
        return compare(ours, ref, all);
    }

    // Drops commands (and then shortens UNPACK/MPG counts) while the same fields still differ.
    Case minimize(PS2Memory &mem, Case c, const std::set<std::string> &fields)
    {
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (size_t i = 0; i < c.cmds.size() && c.cmds.size() > 1u; ++i)
            {
                Case t = c;
                t.cmds.erase(t.cmds.begin() + static_cast<ptrdiff_t>(i));
                if (fieldsOf(runCase(mem, t, false)) == fields)
                {
                    c = std::move(t);
                    progress = true;
                    --i;
                }
            }
            for (size_t i = 0; i < c.cmds.size(); ++i)
            {
                Cmd &cmd = c.cmds[i];
                if (!(isUnpack(cmd.op) || cmd.op == MPG) || cmd.num == 1u)
                    continue;
                for (uint8_t num : {static_cast<uint8_t>(1u), static_cast<uint8_t>(2u), static_cast<uint8_t>(4u)})
                {
                    if (num >= (cmd.num ? cmd.num : 256u))
                        break;
                    Case t = c;
                    t.cmds[i].num = num;
                    if (t.cmds[i].op == MPG)
                        t.cmds[i].pool.resize(num * 2u);
                    if (fieldsOf(runCase(mem, t, false)) == fields)
                    {
                        c = std::move(t);
                        progress = true;
                        break;
                    }
                }
            }
        }
        return c;
    }

    void dumpCase(const Case &c)
    {
        const vif1ref::Regs &r = c.init;
        std::printf("  init: stat=0x%x cycle=cl%u/wl%u mode=%u mask=0x%08x base=0x%x ofst=0x%x tops=0x%x itops=0x%x "
                    "top=0x%x itop=0x%x mark=0x%x\n",
                    r.stat, r.cycle & 0xFFu, r.cycle >> 8, r.mode, r.mask, r.base, r.ofst, r.tops, r.itops, r.top,
                    r.itop, r.mark);
        std::printf("  init: row=%08x %08x %08x %08x col=%08x %08x %08x %08x\n", r.row[0], r.row[1], r.row[2],
                    r.row[3], r.col[0], r.col[1], r.col[2], r.col[3]);
        std::vector<size_t> starts;
        const std::vector<uint32_t> w = assemble(c, &starts);
        for (size_t i = 0; i < c.cmds.size(); ++i)
        {
            const size_t end = i + 1 < c.cmds.size() ? starts[i + 1] : w.size();
            std::printf("  [%03zx] %08x  %s", starts[i], w[starts[i]], describe(c.cmds[i]).c_str());
            if (end - starts[i] > 1u)
                std::printf("  (+%zu words)", end - starts[i] - 1u);
            std::printf("\n");
        }
    }

    void dumpWords(const Case &c)
    {
        const std::vector<uint32_t> w = assemble(c);
        std::printf("  words:");
        for (size_t i = 0; i < w.size(); ++i)
            std::printf("%s%08x", (i % 8u) ? " " : "\n    ", w[i]);
        std::printf("\n");
    }

    int usage()
    {
        std::fprintf(stderr, "usage: ps2x_vif1_pcsx2_diff [--seed S] [--cases N] [--case K] [-v] [--defined] "
                             "[--allow mode3|cyclezero|fill|v3w] [--ignore F,F..] [--minimize] [--examples N]\n");
        return 2;
    }
}

int main(int argc, char **argv)
{
    uint64_t seed = 1, cases = 20000;
    long long only = -1;
    bool verbose = false, defined = false, minimizeOn = false;
    std::set<std::string> allow;
    int examples = 20;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto val = [&]() -> const char *
        { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--seed" || a == "--cases" || a == "--case" || a == "--ignore" || a == "--examples" ||
            a == "--allow")
        {
            const char *v = val();
            if (!v)
                return usage();
            if (a == "--seed")
                seed = std::strtoull(v, nullptr, 0);
            else if (a == "--cases")
                cases = std::strtoull(v, nullptr, 0);
            else if (a == "--case")
                only = std::strtoll(v, nullptr, 0);
            else if (a == "--examples")
                examples = std::atoi(v);
            else if (a == "--allow")
            {
                const std::string e = v;
                if (e != "mode3" && e != "cyclezero" && e != "fill" && e != "v3w")
                    return usage();
                allow.insert(e);
            }
            else
            {
                std::string s = v;
                size_t pos = 0;
                while (pos <= s.size())
                {
                    const size_t e = std::min(s.find(',', pos), s.size());
                    if (e > pos)
                        g_ignore.insert(s.substr(pos, e - pos));
                    pos = e + 1;
                }
            }
        }
        else if (a == "-v")
            verbose = true;
        else if (a == "--defined")
            defined = true;
        else if (a == "--minimize")
            minimizeOn = true;
        else
            return usage();
    }

    PS2Memory mem;
    if (!mem.initialize())
    {
        std::fprintf(stderr, "PS2Memory initialize failed\n");
        return 2;
    }
    mem.setVu1MscalCallback([](uint32_t pc, uint32_t top, uint32_t itop)
                            { g_ourStarts.push_back({false, pc, top, itop}); });
    mem.setVu1MscntCallback([](uint32_t top, uint32_t itop)
                            { g_ourStarts.push_back({true, 0u, top, itop}); });

    Edges edges;
    if (defined)
        edges = {allow.count("mode3") != 0, allow.count("cyclezero") != 0, allow.count("fill") != 0,
                 allow.count("v3w") != 0};
    std::string edgeNote;
    if (defined)
    {
        edgeNote = " [--defined";
        for (const std::string &e : allow)
            edgeNote += " +" + e;
        edgeNote += "]";
    }
    const uint64_t first = only >= 0 ? static_cast<uint64_t>(only) : 0u;
    const uint64_t last = only >= 0 ? first + 1u : cases;
    uint64_t mismatches = 0;
    std::map<std::string, uint64_t> byField;
    for (uint64_t k = first; k < last; ++k)
    {
        Case c = makeCase(seed, k, edges);
        std::vector<Diff> d = runCase(mem, c, verbose);
        if (verbose)
        {
            std::printf("case %llu:\n", static_cast<unsigned long long>(k));
            dumpCase(c);
            dumpWords(c);
        }
        if (d.empty())
        {
            if (verbose)
                std::printf("  match\n");
            continue;
        }
        ++mismatches;
        const std::set<std::string> fields = fieldsOf(d);
        for (const std::string &f : fields)
            ++byField[f];
        if (verbose)
        {
            for (const Diff &x : d)
                std::printf("  %s\n", x.detail.c_str());
        }
        else if (examples < 0 || mismatches <= static_cast<uint64_t>(examples))
        {
            std::string fl;
            for (const std::string &f : fields)
                fl += (fl.empty() ? "" : ",") + f;
            std::printf("case %llu: MISMATCH [%s] %s\n", static_cast<unsigned long long>(k), fl.c_str(),
                        d[0].detail.c_str());
        }
        if (minimizeOn && (verbose || examples < 0 || mismatches <= static_cast<uint64_t>(examples)))
        {
            const Case m = minimize(mem, c, fields);
            std::printf("  minimized (%zu of %zu commands):\n", m.cmds.size(), c.cmds.size());
            dumpCase(m);
            dumpWords(m);
            for (const Diff &x : runCase(mem, m, false))
                std::printf("    -> %s\n", x.detail.c_str());
        }
    }
    std::string classes;
    for (const auto &[f, n] : byField)
        classes += (classes.empty() ? "" : ", ") + f + " " + std::to_string(n);
    std::printf("vif1 diff: %llu cases, %llu mismatches (%s)%s\n", static_cast<unsigned long long>(last - first),
                static_cast<unsigned long long>(mismatches), classes.empty() ? "none" : classes.c_str(),
                edgeNote.c_str());
    return mismatches ? 1 : 0;
}
