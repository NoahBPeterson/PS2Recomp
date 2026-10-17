// ps2x_ee_kernel_replay: the runtime's EE kernel against the real one (tools/conformance/ee_kernel).
//
// A probe (eek.py gen) is an EE ELF of straight-line syscall steps plus a step list; PCSX2 2.8.2 ran it on
// its BIOS kernel and saved v0 of every step and the programs' data areas (eek.py run). This runs the same
// steps through the runtime, the way recompiled code runs them: each thread's steps are one guest function
// that carries straight on after a syscall, as a recompiled caller does after a direct jal, and every
// step's continuation (the address after its syscall) is also registered, so a thread that blocked in a
// syscall resumes there exactly as recompiled code resumes at ctx->pc. Then v0 and the data areas are
// compared: object IDs through a bijection per kind (the kernel's are its own), everything else exact,
// words the step list marks as world-specific (stack and gp pointers) ignored.
//
// usage: ps2x_ee_kernel_replay <probe dir> [--only NAME] [-v] [--examples N] [--known FILE]
// Prints one "EK <test> cases=N match=M mismatch=K" line per test (the program name before '#') and a
// SUMMARY; exit 1 when a mismatch count differs from the --known file ("<test> <count> <reason>").

#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include "ps2_recomp_abi.h"
#include "runtime/ee_scheduler.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
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
#include <thread>
#include <vector>

namespace
{
    struct Arg
    {
        char kind = 'C'; // C constant, D data address, X code address, W data word, R result of a step
        uint32_t value = 0;
    };

    struct Step
    {
        int32_t num = 0;
        uint32_t entry = 0, cont = 0;
        bool helper = false, isId = false, isVoid = false;
        std::string idKind, text;
        std::vector<Arg> args;
        std::vector<std::pair<uint32_t, Arg>> stores;
    };

    struct Program
    {
        std::string name;
        uint32_t first = 0, data = 0, size = 0, helper = 0;
        std::map<uint32_t, int> kinds;
        std::vector<Step> steps;
    };

    struct Probe
    {
        uint32_t load = 0, mailbox = 0, results = 0, mainSp = 0, done = 0, exitEntry = 0, handler = 0;
        std::vector<Program> programs;
    };

    struct Reference
    {
        bool ok = false;
        std::string why;
        std::vector<uint32_t> v0;
        std::map<uint32_t, std::vector<uint8_t>> data;
    };

    uint32_t hexOrDec(const std::string &s) { return static_cast<uint32_t>(std::stoll(s, nullptr, 0)); }

    std::string field(const std::string &line, const std::string &key)
    {
        const size_t at = line.find(" " + key + "=");
        if (at == std::string::npos)
            return {};
        const size_t start = at + key.size() + 2;
        if (line[start] == '[')
            return line.substr(start + 1, line.find(']', start) - start - 1);
        const size_t end = line.find(' ', start);
        return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
    }

    Arg parseArg(const std::string &t)
    {
        Arg a;
        a.kind = t[0];
        a.value = hexOrDec(t.substr(2));
        return a;
    }

    bool parseSteps(const std::string &path, Probe &probe)
    {
        std::ifstream in(path);
        std::string line;
        Program *prog = nullptr;
        while (std::getline(in, line))
        {
            if (line.rfind("layout ", 0) == 0)
            {
                const std::string l = " " + line.substr(7);
                probe.load = hexOrDec(field(l, "load"));
                probe.mailbox = hexOrDec(field(l, "mailbox"));
                probe.results = hexOrDec(field(l, "results"));
                probe.mainSp = hexOrDec(field(l, "main_sp"));
                probe.done = hexOrDec(field(l, "done"));
                probe.exitEntry = hexOrDec(field(l, "exit_entry"));
                probe.handler = hexOrDec(field(l, "handler"));
            }
            else if (line.rfind("program ", 0) == 0)
            {
                probe.programs.emplace_back();
                prog = &probe.programs.back();
                std::istringstream ss(line);
                std::string w, idx;
                ss >> w >> idx >> prog->name;
                prog->first = hexOrDec(field(line, "first"));
                prog->data = hexOrDec(field(line, "data"));
                prog->size = hexOrDec(field(line, "size"));
                prog->helper = hexOrDec(field(line, "helper"));
            }
            else if (line.rfind("kind ", 0) == 0 && prog)
            {
                uint32_t off = 0;
                int kind = 0;
                std::sscanf(line.c_str(), "kind %u %d", &off, &kind);
                prog->kinds[off] = kind;
            }
            else if (line.rfind("step ", 0) == 0 && prog)
            {
                Step st;
                st.num = static_cast<int32_t>(std::stol(field(line, "num")));
                st.entry = hexOrDec(field(line, "entry"));
                st.cont = hexOrDec(field(line, "cont"));
                st.helper = field(line, "helper") == "1";
                const std::string flags = field(line, "flags");
                st.isId = flags.find("id") != std::string::npos;
                st.isVoid = flags.find("void") != std::string::npos;
                st.idKind = field(line, "idkind");
                std::istringstream args(field(line, "args"));
                for (std::string t; args >> t;)
                    st.args.push_back(parseArg(t));
                std::istringstream stores(field(line, "stores"));
                for (std::string t; stores >> t;)
                {
                    const size_t eq = t.find('=');
                    st.stores.emplace_back(hexOrDec(t.substr(0, eq)), parseArg(t.substr(eq + 1)));
                }
                const size_t tx = line.find(" text=");
                st.text = tx == std::string::npos ? "" : line.substr(tx + 6);
                prog->steps.push_back(std::move(st));
            }
        }
        return probe.load != 0 && !probe.programs.empty();
    }

    Reference parseResult(const std::string &path)
    {
        Reference r;
        std::ifstream in(path);
        std::string line;
        if (!std::getline(in, line))
        {
            r.why = "no PCSX2 result";
            return r;
        }
        if (line != "ok")
        {
            r.why = "PCSX2: " + line;
            return r;
        }
        while (std::getline(in, line))
        {
            std::istringstream ss(line);
            std::string tag;
            ss >> tag;
            if (tag == "v0")
                for (std::string w; ss >> w;)
                    r.v0.push_back(static_cast<uint32_t>(std::stoul(w, nullptr, 16)));
            else if (tag == "data")
            {
                std::string addr, hex;
                ss >> addr >> hex;
                std::vector<uint8_t> bytes(hex.size() / 2);
                for (size_t i = 0; i < bytes.size(); ++i)
                    bytes[i] = static_cast<uint8_t>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
                r.data[hexOrDec(addr)] = std::move(bytes);
            }
        }
        r.ok = true;
        return r;
    }

    std::vector<uint8_t> readFile(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    bool loadElf(const std::vector<uint8_t> &elf, uint8_t *ram)
    {
        if (elf.size() < 52 || std::memcmp(elf.data(), "\x7f" "ELF", 4) != 0)
            return false;
        auto le32 = [&](size_t o) { uint32_t v; std::memcpy(&v, elf.data() + o, 4); return v; };
        auto le16 = [&](size_t o) { uint16_t v; std::memcpy(&v, elf.data() + o, 2); return v; };
        const uint32_t phoff = le32(0x1c);
        for (uint32_t k = 0; k < le16(0x2c); ++k)
        {
            const size_t p = phoff + 32u * k;
            if (le32(p) != 1)
                continue;
            const uint32_t off = le32(p + 4), va = le32(p + 8), filesz = le32(p + 16);
            if ((va & PS2_RAM_MASK) + filesz <= PS2_RAM_SIZE && off + filesz <= elf.size())
                std::memcpy(ram + (va & PS2_RAM_MASK), elf.data() + off, filesz);
        }
        return true;
    }

    // ------------------------------------------------------------ the probe's code, as host functions
    struct Position
    {
        size_t sequence = 0, index = 0;
        bool continuation = false;
    };
    struct Ref
    {
        size_t program = 0, step = 0;
    };
    Probe *gProbe = nullptr;
    std::vector<std::vector<Ref>> gSequences; // 0: the step thread; 1..: one per helper
    std::map<uint32_t, Position> gPositions;
    std::atomic<bool> gDone{false};
    std::string gTrouble;

    uint32_t read32(uint8_t *rdram, uint32_t addr)
    {
        uint32_t v;
        std::memcpy(&v, rdram + (addr & PS2_RAM_MASK), 4);
        return v;
    }
    void write32(uint8_t *rdram, uint32_t addr, uint32_t v) { std::memcpy(rdram + (addr & PS2_RAM_MASK), &v, 4); }

    uint32_t argValue(uint8_t *rdram, const Program &p, const Arg &a)
    {
        switch (a.kind)
        {
        case 'D':
            return p.data + a.value;
        case 'W':
            return read32(rdram, p.data + a.value);
        case 'R':
            return read32(rdram, gProbe->results + 4u * (p.first + a.value));
        default: // C, X
            return a.value;
        }
    }

    void syscallInsn(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, int32_t num, uint32_t after)
    {
        SET_GPR_S32(ctx, 3, num); // addiu v1, zero, num
        ctx->pc = after;
        ps2x_abi::handleSyscall(runtime, rdram, ctx, 0u);
    }

    void probeCode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const auto pos = gPositions.find(ctx->pc);
        if (pos == gPositions.end())
        {
            gTrouble = "no probe code at pc " + std::to_string(ctx->pc);
            runtime->requestStop();
            return;
        }
        const auto &seq = gSequences[pos->second.sequence];
        size_t i = pos->second.index;
        auto finish = [&](size_t k)
        {
            const Program &p = gProbe->programs[seq[k].program];
            write32(rdram, gProbe->results + 4u * (p.first + seq[k].step), GPR_U32(ctx, 2)); // sw v0
        };
        if (pos->second.continuation)
            finish(i++);
        for (; i < seq.size(); ++i)
        {
            const Program &p = gProbe->programs[seq[i].program];
            const Step &st = p.steps[seq[i].step];
            for (const auto &[off, arg] : st.stores)
            {
                const uint32_t v = argValue(rdram, p, arg);
                SET_GPR_S32(ctx, 8, static_cast<int32_t>(v));
                write32(rdram, p.data + off, v);
            }
            static const int kRegs[8] = {4, 5, 6, 7, 8, 9, 10, 11};
            for (size_t a = 0; a < st.args.size() && a < 8; ++a)
                SET_GPR_S32(ctx, kRegs[a], static_cast<int32_t>(argValue(rdram, p, st.args[a])));
            syscallInsn(rdram, ctx, runtime, st.num, st.cont);
            finish(i);
        }
        if (pos->second.sequence == 0)
        {
            write32(rdram, gProbe->mailbox, 0x444F4E45u); // "DONE"
            gDone.store(true);
            ctx->pc = gProbe->done;
            runtime->requestStop();
            return;
        }
        syscallInsn(rdram, ctx, runtime, 0x23, 0u); // ExitThread
    }

    void exitEntry(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime) { syscallInsn(rdram, ctx, runtime, 0x23, 0u); }

    void handlerCode(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        SET_GPR_S32(ctx, 2, 0);
        ctx->pc = GPR_U32(ctx, 31);
    }

    void doneSpin(uint8_t *, R5900Context *, PS2Runtime *runtime) { runtime->requestStop(); }

    // ------------------------------------------------------------ comparison
    struct Bijection
    {
        std::map<std::string, std::map<uint32_t, uint32_t>> fwd, back;
        bool link(const std::string &kind, uint32_t a, uint32_t b)
        {
            auto &f = fwd[kind];
            auto &g = back[kind];
            const auto fi = f.find(a);
            const auto gi = g.find(b);
            if ((fi != f.end() && fi->second != b) || (gi != g.end() && gi->second != a))
                return false;
            f[a] = b;
            g[b] = a;
            return true;
        }
        bool same(uint32_t a, uint32_t b) const
        {
            if (a == b)
                return true;
            for (const auto &[kind, m] : fwd)
                if (const auto it = m.find(a); it != m.end() && it->second == b)
                    return true;
            return false;
        }
    };

    std::string compareProgram(const Program &p, const Reference &ref, uint8_t *rdram, bool verbose)
    {
        Bijection ids;
        std::string diff;
        for (size_t k = 0; k < p.steps.size(); ++k)
        {
            const Step &st = p.steps[k];
            const uint32_t a = ref.v0[p.first + k];
            const uint32_t b = read32(rdram, gProbe->results + 4u * (p.first + k));
            std::string bad;
            if (st.isVoid)
            {
            }
            else if (st.isId && static_cast<int32_t>(a) >= 0 && static_cast<int32_t>(b) >= 0) // the kernel numbers semaphores from 0
            {
                if (!ids.link(st.idKind, a, b))
                    bad = "ID mapping broken";
            }
            else if (a != b)
            {
                char buf[96];
                std::snprintf(buf, sizeof buf, "v0 kernel=%#x (%d) runtime=%#x (%d)", a, static_cast<int32_t>(a), b,
                              static_cast<int32_t>(b));
                bad = buf;
            }
            if (verbose)
                std::printf("      %zu: %s -> kernel %d, runtime %d%s\n", k, st.text.c_str(), static_cast<int32_t>(a),
                            static_cast<int32_t>(b), bad.empty() ? "" : ("  <-- " + bad).c_str());
            if (!bad.empty() && diff.empty())
                diff = "step " + std::to_string(k) + " " + st.text + ": " + bad;
        }
        const auto area = ref.data.find(p.data);
        if (area == ref.data.end())
            return diff.empty() ? "no data area in the PCSX2 result" : diff;
        const std::vector<uint8_t> &want = area->second;
        // A program with an empty data area shares its base with the next program's: compare only this
        // program's own bytes.
        const size_t own = std::min<size_t>(want.size(), p.size);
        for (uint32_t o = 0; o + 4 <= own && (diff.empty() || verbose); o += 4)
        {
            uint32_t wa, wb;
            std::memcpy(&wa, want.data() + o, 4);
            wb = read32(rdram, p.data + o);
            const auto kind = p.kinds.find(o);
            if (kind != p.kinds.end() && kind->second == 0)
                continue;
            const bool ok = (kind != p.kinds.end() && kind->second == 1) ? ids.same(wa, wb) : wa == wb;
            if (!ok)
            {
                char buf[96];
                std::snprintf(buf, sizeof buf, "data[%u] kernel=%08x runtime=%08x", o, wa, wb);
                if (verbose)
                    std::printf("      %s\n", buf);
                if (diff.empty())
                    diff = buf;
            }
        }
        return diff;
    }

    std::string testName(const std::string &program)
    {
        return program.substr(0, program.find('#'));
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <probe dir> [--only NAME] [-v] [--examples N] [--known FILE]\n", argv[0]);
        return 2;
    }
    std::string only, knownPath;
    bool verbose = false;
    int examples = 1;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--only" && i + 1 < argc)
            only = argv[++i];
        else if (a == "-v")
            verbose = true;
        else if (a == "--examples" && i + 1 < argc)
            examples = std::atoi(argv[++i]);
        else if (a == "--known" && i + 1 < argc)
            knownPath = argv[++i];
    }
    std::map<std::string, std::pair<int, std::string>> known;
    if (!knownPath.empty())
    {
        std::ifstream kf(knownPath);
        for (std::string line; std::getline(kf, line);)
        {
            if (line.empty() || line[0] == '#')
                continue;
            char name[128] = {};
            int count = 0, used = 0;
            if (std::sscanf(line.c_str(), "%127s %d %n", name, &count, &used) >= 2)
                known[name] = {count, line.substr(static_cast<size_t>(used))};
        }
    }

    namespace fs = std::filesystem;
    std::vector<std::string> probes;
    for (const auto &e : fs::directory_iterator(argv[1]))
        if (e.path().extension() == ".steps")
            probes.push_back(e.path().stem().string());
    std::sort(probes.begin(), probes.end());

    struct Tally
    {
        int cases = 0, match = 0, mismatch = 0;
        std::vector<std::string> examples;
    };
    std::map<std::string, Tally> tally;
    int probesRun = 0, probesFailed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const std::string &stem : probes)
    {
        const std::string base = (fs::path(argv[1]) / stem).string();
        Probe probe;
        if (!parseSteps(base + ".steps", probe))
        {
            std::printf("  %s: cannot read the step list\n", stem.c_str());
            ++probesFailed;
            continue;
        }
        bool wanted = only.empty();
        for (const Program &p : probe.programs)
            wanted |= testName(p.name) == only;
        if (!wanted)
            continue;
        const Reference ref = parseResult(base + ".result");
        ++probesRun;

        auto runtime = std::make_unique<PS2Runtime>();
        runtime->memory().initialize();
        uint8_t *rdram = runtime->memory().getRDRAM();
        if (!loadElf(readFile(base + ".elf"), rdram))
        {
            std::printf("  %s: cannot load the ELF\n", stem.c_str());
            ++probesFailed;
            continue;
        }
        gProbe = &probe;
        gSequences.assign(1, {});
        gPositions.clear();
        gDone = false;
        gTrouble.clear();
        std::map<uint32_t, size_t> helperSequence; // helper entry -> sequence
        for (size_t pi = 0; pi < probe.programs.size(); ++pi)
        {
            const Program &p = probe.programs[pi];
            if (p.helper)
            {
                helperSequence[p.helper] = gSequences.size();
                gSequences.emplace_back();
            }
            for (size_t k = 0; k < p.steps.size(); ++k)
            {
                const size_t s = p.steps[k].helper ? helperSequence[p.helper] : 0u;
                gSequences[s].push_back({pi, k});
            }
        }
        for (size_t s = 0; s < gSequences.size(); ++s)
            for (size_t i = 0; i < gSequences[s].size(); ++i)
            {
                const Step &st = probe.programs[gSequences[s][i].program].steps[gSequences[s][i].step];
                // a sequence is entered at its first step; any step's continuation resumes it
                if (i == 0)
                    gPositions[st.entry] = {s, i, false};
                gPositions[st.cont] = {s, i, true};
            }
        for (const auto &[addr, pos] : gPositions)
            runtime->registerFunction(addr, probeCode);
        for (const auto &[entry, s] : helperSequence)
        {
            // the helper's entry label is its first step's address
            const Step &st = probe.programs[gSequences[s][0].program].steps[gSequences[s][0].step];
            if (st.entry != entry)
                std::printf("  %s: helper entry %#x is not its first step %#x\n", stem.c_str(), entry, st.entry);
        }
        runtime->registerFunction(probe.exitEntry, exitEntry);
        runtime->registerFunction(probe.handler, handlerCode);
        runtime->registerFunction(probe.done, doneSpin);

        R5900Context ctx{};
        ctx.pc = probe.programs.front().steps.front().entry;
        SET_GPR_U32(&ctx, 29, probe.mainSp);
        EeScheduler &ee = runtime->eeScheduler();
        ee.reset(rdram, ctx);
        std::atomic<bool> finished{false};
        std::thread watchdog([&]()
        {
            for (int i = 0; i < 100 && !finished.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (!finished.load())
                ee.requestStop();
        });
        try
        {
            ee.run();
        }
        catch (const std::exception &e)
        {
            gTrouble = std::string("threw: ") + e.what();
        }
        finished = true;
        watchdog.join();

        for (const Program &p : probe.programs)
        {
            const std::string test = testName(p.name);
            if (!only.empty() && test != only)
                continue;
            Tally &t = tally[test];
            ++t.cases;
            std::string why;
            if (!ref.ok)
                why = ref.why;
            else if (ref.v0.size() < p.first + p.steps.size())
                why = "PCSX2 result is short";
            else
            {
                if (verbose)
                    std::printf("    %s:\n", p.name.c_str());
                why = compareProgram(p, ref, rdram, verbose);
                if (!why.empty() && !gDone.load())
                    why += " (the runtime did not finish the probe" + (gTrouble.empty() ? std::string() : ": " + gTrouble) + ")";
            }
            if (why.empty())
                ++t.match;
            else
            {
                ++t.mismatch;
                if (static_cast<int>(t.examples.size()) < examples)
                    t.examples.push_back(p.name + ": " + why);
            }
        }
    }

    int exports = 0, clean = 0, cases = 0, mismatches = 0, unexpected = 0;
    for (const auto &[test, t] : tally)
    {
        ++exports;
        cases += t.cases;
        mismatches += t.mismatch;
        clean += t.mismatch == 0;
        std::printf("  EK %s cases=%d match=%d mismatch=%d\n", test.c_str(), t.cases, t.match, t.mismatch);
        for (const auto &e : t.examples)
            std::printf("      %s\n", e.c_str());
        const auto k = known.find(test);
        const int expected = k == known.end() ? 0 : k->second.first;
        if (only.empty() && t.mismatch != expected)
        {
            std::printf("  MOVED %s: %d mismatches, recorded %d\n", test.c_str(), t.mismatch, expected);
            ++unexpected;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  SUMMARY tests=%d clean=%d cases=%d mismatches=%d probes=%d failed=%d seconds=%.2f\n", exports, clean,
                cases, mismatches, probesRun, probesFailed, secs);
    return unexpected || probesFailed ? 1 : 0;
}
