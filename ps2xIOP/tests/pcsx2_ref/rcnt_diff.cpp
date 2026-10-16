// Differential test of the IOP root counters + INTC (IopDevices) against PCSX2 2.8.2's own
// IopCounters.cpp, compiled from a PCSX2 source tree (PS2X_PCSX2_SRC; never committed) with the
// stand-in headers in shim/. Random register/time/blanking sequences go to both models; every
// register read and every I_STAT value must match.
//   ps2_iop_rcnt_diff [sequences=2000] [ops=400] [seed=1]   exit 0 = all match
#include "emulator/core/iop_devices.h"

#include "IopCounters.h"
#include "IopHw.h"
#include "R3000A.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

psxRegisters psxRegs;
u32 g_refIopHw[0x1000];
extern s32 psxNextDeltaCounter;
extern u64 psxNextStartCounter;
extern bool hBlanking, vBlanking; // not reset by psxRcntInit

void psxSetNextBranch(u64 startCycle, s32 delta)
{
    if (static_cast<int>(psxRegs.iopNextEventCycle - startCycle) > delta)
        psxRegs.iopNextEventCycle = startCycle + delta;
}
void iopTestIntc() {}
void iopIntcIrq(uint irqType) { psxHu32(HW_ISTAT) |= 1u << irqType; }

namespace
{
    using ps2x::iop::detail::IopDevices;

    // PCSX2's IOP event loop, reduced to the counters: psxRcntUpdate runs whenever the clock
    // reaches psxNextStartCounter + psxNextDeltaCounter. The CPU runs at least one cycle between
    // two evaluations (a delta of 0 would otherwise re-run it forever), as IopDevices does.
    u64 g_refLastUpdate = 0;
    void refAdvanceTo(u64 cycle)
    {
        for (;;)
        {
            u64 event = psxNextStartCounter + static_cast<u64>(static_cast<s64>(psxNextDeltaCounter));
            event = std::max<u64>(event, g_refLastUpdate + 1u);
            if (event > cycle)
                break;
            psxRegs.cycle = std::max<u64>(psxRegs.cycle, event);
            psxRcntUpdate();
            g_refLastUpdate = psxRegs.cycle;
        }
        psxRegs.cycle = cycle;
    }

    uint32_t counterBase(int i) { return i < 3 ? 0x1F801100u + 0x10u * i : 0x1F801480u + 0x10u * (i - 3); }

    uint32_t refRead(int i, uint32_t off)
    {
        switch (off)
        {
        case 0x0:
            return i < 3 ? psxRcntRcount16(i) : psxRcntRcount32(i);
        case 0x2:
            return psxRcntRcount32(i) >> 16;
        case 0x4:
        {
            const uint32_t mode = psxCounters[i].mode.modeval;
            psxRcntSetNewIntrMode(i);
            return mode;
        }
        case 0x8:
            return static_cast<uint32_t>(psxCounters[i].target);
        default:
            return static_cast<uint32_t>(psxCounters[i].target >> 16);
        }
    }

    void refWrite(int i, uint32_t off, uint32_t v)
    {
        if (i < 3)
        {
            if (off == 0x0) psxRcntWcount16(i, static_cast<u16>(v));
            else if (off == 0x4) psxRcntWmode16(i, v);
            else psxRcntWtarget16(i, v);
            return;
        }
        if (off == 0x0) psxRcntWcount32(i, v);
        else if (off == 0x2) psxRcntWcount32(i, v << 16);
        else if (off == 0x4) psxRcntWmode32(i, v);
        else if (off == 0x8) psxRcntWtarget32(i, v);
        else psxRcntWtarget32(i, v << 16);
    }
}

int main(int argc, char **argv)
{
    const int sequences = argc > 1 ? std::atoi(argv[1]) : 2000;
    const int ops = argc > 2 ? std::atoi(argv[2]) : 400;
    const uint32_t seed0 = argc > 3 ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 0)) : 1u;
    uint64_t mismatches = 0, checks = 0;
    int shown = 0;

    const char *only = std::getenv("RCNT_ONLY"); // run one sequence, full history
    for (int seq = only ? std::atoi(only) : 0; seq < (only ? std::atoi(only) + 1 : sequences); ++seq)
    {
        std::mt19937 rng(seed0 * 7919u + static_cast<uint32_t>(seq));
        auto pick = [&](uint32_t n) { return static_cast<uint32_t>(rng() % n); };
        psxRegs = psxRegisters{};
        std::memset(g_refIopHw, 0, sizeof(g_refIopHw));
        psxRcntInit();
        hBlanking = vBlanking = false;
        g_refLastUpdate = 0;
        IopDevices ours;
        bool hblank = false, vblank = false;
        std::string history;

        auto check = [&](const char *what, uint32_t a, uint32_t b) {
            ++checks;
            if (a == b)
                return true;
            ++mismatches;
            if (shown++ < 12)
            {
                std::printf("MISMATCH seq %d (seed %u) at cycle %llu: %s ours=0x%x pcsx2=0x%x\n  history:%s\n", seq, seed0,
                            static_cast<unsigned long long>(ours.cycle()), what, a, b, history.c_str());
                if (only)
                {
                    std::printf("  pcsx2 nextStart=%llx delta=%d lastUpdate=%llx\n", (unsigned long long)psxNextStartCounter,
                                psxNextDeltaCounter, (unsigned long long)g_refLastUpdate);
                    for (int k = 0; k < 6; ++k)
                    {
                        const auto &o = ours.counter(k);
                        const auto &r = psxCounters[k];
                        std::printf("  c%d ours cnt=%llx tgt=%llx start=%llx rate=%u mode=%x | pcsx2 cnt=%llx tgt=%llx start=%llx rate=%u mode=%x\n",
                                    k, (unsigned long long)o.count, (unsigned long long)o.target, (unsigned long long)o.startCycle, o.rate, o.mode,
                                    (unsigned long long)r.count, (unsigned long long)r.target, (unsigned long long)r.startCycle, r.rate, r.mode.modeval);
                    }
                }
            }
            return false;
        };

        bool ok = true;
        for (int op = 0; op < ops && ok; ++op)
        {
            const int i = static_cast<int>(pick(6));
            const uint32_t kind = pick(100);
            char note[96];
            if (kind < 30)
            {
                const uint32_t scale = pick(10);
                const uint64_t step = scale < 6 ? 1 + pick(64) : scale < 9 ? 1 + pick(4096) : 1 + pick(400'000);
                const uint64_t to = ours.cycle() + step;
                ours.advanceTo(to);
                refAdvanceTo(to);
                std::snprintf(note, sizeof(note), " +%llu", static_cast<unsigned long long>(step));
            }
            else if (kind < 45)
            {
                // Modes: mostly realistic (target/overflow IRQs, repeat, toggle, prescalers, gates).
                uint32_t mode = pick(0x10000);
                if (pick(3) != 0)
                    mode &= 0x63F9u | (pick(2) ? 0x1u : 0u) | (pick(4) == 0 ? 0x6u : 0u);
                if (pick(4) == 0)
                    mode &= ~0x100u; // keep most counters on the system clock
                ours.write(counterBase(i) + 4u, i < 3 ? 2u : 4u, mode);
                refWrite(i, 0x4, mode);
                std::snprintf(note, sizeof(note), " mode%d=%x", i, mode);
            }
            else if (kind < 55)
            {
                uint32_t v = pick(3) == 0 ? pick(64) : pick(4) == 0 ? (i < 3 ? 0xFFF0u + pick(16) : 0xFFFFFFF0u + pick(16)) : static_cast<uint32_t>(rng());
                const uint32_t off = (i >= 3 && pick(6) == 0) ? 0xAu : 0x8u;
                if (i < 3)
                    v &= 0xFFFFu;
                if (off == 0xAu)
                    v &= 0xFFFFu;
                ours.write(counterBase(i) + off, i < 3 ? 2u : 4u, v);
                refWrite(i, off, v);
                std::snprintf(note, sizeof(note), " tgt%d@%x=%x", i, off, v);
            }
            else if (kind < 62)
            {
                uint32_t v = pick(2) ? pick(256) : static_cast<uint32_t>(rng());
                const uint32_t off = (i >= 3 && pick(6) == 0) ? 0x2u : 0x0u;
                if (i < 3 || off == 0x2u)
                    v &= 0xFFFFu;
                ours.write(counterBase(i) + off, i < 3 ? 2u : 4u, v);
                refWrite(i, off, v);
                std::snprintf(note, sizeof(note), " cnt%d@%x=%x", i, off, v);
            }
            else if (kind < 85)
            {
                static constexpr uint32_t kOffs16[3] = {0x0, 0x4, 0x8};
                static constexpr uint32_t kOffs32[5] = {0x0, 0x2, 0x4, 0x8, 0xA};
                const uint32_t off = i < 3 ? kOffs16[pick(3)] : kOffs32[pick(5)];
                uint32_t a = ours.read(counterBase(i) + off, i < 3 ? 2u : 4u);
                uint32_t b = refRead(i, off);
                if (i < 3 || off == 0x2u || off == 0xAu)
                {
                    a &= 0xFFFFu;
                    b &= 0xFFFFu;
                }
                std::snprintf(note, sizeof(note), " rd%d@%x", i, off);
                history += note;
                char what[48];
                std::snprintf(what, sizeof(what), "counter %d reg +%x", i, off);
                ok = check(what, a, b);
                continue;
            }
            else if (kind < 93)
            {
                if (hblank) { ours.hblankEnd(); psxHBlankEnd(); }
                else { ours.hblankStart(); psxHBlankStart(); }
                hblank = !hblank;
                std::snprintf(note, sizeof(note), hblank ? " hbS" : " hbE");
            }
            else if (kind < 97)
            {
                if (vblank) { ours.vblankEnd(); psxVBlankEnd(); }
                else { ours.vblankStart(); psxVBlankStart(); }
                vblank = !vblank;
                std::snprintf(note, sizeof(note), vblank ? " vbS" : " vbE");
            }
            else
            {
                const uint32_t ack = ~(1u << pick(32));
                ours.write(0x1F801070u, 4u, ack);
                psxHu32(HW_ISTAT) &= ack;
                std::snprintf(note, sizeof(note), " ack%x", ~ack);
            }
            history += note;
            if (!only && history.size() > 600)
                history.erase(0, history.size() - 600);
            ok = check("I_STAT", ours.iStat(), psxHu32(HW_ISTAT));
            // Internal state, so a divergence shows at the operation that caused it.
            for (int k = 0; k < 6 && ok; ++k)
            {
                const auto &o = ours.counter(k);
                const auto &r = psxCounters[k];
                char what[64];
                std::snprintf(what, sizeof(what), "counter %d count", k);
                ok = check(what, static_cast<uint32_t>(o.count), static_cast<uint32_t>(r.count));
                std::snprintf(what, sizeof(what), "counter %d target(lo)", k);
                ok = ok && check(what, static_cast<uint32_t>(o.target), static_cast<uint32_t>(r.target));
                std::snprintf(what, sizeof(what), "counter %d target(hi)", k);
                ok = ok && check(what, static_cast<uint32_t>(o.target >> 32), static_cast<uint32_t>(r.target >> 32));
                std::snprintf(what, sizeof(what), "counter %d mode", k);
                ok = ok && check(what, o.mode, r.mode.modeval);
                std::snprintf(what, sizeof(what), "counter %d rate", k);
                ok = ok && check(what, o.rate, r.rate);
                std::snprintf(what, sizeof(what), "counter %d irqmode", k);
                ok = ok && check(what, (o.repeatIrq ? 1u : 0u) | (o.toggleIrq ? 2u : 0u),
                                 (r.currentIrqMode.repeatInterrupt ? 1u : 0u) | (r.currentIrqMode.toggleInterrupt ? 2u : 0u));
                std::snprintf(what, sizeof(what), "counter %d startCycle", k);
                ok = ok && check(what, static_cast<uint32_t>(o.startCycle), static_cast<uint32_t>(r.startCycle));
            }
            if (ok)
                ok = check("next evaluation cycle", static_cast<uint32_t>(ours.nextEventCycle()),
                           static_cast<uint32_t>(std::max<u64>(psxNextStartCounter + static_cast<u64>(static_cast<s64>(psxNextDeltaCounter)), g_refLastUpdate + 1u)));
        }
    }
    std::printf("rcnt_diff: %d sequences x %d ops, %llu checks, %llu mismatches\n", sequences, ops,
                static_cast<unsigned long long>(checks), static_cast<unsigned long long>(mismatches));
    return mismatches == 0 ? 0 : 1;
}
