// Differential runner: one generated case = memory writes + a sequence of calls, run once through the
// original Sony export (LleMachine) and once through our HLE handler, then compared.
#pragma once

#include "gt_lle.h"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace gt
{
    constexpr uint32_t kArenaBase = 0x00100000u;
    constexpr uint32_t kArenaEnd = 0x0011C000u;
    constexpr uint32_t kCodeBase = 0x0011F000u;    // harness-owned guest code (callbacks), part of the baseline
    constexpr uint32_t kStackWindow = 0x8000u;     // callee frames below the initial $sp
    constexpr uint32_t kSentinelLo = 0x1FFFFF00u;  // return addresses the harness hands out
    constexpr uint32_t kAltReturn = 0x1FFFFF08u;

    struct Call
    {
        uint32_t ordinal = 0;
        std::vector<uint32_t> args;
        uint32_t spDelta = 0;        // call with $sp lowered by this much
        uint32_t ra = kSentinel;     // return address handed to the callee
        bool ignoreV0 = false;       // void function: $v0 is whatever was left in it
        // args[i] += v0 of an earlier call (each side uses its own results): {argIndex, callIndex}
        std::vector<std::pair<size_t, size_t>> refs;
    };

    struct Case
    {
        std::vector<std::pair<uint32_t, std::vector<uint8_t>>> writes;
        std::vector<Call> calls;
        std::string note; // human-readable description of the inputs
    };

    class Builder
    {
    public:
        explicit Builder(Rng &r) : rng(r) {}
        Rng &rng;
        Case c;
        uint32_t cursor = kArenaBase;

        // Reserves n bytes at a random misalignment (0..3 unless align says otherwise).
        uint32_t reserve(uint32_t n, uint32_t align = 0)
        {
            cursor = (cursor + 15u) & ~15u;
            cursor += 16u; // a guard gap so overruns land in garbage, not in the next object
            if (align == 0)
                cursor += rng.range(0, 3);
            else
                cursor = (cursor + align - 1u) & ~(align - 1u);
            const uint32_t at = cursor;
            cursor += n;
            if (cursor > kArenaEnd - 0x400)
                throw std::runtime_error("arena exhausted");
            return at;
        }
        uint32_t bytes(const std::vector<uint8_t> &data, uint32_t align = 0)
        {
            const uint32_t at = reserve(static_cast<uint32_t>(data.size()), align);
            c.writes.emplace_back(at, data);
            return at;
        }
        uint32_t str(const std::string &s, uint32_t align = 0, uint32_t slack = 0)
        {
            std::vector<uint8_t> data(s.begin(), s.end());
            data.push_back(0);
            for (uint32_t i = 0; i < slack; ++i)
                data.push_back(static_cast<uint8_t>(rng.u32()));
            return bytes(data, align);
        }
        uint32_t garbage(uint32_t n, uint32_t align = 0)
        {
            std::vector<uint8_t> data(n);
            for (auto &b : data)
                b = static_cast<uint8_t>(rng.u32());
            return bytes(data, align);
        }
        uint32_t words(const std::vector<uint32_t> &w)
        {
            std::vector<uint8_t> data(w.size() * 4);
            std::memcpy(data.data(), w.data(), data.size());
            return bytes(data, 4);
        }
        Call &call(uint32_t ordinal, std::vector<uint32_t> args)
        {
            c.calls.push_back(Call{ordinal, std::move(args)});
            return c.calls.back();
        }
        size_t lastCall() const { return c.calls.size() - 1; }
        void note(const std::string &s)
        {
            if (!c.note.empty())
                c.note += "; ";
            c.note += s;
        }
    };

    inline std::string quote(const std::string &s, size_t limit = 160)
    {
        std::ostringstream o;
        o << '"';
        size_t n = 0;
        for (unsigned char ch : s)
        {
            if (n++ >= limit)
            {
                o << "...";
                break;
            }
            if (ch == '"' || ch == '\\')
                o << '\\' << ch;
            else if (ch >= 0x20 && ch < 0x7F)
                o << ch;
            else if (ch == '\n')
                o << "\\n";
            else
                o << "\\x" << std::hex << std::setw(2) << std::setfill('0') << int(ch) << std::dec;
        }
        o << '"';
        return o.str();
    }

    inline std::string hex(uint32_t v)
    {
        char buf[16];
        std::snprintf(buf, sizeof buf, "0x%08x", v);
        return buf;
    }

    // What one side of the comparison observed for one call.
    struct CallOutcome
    {
        bool handled = true;  // HLE: dispatch returned true; LLE: always
        LleMachine::RunResult run;
        IopCpuState cpu;      // final registers
        uint32_t returnPc = 0; // where control went back to
    };
}
