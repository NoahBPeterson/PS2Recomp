// Input generator for the EE oracle replay (T-0002 tier T2).
//
// MIRROR of make_input() in tools/ee_oracle/gen/src/main.rs: the vector file stores only what each
// oracle call produced, and both sides rebuild the inputs from (function address, case index). Any
// change here must be made there too, in the same draw order, and the vectors regenerated.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ee_oracle
{
    constexpr uint32_t kRamSize = 32u << 20;
    constexpr uint32_t kBufBase = 0x01800000u;
    constexpr uint32_t kBufStride = 0x1000u;
    constexpr uint32_t kBufCount = 8u; // one per argument register a0..a3, t0..t3
    constexpr uint32_t kBufWords = 256u;
    constexpr uint32_t kStackLo = 0x01ff0000u;
    constexpr uint32_t kStackTop = kRamSize - 0x4000u;
    constexpr uint64_t kDone = 0xfffffffffffffff0ull;
    constexpr uint32_t kGp = 0x00505170u; // TC2.ELF's _gp (crt0 at 0x4b2468)

    struct Rng
    {
        uint64_t s;
        uint64_t next()
        {
            s += 0x9e3779b97f4a7c15ull;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            return z ^ (z >> 31);
        }
    };

    inline uint64_t sx32(uint32_t v) { return (uint64_t)(int64_t)(int32_t)v; }

    inline uint32_t genFloat(uint32_t h)
    {
        return (h & 0x80000000u) | ((111u + ((h >> 23) & 31u)) << 23) | (h & 0x007fffffu);
    }

    // rich: spec ends in '+', pointer-rich buffers for code that walks pointer structures
    inline uint32_t genWord(Rng &rng, bool rich)
    {
        const uint64_t r = rng.next();
        const uint32_t sel = (uint32_t)(r & 0xff);
        const uint32_t h = (uint32_t)(r >> 32);
        const uint32_t f = rich ? 77u : 180u, p = rich ? 230u : 215u, i = rich ? 248u : 245u;
        if (sel < f)
            return genFloat(h);
        if (sel < p)
            return kBufBase + (h % kBufCount) * kBufStride + ((h >> 8) % 64u) * 16u;
        if (sel < i)
            return h % 64u;
        return 0u;
    }

    struct U128
    {
        uint64_t lo = 0, hi = 0;
        bool operator==(const U128 &o) const { return lo == o.lo && hi == o.hi; }
        bool operator!=(const U128 &o) const { return !(*this == o); }
    };

    struct State
    {
        std::array<U128, 32> gpr{};
        std::array<uint32_t, 32> fpr{};
        uint32_t acc = 0;
        bool cond = false;
        uint64_t hi = 0, lo = 0, hi1 = 0, lo1 = 0;
        std::array<std::array<uint32_t, 4>, 32> vf{};
        std::array<uint32_t, 4> vacc{};
        uint32_t q = 0;
    };

    struct Input
    {
        std::vector<uint32_t> words; // kBufCount * kBufWords
        State st;
    };

    inline Input makeInput(uint32_t addr, uint32_t caseIdx, const std::string &spec)
    {
        Input in;
        Rng rng{((uint64_t)addr << 32) | caseIdx};
        in.words.resize(kBufCount * kBufWords);
        const bool rich = !spec.empty() && spec.back() == '+';
        for (auto &w : in.words)
            w = genWord(rng, rich);
        State &s = in.st;
        for (int i = 1; i < 32; ++i)
        {
            s.gpr[i].lo = sx32((uint32_t)rng.next());
            s.gpr[i].hi = rng.next();
        }
        for (int k = 0; k < 8; ++k) // a0..a3 = $4..$7, a4..a7 (t0..t3) = $8..$11
        {
            const uint64_t v = rng.next();
            const char c = k < (int)spec.size() ? spec[k] : '-';
            if (c == 'p')
                s.gpr[4 + k].lo = sx32(kBufBase + (uint32_t)k * kBufStride);
            else if (c == 'i')
                s.gpr[4 + k].lo = v % 64u;
            else if (c == 'u')
                s.gpr[4 + k].lo = sx32((uint32_t)v);
            else if (c == 'v')
            {
                // a 128-bit vector of four floats: x,y from this draw, z,w from the next
                const uint64_t w = rng.next();
                s.gpr[4 + k].lo = (uint64_t)genFloat((uint32_t)(v >> 32)) << 32 | genFloat((uint32_t)v);
                s.gpr[4 + k].hi = (uint64_t)genFloat((uint32_t)(w >> 32)) << 32 | genFloat((uint32_t)w);
            }
        }
        s.gpr[28].lo = sx32(kGp);
        s.gpr[29].lo = sx32(kStackTop);
        s.gpr[31].lo = kDone;
        for (auto &f : s.fpr)
            f = genFloat((uint32_t)rng.next());
        s.acc = genFloat((uint32_t)rng.next());
        s.hi = rng.next();
        s.lo = rng.next();
        s.hi1 = rng.next();
        s.lo1 = rng.next();
        for (int i = 1; i < 32; ++i)
            for (auto &l : s.vf[i])
                l = genFloat((uint32_t)rng.next());
        s.vf[0] = {0u, 0u, 0u, 0x3f800000u};
        for (auto &l : s.vacc)
            l = genFloat((uint32_t)rng.next());
        s.q = genFloat((uint32_t)rng.next());
        return in;
    }
}
