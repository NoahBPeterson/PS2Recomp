// Stage 1 of the IOP ground-truth suite (tickets T-0024, T-0025, T-0029): the self-contained
// libraries. Each export is called with generated inputs through the original Sony code (SYSCLIB and
// STDIO from IOPRP300.IMG, run in our interpreter) and through our HLE handler; return values,
// callee-saved registers, every byte of guest RAM outside the callee's stack frame, and console
// output are compared.
#include "gt_stage1.h"
#include "gt_diff.h"

#include "emulator/imports/iop_heaplib.h"
#include "emulator/imports/iop_stdio.h"
#include "emulator/imports/iop_sysclib.h"
#include "emulator/services/iop_rpc.h"

#include <cinttypes>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace gt
{
    namespace
    {
        // ---------------------------------------------------------------- input generators
        std::string randString(Rng &rng, uint32_t maxLen, int kind = -1)
        {
            static const std::string small = "abcab";
            static const std::string printable = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
            if (kind < 0)
                kind = static_cast<int>(rng.range(0, 3));
            const uint32_t len = rng.range(0, maxLen);
            std::string s;
            for (uint32_t i = 0; i < len; ++i)
            {
                switch (kind)
                {
                case 0:
                    s += small[rng.range(0, static_cast<uint32_t>(small.size() - 1))];
                    break;
                case 1:
                    s += printable[rng.range(0, static_cast<uint32_t>(printable.size() - 1))];
                    break;
                case 2: // any non-zero byte, including 0x80..0xFF (signedness)
                    s += static_cast<char>(rng.range(1, 255));
                    break;
                default:
                    s += rng.chance(70) ? small[rng.range(0, 4)] : static_cast<char>(rng.range(0x7E, 0xFF));
                    break;
                }
            }
            return s;
        }

        uint32_t randChar(Rng &rng)
        {
            switch (rng.range(0, 9))
            {
            case 0:
                return rng.u32();
            case 1:
                return 0xFFFFFFFFu; // EOF
            case 2:
                return rng.range(0, 0x1FF);
            default:
                return rng.range(0, 255);
            }
        }

        // Arguments for the ctype-table functions. The firmware indexes its table with the (signed)
        // character, so 0x80..0xFE read unrelated, load-address dependent bytes in front of the
        // table: only EOF and 0..127 have a defined result. _toupper/_tolower truncate to a char
        // first, so random upper bits are fair game there.
        uint32_t randCtypeChar(Rng &rng, bool upperBits)
        {
            uint32_t c = rng.chance(5) ? 0xFFu : rng.range(0, 127);
            if (upperBits && rng.chance(30))
                c |= rng.u32() & 0xFFFFFF00u;
            return c;
        }

        uint32_t randInt(Rng &rng)
        {
            static const std::vector<uint32_t> edges = {0u, 1u, 0xFFFFFFFFu, 0x7FFFFFFFu, 0x80000000u, 9u, 10u, 15u, 16u, 255u, 0x8000u, 0xFFFFu};
            switch (rng.range(0, 5))
            {
            case 0:
                return rng.pick(edges);
            case 1:
                return rng.range(0, 100);
            case 2:
                return static_cast<uint32_t>(-static_cast<int32_t>(rng.range(0, 100000)));
            case 3:
                return rng.range(0, 0xFFFFF);
            default:
                return rng.u32();
            }
        }

        // A printf-family format plus the argument words it consumes. "exotic" adds flag/width/
        // precision/length combinations and conversions the IOP's formatter may not support.
        struct Format
        {
            std::string text;
            std::vector<uint32_t> args;
        };

        Format randFormat(Rng &rng, Builder &b, bool exotic)
        {
            Format f;
            const uint32_t pieces = rng.range(1, 6);
            for (uint32_t p = 0; p < pieces; ++p)
            {
                if (rng.chance(40))
                {
                    std::string lit = randString(rng, 6, 1);
                    std::string clean;
                    for (char ch : lit)
                        if (ch != '%')
                            clean += ch;
                    f.text += clean;
                    continue;
                }
                std::string spec = "%";
                if (exotic || rng.chance(30))
                {
                    const char *flags = "-0 +#";
                    const uint32_t n = rng.range(0, exotic ? 3 : 1);
                    for (uint32_t i = 0; i < n; ++i)
                        spec += flags[rng.range(0, 4)];
                }
                if (rng.chance(exotic ? 50 : 40))
                {
                    if (exotic && rng.chance(25))
                    {
                        spec += '*';
                        f.args.push_back(static_cast<uint32_t>(static_cast<int32_t>(rng.range(0, 30)) - 10));
                    }
                    else
                        spec += std::to_string(rng.range(1, exotic ? 24 : 12));
                }
                if (rng.chance(exotic ? 40 : 15))
                {
                    spec += '.';
                    if (exotic && rng.chance(25))
                    {
                        spec += '*';
                        f.args.push_back(static_cast<uint32_t>(static_cast<int32_t>(rng.range(0, 20)) - 5));
                    }
                    else if (!rng.chance(15))
                        spec += std::to_string(rng.range(0, exotic ? 20 : 8));
                }
                bool literalConversion = false; // the firmware ends the conversion at q/z and prints it
                if (exotic && rng.chance(35))
                {
                    static const std::vector<std::string> lens = {"l", "h", "hh", "ll", "L", "q", "z"};
                    const std::string len = rng.pick(lens);
                    spec += len;
                    literalConversion = len == "q" || len == "z";
                }
                else if (rng.chance(10))
                    spec += 'l';
                static const std::string common = "dduuxxXsscc%i";
                static const std::string rare = "dioxXucsp%nfegEGbBkDOUy";
                char conv = exotic && rng.chance(40) ? rare[rng.range(0, static_cast<uint32_t>(rare.size() - 1))]
                                                     : common[rng.range(0, static_cast<uint32_t>(common.size() - 1))];
                if (literalConversion && conv == '%')
                    conv = 'k'; // "%q%" would pair the printed '%' with the next piece's '%'
                spec += conv;
                f.text += spec;
                // Arguments exactly as the firmware's prnt consumes them (unknown conversions print
                // the character and take none), so '*' always reads the intended small value.
                switch (literalConversion ? '%' : conv)
                {
                case 's':
                    if (rng.chance(5))
                        f.args.push_back(0u);
                    else
                        f.args.push_back(b.str(randString(rng, 16, rng.chance(80) ? 1 : 2)));
                    break;
                case 'c':
                    f.args.push_back(randChar(rng));
                    break;
                case 'n':
                    f.args.push_back(b.garbage(4, 4));
                    break;
                case 'd': case 'i': case 'D': case 'o': case 'O': case 'p': case 'u': case 'U': case 'x': case 'X':
                    f.args.push_back(randInt(rng));
                    break;
                default:
                    break;
                }
            }
            // Spare argument words: a formatter that consumes more than the format implies
            // reads defined values (identical on both sides).
            for (int i = 0; i < 4; ++i)
                f.args.push_back(randInt(rng));
            return f;
        }

        // Guest callback for prnt: ctx points at a cursor word; store the character and advance.
        const std::vector<uint32_t> kPutcCallback = {
            0x8C880000u, // lw    t0, 0(a0)
            0x00000000u, // nop
            0xA1050000u, // sb    a1, 0(t0)
            0x25080001u, // addiu t0, t0, 1
            0xAC880000u, // sw    t0, 0(a0)
            0x03E00008u, // jr    ra
            0x00000000u, // nop
        };

        std::string randNumberString(Rng &rng)
        {
            std::string s;
            const uint32_t ws = rng.chance(30) ? rng.range(1, 3) : 0;
            for (uint32_t i = 0; i < ws; ++i)
                s += " \t\n\v\f\r"[rng.range(0, 5)];
            if (rng.chance(40))
                s += rng.chance(50) ? '-' : '+';
            if (rng.chance(20))
                s += rng.chance(50) ? "0x" : "0X";
            else if (rng.chance(15))
                s += '0';
            static const std::string digits = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFXYZ";
            const uint32_t nd = rng.chance(10) ? rng.range(10, 24) : rng.range(0, 9);
            for (uint32_t i = 0; i < nd; ++i)
                s += digits[rng.chance(60) ? rng.range(0, 9) : rng.range(0, static_cast<uint32_t>(digits.size() - 1))];
            if (rng.chance(30))
                s += randString(rng, 4, 1);
            return s;
        }

        uint32_t randBase(Rng &rng)
        {
            static const std::vector<uint32_t> bases = {0, 0, 10, 10, 16, 16, 8, 2, 36, 1, 37, 0xFFFFFFFFu};
            return rng.chance(85) ? rng.pick(bases) : rng.range(2, 36);
        }

        // ---------------------------------------------------------------- test table
        using Gen = void (*)(Builder &);

        struct ExportTest
        {
            const char *library;
            uint16_t ordinal;
            const char *name;
            Gen gen;
            uint32_t flags = 0;
        };
        enum : uint32_t
        {
            kCompareCtypeTable = 1, // v0 is a table pointer: compare the bytes it points at
            kCompareJumpRegs = 2,   // longjmp: compare the restored registers and return pc
        };

        void genStringPair(Builder &b, uint32_t ord, bool bounded)
        {
            Rng &r = b.rng;
            const int kind = static_cast<int>(r.range(0, 3));
            std::string s1 = randString(r, 24, kind);
            std::string s2 = r.chance(40) ? s1 : randString(r, 24, kind);
            if (r.chance(30) && !s1.empty())
                s2 = s1.substr(0, r.range(0, static_cast<uint32_t>(s1.size())));
            if (r.chance(20))
                s2 += static_cast<char>(r.range(1, 255));
            const uint32_t a = b.str(s1, 0, 8);
            const uint32_t c2 = b.str(s2, 0, 8);
            std::vector<uint32_t> args = {a, c2};
            if (bounded)
            {
                const uint32_t n = r.chance(10) ? r.u32() : r.range(0, 30);
                args.push_back(n);
                b.note("n=" + std::to_string(n));
            }
            b.call(ord, args);
            b.note("s1=" + quote(s1) + " s2=" + quote(s2));
        }

        const std::vector<ExportTest> &sysclibTests()
        {
            static const std::vector<ExportTest> tests = {
                {"sysclib", 4, "setjmp", [](Builder &b)
                 {
                     const uint32_t buf = b.garbage(64, 4);
                     b.call(4, {buf});
                 }},
                {"sysclib", 5, "longjmp", [](Builder &b)
                 {
                     const uint32_t buf = b.garbage(64, 4);
                     b.call(4, {buf});
                     const uint32_t val = b.rng.chance(20) ? 0u : randInt(b.rng);
                     Call &jump = b.call(5, {buf, val});
                     jump.spDelta = 16u * b.rng.range(1, 64);
                     jump.ra = kAltReturn;
                     b.note("val=" + hex(val));
                 },
                 kCompareJumpRegs},
                {"sysclib", 6, "_toupper", [](Builder &b)
                 { const uint32_t c = randCtypeChar(b.rng, true); b.call(6, {c}); b.note("c=" + hex(c)); }},
                {"sysclib", 7, "_tolower", [](Builder &b)
                 { const uint32_t c = randCtypeChar(b.rng, true); b.call(7, {c}); b.note("c=" + hex(c)); }},
                {"sysclib", 8, "look_ctype_table", [](Builder &b)
                 {
                     uint32_t c = randCtypeChar(b.rng, false);
                     if (c == 0xFFu)
                         c = 0xFFFFFFFFu;
                     b.call(8, {c});
                     b.note("c=" + hex(c));
                 }},
                {"sysclib", 9, "get_ctype_table", [](Builder &b)
                 { b.call(9, {}); },
                 kCompareCtypeTable},
                {"sysclib", 10, "memchr", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.range(0, 64);
                     const uint32_t buf = b.garbage(n + 8);
                     const uint32_t c = r.chance(50) ? r.u32() : r.range(0, 255);
                     const uint32_t len = r.chance(10) ? 0 : r.range(0, n);
                     b.call(10, {buf, c, len});
                     b.note("c=" + hex(c) + " n=" + std::to_string(len));
                 }},
                {"sysclib", 11, "memcmp", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.range(0, 48);
                     std::vector<uint8_t> x(n + 4), y;
                     for (auto &v : x)
                         v = static_cast<uint8_t>(r.chance(50) ? r.range(0x7E, 0x82) : r.u32());
                     y = x;
                     if (r.chance(70) && n)
                         y[r.range(0, n - 1)] = static_cast<uint8_t>(r.u32());
                     b.call(11, {b.bytes(x), b.bytes(y), r.chance(5) ? 0u : n});
                     b.note("n=" + std::to_string(n));
                 }},
                {"sysclib", 12, "memcpy", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.chance(10) ? r.range(200, 600) : r.range(0, 70);
                     const uint32_t src = b.garbage(n + 8);
                     const uint32_t dst = b.garbage(n + 8);
                     b.call(12, {dst, src, n});
                     b.note("n=" + std::to_string(n) + " dst&3=" + std::to_string(dst & 3) + " src&3=" + std::to_string(src & 3));
                 }},
                {"sysclib", 13, "memmove", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.range(0, 90);
                     const uint32_t buf = b.garbage(n * 2 + 16);
                     const uint32_t src = buf + r.range(0, n);
                     const uint32_t dst = buf + r.range(0, n);
                     b.call(13, {dst, src, n});
                     b.note("n=" + std::to_string(n) + " dst-src=" + std::to_string(int(dst - src)));
                 }},
                {"sysclib", 14, "memset", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.chance(10) ? r.range(100, 400) : r.range(0, 70);
                     const uint32_t c = r.chance(50) ? r.u32() : r.range(0, 255);
                     b.call(14, {b.garbage(n + 8), c, n});
                     b.note("c=" + hex(c) + " n=" + std::to_string(n));
                 }},
                {"sysclib", 15, "bcmp", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.range(0, 48);
                     std::vector<uint8_t> x(n + 4), y;
                     for (auto &v : x)
                         v = static_cast<uint8_t>(r.u32());
                     y = x;
                     if (r.chance(70) && n)
                         y[r.range(0, n - 1)] = static_cast<uint8_t>(r.u32());
                     b.call(15, {b.bytes(x), b.bytes(y), n});
                     b.note("n=" + std::to_string(n));
                 }},
                {"sysclib", 16, "bcopy", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.range(0, 90);
                     const uint32_t buf = b.garbage(n * 2 + 16);
                     const uint32_t src = buf + r.range(0, n);
                     const uint32_t dst = buf + r.range(0, n);
                     b.call(16, {src, dst, n});
                     b.note("n=" + std::to_string(n) + " dst-src=" + std::to_string(int(dst - src)));
                 }},
                {"sysclib", 17, "bzero", [](Builder &b)
                 {
                     const uint32_t n = b.rng.range(0, 90);
                     b.call(17, {b.garbage(n + 8), n});
                     b.note("n=" + std::to_string(n));
                 }},
                {"sysclib", 18, "prnt", [](Builder &b)
                 {
                     const Format f = randFormat(b.rng, b, false);
                     const uint32_t out = b.garbage(512);
                     const uint32_t ctx = b.words({out});
                     const uint32_t ap = b.words(f.args);
                     b.call(18, {kCodeBase, ctx, b.str(f.text), ap});
                     b.note("fmt=" + quote(f.text));
                 }},
                {"sysclib", 19, "sprintf", [](Builder &b)
                 {
                     const Format f = randFormat(b.rng, b, false);
                     std::vector<uint32_t> args = {b.garbage(512), b.str(f.text)};
                     args.insert(args.end(), f.args.begin(), f.args.end());
                     b.call(19, args);
                     b.note("fmt=" + quote(f.text));
                 }},
                {"sysclib", 19, "sprintf[exotic]", [](Builder &b)
                 {
                     const Format f = randFormat(b.rng, b, true);
                     std::vector<uint32_t> args = {b.garbage(512), b.str(f.text)};
                     args.insert(args.end(), f.args.begin(), f.args.end());
                     b.call(19, args);
                     b.note("fmt=" + quote(f.text));
                 }},
                {"sysclib", 20, "strcat", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string d = randString(r, 20), s = randString(r, 20);
                     b.call(20, {b.str(d, 0, 48), b.str(s)});
                     b.note("dst=" + quote(d) + " src=" + quote(s));
                 }},
                {"sysclib", 21, "strchr", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 24);
                     const uint32_t c = r.chance(60) && !s.empty() ? static_cast<uint8_t>(s[r.range(0, static_cast<uint32_t>(s.size() - 1))]) | (r.chance(30) ? 0xFFFFFF00u : 0u) : randChar(r);
                     b.call(21, {b.str(s, 0, 4), r.chance(5) ? 0u : c});
                     b.note("s=" + quote(s) + " c=" + hex(c));
                 }},
                {"sysclib", 22, "strcmp", [](Builder &b)
                 { genStringPair(b, 22, false); }},
                {"sysclib", 23, "strcpy", [](Builder &b)
                 {
                     const std::string s = randString(b.rng, 40);
                     b.call(23, {b.garbage(48), b.str(s)});
                     b.note("src=" + quote(s));
                 }},
                {"sysclib", 24, "strcspn", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 24), set = randString(r, 5);
                     b.call(24, {b.str(s), b.str(set)});
                     b.note("s=" + quote(s) + " set=" + quote(set));
                 }},
                {"sysclib", 25, "index", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 24);
                     const uint32_t c = r.chance(60) && !s.empty() ? static_cast<uint8_t>(s[r.range(0, static_cast<uint32_t>(s.size() - 1))]) : randChar(r);
                     b.call(25, {b.str(s, 0, 4), r.chance(5) ? 0u : c});
                     b.note("s=" + quote(s) + " c=" + hex(c));
                 }},
                {"sysclib", 26, "rindex", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 24);
                     const uint32_t c = r.chance(60) && !s.empty() ? static_cast<uint8_t>(s[r.range(0, static_cast<uint32_t>(s.size() - 1))]) : randChar(r);
                     b.call(26, {b.str(s, 0, 4), r.chance(5) ? 0u : c});
                     b.note("s=" + quote(s) + " c=" + hex(c));
                 }},
                {"sysclib", 27, "strlen", [](Builder &b)
                 {
                     const std::string s = randString(b.rng, 64);
                     b.call(27, {b.str(s)});
                     b.note("s=" + quote(s));
                 }},
                {"sysclib", 28, "strncat", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string d = randString(r, 20), s = randString(r, 20);
                     const uint32_t n = r.chance(10) ? r.u32() : r.range(0, 25);
                     b.call(28, {b.str(d, 0, 48), b.str(s), n});
                     b.note("dst=" + quote(d) + " src=" + quote(s) + " n=" + std::to_string(n));
                 }},
                {"sysclib", 29, "strncmp", [](Builder &b)
                 { genStringPair(b, 29, true); }},
                {"sysclib", 30, "strncpy", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 30);
                     const uint32_t n = r.range(0, 40);
                     b.call(30, {b.garbage(48), b.str(s), n});
                     b.note("src=" + quote(s) + " n=" + std::to_string(n));
                 }},
                {"sysclib", 31, "strpbrk", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 24), set = randString(r, 5);
                     b.call(31, {b.str(s), b.str(set)});
                     b.note("s=" + quote(s) + " set=" + quote(set));
                 }},
                {"sysclib", 32, "strrchr", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 24);
                     const uint32_t c = r.chance(60) && !s.empty() ? static_cast<uint8_t>(s[r.range(0, static_cast<uint32_t>(s.size() - 1))]) | (r.chance(30) ? 0xFFFFFF00u : 0u) : randChar(r);
                     b.call(32, {b.str(s, 0, 4), r.chance(5) ? 0u : c});
                     b.note("s=" + quote(s) + " c=" + hex(c));
                 }},
                {"sysclib", 33, "strspn", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 24), set = randString(r, 5);
                     b.call(33, {b.str(s), b.str(set)});
                     b.note("s=" + quote(s) + " set=" + quote(set));
                 }},
                {"sysclib", 34, "strstr", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string h = randString(r, 30, 0);
                     std::string n = randString(r, 4, 0);
                     if (r.chance(50) && !h.empty())
                     {
                         const uint32_t at = r.range(0, static_cast<uint32_t>(h.size() - 1));
                         n = h.substr(at, r.range(0, static_cast<uint32_t>(h.size() - at)));
                     }
                     b.call(34, {b.str(h), b.str(n)});
                     b.note("hay=" + quote(h) + " needle=" + quote(n));
                 }},
                {"sysclib", 35, "strtok", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 30, 0) + randString(r, 6, 1);
                     const std::string delim = randString(r, 3, 0);
                     const uint32_t ds = b.str(delim);
                     b.call(35, {b.str(s), ds});
                     const uint32_t more = r.range(1, 8);
                     for (uint32_t i = 0; i < more; ++i)
                         b.call(35, {0u, r.chance(80) ? ds : b.str(randString(r, 3, 0))});
                     b.note("s=" + quote(s) + " delim=" + quote(delim) + " calls=" + std::to_string(more + 1));
                 }},
                {"sysclib", 36, "strtol", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randNumberString(r);
                     const uint32_t base = randBase(r);
                     b.call(36, {b.str(s), r.chance(10) ? 0u : b.garbage(4, 4), base});
                     b.note("s=" + quote(s) + " base=" + std::to_string(int32_t(base)));
                 }},
                {"sysclib", 37, "atob", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randNumberString(r);
                     b.call(37, {b.str(s), b.garbage(4, 4)});
                     b.note("s=" + quote(s));
                 }},
                {"sysclib", 38, "strtoul", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randNumberString(r);
                     const uint32_t base = randBase(r);
                     b.call(38, {b.str(s), r.chance(10) ? 0u : b.garbage(4, 4), base});
                     b.note("s=" + quote(s) + " base=" + std::to_string(int32_t(base)));
                 }},
                {"sysclib", 39, "_retonly", [](Builder &b)
                 { b.call(39, {b.rng.u32(), b.rng.u32()}); }},
                {"sysclib", 44, "_retnegativeone[44-47]", [](Builder &b)
                 { b.call(44 + b.rng.range(0, 3), {b.rng.u32()}); }},
                {"sysclib", 40, "_wmemcopy", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.range(0, 40) * (r.chance(80) ? 4u : 1u);
                     b.call(40, {b.garbage(n * 4 + 16, 4), b.garbage(n * 4 + 16, 4), n});
                     b.note("n=" + std::to_string(n));
                 }},
                {"sysclib", 41, "_wmemset", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const uint32_t n = r.range(0, 40) * (r.chance(80) ? 4u : 1u);
                     const uint32_t c = r.u32();
                     b.call(41, {b.garbage(n * 4 + 16, 4), c, n});
                     b.note("c=" + hex(c) + " n=" + std::to_string(n));
                 }},
                {"sysclib", 42, "vsprintf", [](Builder &b)
                 {
                     const Format f = randFormat(b.rng, b, false);
                     b.call(42, {b.garbage(512), b.str(f.text), b.words(f.args)});
                     b.note("fmt=" + quote(f.text));
                 }},
                {"sysclib", 43, "strtok_r", [](Builder &b)
                 {
                     Rng &r = b.rng;
                     const std::string s = randString(r, 30, 0) + randString(r, 6, 1);
                     const std::string delim = randString(r, 3, 0);
                     const uint32_t ds = b.str(delim);
                     const uint32_t save = b.garbage(4, 4);
                     b.call(43, {b.str(s), ds, save});
                     const uint32_t more = r.range(1, 8);
                     for (uint32_t i = 0; i < more; ++i)
                         b.call(43, {0u, ds, save});
                     b.note("s=" + quote(s) + " delim=" + quote(delim) + " calls=" + std::to_string(more + 1));
                 }},
            };
            return tests;
        }

        const std::vector<ExportTest> &stdioTests()
        {
            static const std::vector<ExportTest> tests = {
                {"stdio", 4, "printf", [](Builder &b)
                 {
                     const Format f = randFormat(b.rng, b, false);
                     std::vector<uint32_t> args = {b.str(f.text + (b.rng.chance(50) ? "\n" : ""))};
                     args.insert(args.end(), f.args.begin(), f.args.end());
                     b.call(4, args);
                     b.note("fmt=" + quote(f.text));
                 }},
                {"stdio", 5, "getchar", [](Builder &b)
                 { b.call(5, {}); }},
                {"stdio", 6, "putchar", [](Builder &b)
                 { const uint32_t c = randChar(b.rng); b.call(6, {c}); b.note("c=" + hex(c)); }},
                {"stdio", 7, "puts", [](Builder &b)
                 { const std::string s = randString(b.rng, 40, 1); b.call(7, {b.str(s)}); b.note("s=" + quote(s)); }},
                {"stdio", 8, "gets", [](Builder &b)
                 { b.call(8, {b.garbage(64)}); }},
                {"stdio", 9, "fdprintf", [](Builder &b)
                 {
                     const Format f = randFormat(b.rng, b, false);
                     std::vector<uint32_t> args = {1u, b.str(f.text + "\n")};
                     args.insert(args.end(), f.args.begin(), f.args.end());
                     b.call(9, args);
                     b.note("fmt=" + quote(f.text));
                 }},
                {"stdio", 10, "fdgetc", [](Builder &b)
                 { b.call(10, {0u}); }},
                {"stdio", 11, "fdputc", [](Builder &b)
                 { const uint32_t c = randChar(b.rng); b.call(11, {c, 1u}); b.note("c=" + hex(c)); }},
                {"stdio", 12, "fdputs", [](Builder &b)
                 { const std::string s = randString(b.rng, 40, 1); b.call(12, {b.str(s), 1u}); b.note("s=" + quote(s)); }},
                {"stdio", 13, "fdgets", [](Builder &b)
                 { b.call(13, {b.garbage(64), 0u}); }},
                {"stdio", 14, "vfdprintf", [](Builder &b)
                 {
                     const Format f = randFormat(b.rng, b, false);
                     b.call(14, {1u, b.str(f.text + "\n"), b.words(f.args)});
                     b.note("fmt=" + quote(f.text));
                 }},
            };
            return tests;
        }

        // ---------------------------------------------------------------- HEAPLIB (BIOS)
        uint32_t randHeapSize(Rng &r)
        {
            static const std::vector<uint32_t> sizes = {0u, 8u, 40u, 41u, 56u, 57u, 64u, 100u, 256u, 1000u, 4096u, 0xFFFFFFF0u};
            return r.chance(40) ? r.pick(sizes) : r.range(0, 6000);
        }
        uint32_t randAllocSize(Rng &r)
        {
            static const std::vector<uint32_t> sizes = {0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u, 24u, 0x7FFFFFFFu, 0xFFFFFFFFu};
            return r.chance(30) ? r.pick(sizes) : r.chance(70) ? r.range(0, 200) : r.range(0, 5000);
        }

        // A heap session: CreateHeap, a random mix of the other heap calls weighted towards
        // `focus`, DeleteHeap. Pointers flow from earlier results (each side uses its own).
        void heapSession(Builder &b, uint32_t focus)
        {
            Rng &r = b.rng;
            const uint32_t size = randHeapSize(r), flag = r.range(0, 3);
            b.call(4, {size, flag});
            const size_t heap = b.lastCall();
            b.note("CreateHeap(" + std::to_string(int32_t(size)) + ", " + std::to_string(flag) + ")");
            std::vector<size_t> allocs;
            const uint32_t ops = r.range(1, focus == 4 ? 4 : 24);
            for (uint32_t i = 0; i < ops; ++i)
            {
                uint32_t op = r.chance(50) ? focus : r.pick(std::vector<uint32_t>{6, 6, 7, 8, 15});
                if (op == 4 || op == 5)
                    op = 6;
                if (op == 6)
                {
                    const uint32_t n = randAllocSize(r);
                    Call &c = b.call(6, {0u, n});
                    c.refs = {{0, heap}};
                    allocs.push_back(b.lastCall());
                }
                else if (op == 7)
                {
                    if (!allocs.empty() && r.chance(80))
                    {
                        Call &c = b.call(7, {0u, r.chance(90) ? 0u : 4u * r.range(1, 3)}); // word-aligned: the firmware faults otherwise
                        c.refs = {{0, heap}, {1, allocs[r.range(0, static_cast<uint32_t>(allocs.size() - 1))]}};
                    }
                    else
                    {
                        const uint32_t bogus = r.chance(30) ? 0u : 4u * r.range(0, 100);
                        Call &c = b.call(7, {0u, bogus});
                        c.refs = {{0, heap}};
                        if (bogus != 0u)
                            c.refs.push_back({1, heap});
                    }
                }
                else if (op == 8)
                {
                    Call &c = b.call(8, {0u});
                    c.refs = {{0, heap}};
                }
                else // 15: HeapChunkSize of the first chunk
                {
                    Call &c = b.call(15, {16u});
                    c.refs = {{0, heap}};
                }
            }
            if (r.chance(10))
            {
                Call &c = b.call(8, {0u}); // a freed heap: validation fails
                c.refs = {{0, heap}};
            }
            Call &del = b.call(5, {0u});
            del.refs = {{0, heap}};
            del.ignoreV0 = true;
            b.note(std::to_string(b.c.calls.size()) + " calls");
        }

        // The chunk primitives on a caller-provided buffer: HeapPrepare, then allocate/free.
        void chunkSession(Builder &b, uint32_t focus)
        {
            Rng &r = b.rng;
            const uint32_t size = r.chance(15) ? r.range(0, 0x30) : r.range(0x29, 2048);
            const uint32_t buf = b.garbage(std::min<uint32_t>(size, 2048) + 16, r.chance(80) ? 8 : 4);
            Call &prep = b.call(11, {buf, size});
            prep.ignoreV0 = true;
            b.note("HeapPrepare(buf, " + std::to_string(size) + ")");
            std::vector<size_t> allocs;
            const uint32_t ops = r.range(1, 20);
            for (uint32_t i = 0; i < ops; ++i)
            {
                uint32_t op = r.chance(50) ? focus : r.pick(std::vector<uint32_t>{13, 13, 14, 12, 15});
                if (op == 11)
                    op = 13;
                if (op == 13)
                {
                    b.call(13, {buf, randAllocSize(r)});
                    allocs.push_back(b.lastCall());
                }
                else if (op == 14)
                {
                    if (!allocs.empty() && r.chance(85))
                    {
                        Call &c = b.call(14, {buf, 0u});
                        c.refs = {{1, allocs[r.range(0, static_cast<uint32_t>(allocs.size() - 1))]}};
                    }
                    else
                        b.call(14, {buf, r.chance(30) ? 0u : ((buf + r.range(0, 64)) & ~3u)});
                }
                else
                    b.call(op, {buf});
            }
            b.note(std::to_string(b.c.calls.size()) + " calls");
        }

        const std::vector<ExportTest> &heaplibTests()
        {
            static const std::vector<ExportTest> tests = {
                {"heaplib", 4, "CreateHeap", [](Builder &b)
                 { heapSession(b, 4); }},
                {"heaplib", 5, "DeleteHeap", [](Builder &b)
                 { heapSession(b, 6); }},
                {"heaplib", 6, "AllocHeapMemory", [](Builder &b)
                 { heapSession(b, 6); }},
                {"heaplib", 7, "FreeHeapMemory", [](Builder &b)
                 { heapSession(b, 7); }},
                {"heaplib", 8, "HeapTotalFreeSize", [](Builder &b)
                 { heapSession(b, 8); }},
                {"heaplib", 11, "HeapPrepare", [](Builder &b)
                 { chunkSession(b, 11); }},
                {"heaplib", 12, "heaplib_12_chunk_is_valid", [](Builder &b)
                 { chunkSession(b, 12); }},
                {"heaplib", 13, "heaplib_13_chunk_do_allocate", [](Builder &b)
                 { chunkSession(b, 13); }},
                {"heaplib", 14, "heaplib_14_chunk_do_iterate", [](Builder &b)
                 { chunkSession(b, 14); }},
                {"heaplib", 15, "HeapChunkSize", [](Builder &b)
                 { heapSession(b, 15); }},
                {"heaplib", 9, "_retonly[9,10,16,17]", [](Builder &b)
                 {
                     Call &c = b.call(b.rng.pick(std::vector<uint32_t>{9, 10, 16, 17}), {b.rng.u32()});
                     c.ignoreV0 = false; // _retonly leaves v0 alone; so does the HLE
                 }},
            };
            return tests;
        }

        // ---------------------------------------------------------------- the two worlds
        // Runs guest callbacks the HLE invokes (prnt's per-character callback) on the HLE memory.
        class HleCallbacks final : public IopGuestExecutor
        {
        public:
            explicit HleCallbacks(IopMemory &memory) : m_memory(memory) {}
            uint32_t executeGuestFunction(uint32_t address, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t gp) override
            {
                IopCpuCore core(m_memory);
                IopCpuState cpu = LleMachine::initialCpu(address, gp, LleMachine::kStackTop - 0x4000u);
                cpu.gpr[4] = a0;
                cpu.gpr[5] = a1;
                cpu.gpr[6] = a2;
                cpu.gpr[7] = a3;
                for (uint32_t n = 0; n < 100000u; ++n)
                {
                    if ((cpu.pc & ~0xFu) == (kSentinel & ~0xFu) && !cpu.branchPending)
                        break;
                    (void)core.executeInstruction(cpu);
                    if (cpu.exception)
                        break;
                }
                LleMachine::flushPendingLoad(cpu);
                return cpu.gpr[2];
            }

        private:
            IopMemory &m_memory;
        };

        struct World
        {
            LleMachine lle;
            IopMemory hleMem;
            CaptureHost host;
            IopSysclib sysclib{hleMem};
            IopStdio stdio{host, hleMem};
            IopHeaplib heaplib{hleMem};
            HleCallbacks callbacks{hleMem};
            std::vector<uint8_t> baseline;
            std::string lleOut, hleOut; // console bytes
            uint32_t moduleEnd = 0;
            // Heap world: both sides allocate sysmem blocks with the same allocator, so the heap
            // region is compared too, and both memories (allocator state included) start every
            // case from a snapshot.
            bool heapWorld = false;
            IopMemory lleSnapshot, hleSnapshot;

            bool hleDispatch(std::string_view lib, uint16_t ord, IopCpuState &cpu)
            {
                if (lib == "sysclib")
                    return sysclib.dispatchImport(ord, cpu, &callbacks);
                if (lib == "stdio")
                    return stdio.dispatchImport(ord, cpu);
                if (lib == "heaplib")
                    return heaplib.dispatchImport(ord, cpu);
                return false;
            }

            void writeBoth(uint32_t at, const void *p, size_t n)
            {
                (void)lle.memory.writeRam(at, p, n);
                (void)hleMem.writeRam(at, p, n);
            }

            void restoreRange(uint32_t lo, uint32_t hi)
            {
                (void)lle.memory.writeRam(lo, baseline.data() + lo, hi - lo);
                (void)hleMem.writeRam(lo, baseline.data() + lo, hi - lo);
            }
        };

        // The sysmem calls of BIOS HEAPLIB, served the way our HLE sysmem serves them
        // (iop_sysmem.cpp), so that heaplib is compared in isolation.
        bool sysmemStub(IopMemory &mem, const IopImportCall &call, IopCpuState &cpu)
        {
            if (call.library != "sysmem")
                return false;
            switch (call.ordinal)
            {
            case 4: // AllocSysMemory(type, size, addr)
                cpu.gpr[2] = cpu.gpr[4] == 2u ? mem.allocate(cpu.gpr[5], 16u, cpu.gpr[6]) : mem.allocate(cpu.gpr[5], 16u);
                return true;
            case 5: // FreeSysMemory
                cpu.gpr[2] = mem.freeAllocation(cpu.gpr[4]) ? 0u : 0xFFFFFFFFu;
                return true;
            case 10: // QueryBlockSize
                if (const auto block = mem.allocationContaining(cpu.gpr[4]))
                    cpu.gpr[2] = block->size;
                else
                    cpu.gpr[2] = 0xFFFFFFFFu;
                return true;
            default:
                return false;
            }
        }

        // world: "ioprp" = SYSCLIB + STDIO from IOPRP300.IMG; "heaplib" = HEAPLIB from the BIOS.
        std::unique_ptr<World> makeWorld(const Options &opt, const std::string &kind, std::string &error)
        {
            auto w = std::make_unique<World>();
            const std::string &imagePath = kind == "heaplib" ? opt.bios : opt.ioprp;
            if (imagePath.empty())
            {
                error = "no " + std::string(kind == "heaplib" ? "--bios" : "--ioprp") + " image given";
                return nullptr;
            }
            const auto image = parseRomdir(readFile(imagePath));
            if (image.empty())
            {
                error = "cannot read ROMDIR image " + imagePath;
                return nullptr;
            }
            const std::vector<std::string> names = kind == "heaplib" ? std::vector<std::string>{"HEAPLIB"}
                                                                     : std::vector<std::string>{"SYSCLIB", "STDIO"};
            World *raw = w.get();
            if (kind == "heaplib")
            {
                w->heapWorld = true;
                w->lle.stub = [raw](const IopImportCall &call, IopCpuState &cpu)
                { return sysmemStub(raw->lle.memory, call, cpu); };
            }
            else
            {
                // STDIO's imports that no loaded module serves: ioman read/write (the console);
                // anything else is recorded in stubHits.
                w->lle.stub = [raw](const IopImportCall &call, IopCpuState &cpu)
                {
                    if (call.library == "ioman" && call.ordinal == 7) // write(fd, buf, n)
                    {
                        for (uint32_t i = 0; i < cpu.gpr[6]; ++i)
                            raw->lleOut += static_cast<char>(raw->lle.memory.read8(cpu.gpr[5] + i));
                        cpu.gpr[2] = cpu.gpr[6];
                        return true;
                    }
                    if (call.library == "ioman" && call.ordinal == 6) // read: the console has no input (EOF)
                    {
                        cpu.gpr[2] = 0;
                        return true;
                    }
                    return false;
                };
            }
            for (const auto &name : names)
            {
                const auto it = image.find(name);
                if (it == image.end())
                {
                    error = "no " + name + " in " + imagePath;
                    return nullptr;
                }
                if (!w->lle.loadAndStart(name, it->second))
                {
                    error = "starting the real " + name + " failed";
                    return nullptr;
                }
            }
            (void)w->lle.memory.writeRam(kCodeBase, kPutcCallback.data(), kPutcCallback.size() * 4);
            w->moduleEnd = w->lle.cursor;
            w->baseline.assign(w->lle.memory.ram().begin(), w->lle.memory.ram().end());
            (void)w->hleMem.writeRam(0, w->baseline.data(), w->baseline.size());
            w->stdio.tapOutput(&w->hleOut);
            if (w->heapWorld)
            {
                w->lleSnapshot = w->lle.memory;
                w->hleSnapshot = w->hleMem;
            }
            return w;
        }

        struct Verdict
        {
            bool match = true;
            std::string kind;   // first failing aspect
            std::string detail; // repro text
        };

        bool isHleOwnedHeap(uint32_t a) { return a >= IopMemory::HeapBase && a < IopMemory::HeapLimit; }

        Verdict runCase(World &w, const ExportTest &t, const Case &c, bool verbose)
        {
            Verdict v;
            auto fail = [&](const std::string &kind, const std::string &detail)
            {
                if (v.match)
                {
                    v.match = false;
                    v.kind = kind;
                }
                v.detail += (v.detail.empty() ? "" : "\n      ") + detail;
            };
            if (w.heapWorld)
            {
                w.lle.memory = w.lleSnapshot;
                w.hleMem = w.hleSnapshot;
            }
            for (const auto &[at, data] : c.writes)
                w.writeBoth(at, data.data(), data.size());
            w.lleOut.clear();
            w.hleOut.clear();
            w.sysclib.reset(); // the Sony modules' own state (strtok, the tab column) is restored per case
            w.stdio.reset();
            w.host.logs.clear();
            w.lle.stubHits.clear();

            Rng regs(hashName(c.note) ^ 0x5EEDull);
            uint32_t lowestSp = LleMachine::kStackTop;
            std::vector<uint32_t> lleResults, hleResults;
            for (size_t k = 0; k < c.calls.size(); ++k)
            {
                const Call &call = c.calls[k];
                const uint32_t function = w.lle.resolve(t.library, call.ordinal);
                if (function == 0u)
                {
                    fail("no-export", "the Sony module has no " + std::string(t.library) + ":" + std::to_string(call.ordinal));
                    break;
                }
                IopCpuState init = LleMachine::initialCpu(function, 0, LleMachine::kStackTop - call.spDelta);
                for (uint32_t r = 1; r < 28; ++r)
                    init.gpr[r] = regs.u32();
                init.gpr[28] = w.lle.modules.front().gp;
                init.gpr[30] = regs.u32();
                init.gpr[31] = call.ra;
                lowestSp = std::min(lowestSp, init.gpr[29]);
                // Arguments; a reference to an earlier result resolves per side.
                std::vector<uint32_t> argsL = call.args, argsH = call.args;
                for (const auto &[argIndex, callIndex] : call.refs)
                {
                    argsL[argIndex] += lleResults.at(callIndex);
                    argsH[argIndex] += hleResults.at(callIndex);
                }
                IopCpuState lle = init, hle = init;
                for (size_t i = 0; i < call.args.size(); ++i)
                {
                    if (i < 4)
                    {
                        lle.gpr[4 + i] = argsL[i];
                        hle.gpr[4 + i] = argsH[i];
                    }
                    else
                    {
                        const uint32_t at = init.gpr[29] + 16u + 4u * static_cast<uint32_t>(i - 4);
                        (void)w.lle.memory.writeRam(at, &argsL[i], 4);
                        (void)w.hleMem.writeRam(at, &argsH[i], 4);
                    }
                }
                const auto run = w.lle.run(lle);
                const bool handled = w.hleDispatch(t.library, static_cast<uint16_t>(call.ordinal), hle);
                lleResults.push_back(lle.gpr[2]);
                hleResults.push_back(hle.gpr[2]);
                const std::string tag = c.calls.size() > 1 ? "call " + std::to_string(k + 1) + ": " : "";
                if (run.exit != LleMachine::Exit::Return)
                {
                    fail("lle-fault", tag + "Sony code: " + exitName(run.exit) + " cause=" + std::to_string(run.cause) +
                                          " epc=" + hex(run.epc) + " badvaddr=" + hex(run.badAddress));
                    break;
                }
                if (!handled)
                {
                    fail("hle-unhandled", tag + "HLE has no handler for " + std::string(t.library) + ":" + std::to_string(call.ordinal));
                    break;
                }
                const uint32_t lleRet = lle.pc, hleRet = hle.gpr[31];
                if (t.flags & kCompareJumpRegs)
                {
                    static const int saved[] = {16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31};
                    for (int r : saved)
                        if (lle.gpr[r] != hle.gpr[r])
                            fail("regs", tag + "$" + std::to_string(r) + " Sony=" + hex(lle.gpr[r]) + " HLE=" + hex(hle.gpr[r]));
                    if (lleRet != hleRet)
                        fail("regs", tag + "returned to Sony=" + hex(lleRet) + " HLE=" + hex(hleRet));
                }
                else
                {
                    // ABI: the Sony code must preserve s0-s8, gp, sp and return to ra.
                    static const int saved[] = {16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30};
                    for (int r : saved)
                        if (lle.gpr[r] != init.gpr[r])
                            fail("lle-abi", tag + "Sony code clobbered $" + std::to_string(r));
                    if (hle.gpr[31] != init.gpr[31])
                        fail("regs", tag + "HLE changed $ra");
                }
                if (call.ignoreV0)
                {
                }
                else if (t.flags & kCompareCtypeTable)
                {
                    if ((lle.gpr[2] == 0) != (hle.gpr[2] == 0))
                        fail("v0", tag + "table pointer Sony=" + hex(lle.gpr[2]) + " HLE=" + hex(hle.gpr[2]));
                    else
                        for (int32_t i = -1; i < 128; ++i)
                        {
                            const uint8_t x = w.lle.memory.read8(lle.gpr[2] + i), y = w.hleMem.read8(hle.gpr[2] + i);
                            if (x != y)
                            {
                                fail("v0", tag + "table[" + std::to_string(i) + "] Sony=" + std::to_string(x) + " HLE=" + std::to_string(y));
                                break;
                            }
                        }
                }
                else if (lle.gpr[2] != hle.gpr[2])
                {
                    auto describe = [&](uint32_t p) -> std::string
                    {
                        if (p >= kArenaBase && p < kArenaEnd)
                            return " (arena+" + std::to_string(p - kArenaBase) + ")";
                        return "";
                    };
                    fail("v0", tag + "v0 Sony=" + hex(lle.gpr[2]) + describe(lle.gpr[2]) + " HLE=" + hex(hle.gpr[2]) + describe(hle.gpr[2]));
                }
            }

            // Memory: all of RAM except the callee stack window, the module images (Sony-internal
            // state such as strtok's saved pointer) and the HLE-owned heap.
            const auto a = w.lle.memory.ram();
            const auto bmem = w.hleMem.ram();
            const uint32_t stackLo = lowestSp - kStackWindow, stackHi = LleMachine::kStackTop + 0x100;
            uint32_t diffs = 0;
            std::string firstDiffs;
            bool outsideWindows = false;
            for (uint32_t addr = 0; addr < IopMemory::RamSize; addr += 4096)
            {
                if (std::memcmp(a.data() + addr, bmem.data() + addr, 4096) == 0)
                    continue;
                for (uint32_t i = addr; i < addr + 4096; ++i)
                {
                    if (a[i] == bmem[i])
                        continue;
                    if ((i >= stackLo && i < stackHi) || (i >= kModuleLoadBase && i < w.moduleEnd) || (!w.heapWorld && isHleOwnedHeap(i)))
                        continue;
                    if (i < kArenaBase || i >= kArenaEnd)
                        outsideWindows = true;
                    if (diffs < 8)
                    {
                        char buf[64];
                        std::snprintf(buf, sizeof buf, "%s[%06x] Sony=%02x HLE=%02x", diffs ? ", " : "", i, a[i], bmem[i]);
                        firstDiffs += buf;
                    }
                    ++diffs;
                }
            }
            if (diffs)
                fail("memory", std::to_string(diffs) + " byte(s) differ: " + firstDiffs);
            if (w.lleOut != w.hleOut)
                fail("output", "console Sony=" + quote(w.lleOut) + " HLE=" + quote(w.hleOut));
            for (const auto &[k, n] : w.lle.stubHits)
                fail("lle-stub", "Sony code called unserved import " + k + " x" + std::to_string(n));

            if (verbose)
            {
                std::cout << "    inputs: " << c.note << "\n";
                if (!w.lleOut.empty() || !w.hleOut.empty())
                    std::cout << "    console Sony=" << quote(w.lleOut, 400) << "\n            HLE =" << quote(w.hleOut, 400) << "\n";
                if (!v.match)
                    std::cout << "    " << v.detail << "\n";
            }

            // Restore both machines for the next case.
            if (w.heapWorld)
            {
                // the next case starts from the snapshots
            }
            else if (outsideWindows)
            {
                w.restoreRange(0, IopMemory::RamSize);
            }
            else
            {
                w.restoreRange(kArenaBase, kArenaEnd);
                w.restoreRange(std::max<uint32_t>(stackLo, 0x1F0000u), stackHi);
                w.restoreRange(kModuleLoadBase, w.moduleEnd);
            }
            return v;
        }
    }

    int runStage1(const Options &opt, Summary &summary)
    {
        struct Suite
        {
            const char *world;
            const std::vector<ExportTest> *tests;
        };
        const Suite suites[] = {{"ioprp", &sysclibTests()}, {"ioprp", &stdioTests()}, {"heaplib", &heaplibTests()}};
        std::map<std::string, std::unique_ptr<World>> worlds;
        int rc = 0;
        for (const Suite &suite : suites)
        {
            std::vector<const ExportTest *> tests;
            for (const auto &t : *suite.tests)
            {
                const std::string id = std::string(t.library) + ":" + std::to_string(t.ordinal);
                const std::string full = id + " " + t.name;
                if (opt.filter.empty() || full.find(opt.filter) != std::string::npos || id == opt.filter)
                    tests.push_back(&t);
            }
            if (tests.empty())
                continue;
            auto &world = worlds[suite.world];
            if (!world)
            {
                std::string error;
                world = makeWorld(opt, suite.world, error);
                if (!world)
                {
                    std::cout << "stage1 " << suite.world << ": SKIP " << error << "\n";
                    rc = 2;
                    continue;
                }
                if (opt.verbose)
                {
                    for (const auto &m : world->lle.modules)
                        std::cout << "  loaded Sony " << m.name << " base=" << hex(m.base) << " start=" << m.startResult << "\n";
                    for (const auto &lib : world->lle.libraries)
                    {
                        std::cout << "  export table " << lib.name << " v" << std::hex << lib.version << std::dec << " ("
                                  << lib.functions.size() << " slots)\n";
                        for (size_t i = 0; i < lib.functions.size(); ++i)
                            std::cout << "    " << lib.name << ":" << i << " " << hex(lib.functions[i]) << "\n";
                    }
                }
            }
            if (!world)
                continue;
            for (const ExportTest *t : tests)
            {
                ExportResult res;
                res.id = std::string(t->library) + ":" + std::to_string(t->ordinal);
                res.name = t->name;
                const uint32_t first = opt.onlyCase >= 0 ? static_cast<uint32_t>(opt.onlyCase) : 0u;
                const uint32_t last = opt.onlyCase >= 0 ? first + 1 : opt.cases;
                for (uint32_t k = first; k < last; ++k)
                {
                    Rng rng(opt.seed ^ hashName(t->name) ^ (uint64_t(k) * 0x9E3779B97F4A7C15ull));
                    Builder b(rng);
                    t->gen(b);
                    if (b.c.note.empty())
                        b.c.note = "(no inputs)";
                    b.c.note = "case " + std::to_string(k) + ": " + b.c.note;
                    const Verdict v = runCase(*world, *t, b.c, opt.verbose);
                    ++res.cases;
                    if (v.match)
                        ++res.matches;
                    else
                    {
                        ++res.mismatches;
                        ++res.kinds[v.kind];
                        if (res.examples.size() < opt.examples)
                            res.examples.push_back(b.c.note + "\n      " + v.detail);
                    }
                }
                summary.add(res);
            }
        }
        return rc;
    }
}
