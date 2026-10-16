// SYSCLIB (the IOP's C library) as the IOPRP300 firmware that TC2 boots with implements it.
// Every export here is checked against the original module, run in our interpreter, by
// ps2_iop_ground_truth (tools/conformance/iop.sh). The firmware differs from ISO C in places that
// matter to callers, and those differences are reproduced on purpose:
//   - chars are signed: strcmp/strncmp return (signed char) differences, _toupper sign-extends;
//   - memcmp/bcmp return -1/0/1; sizes are signed ints (a "negative" size is 0);
//   - strtol/strtoul: base 0 means 10, a "0x"/"0b"/"O" prefix overrides the base, '+' is not a
//     sign, every '-' flips the sign, *endptr is set even when no digit was read;
//   - strtok_r stores NULL in *lasts at the end of the string; longjmp(buf, 0) returns 0;
//   - NULL string/buffer arguments return 0 instead of faulting in most functions.
#include "iop_sysclib.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "../services/iop_rpc.h"

#include <string>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kStringLimit = 1u << 20; // runaway guard for unterminated guest strings

        int8_t s8(uint32_t v) { return static_cast<int8_t>(v & 0xFFu); }
        uint32_t u(int32_t v) { return static_cast<uint32_t>(v); }
    }

    uint8_t iopCtype(int32_t c) noexcept
    {
        if (c < 0 || c > 127)
            return 0; // EOF; the firmware reads unrelated bytes for 128..255 (load-address dependent)
        if (c >= '0' && c <= '9')
            return 0x04;
        if (c >= 'A' && c <= 'Z')
            return c <= 'F' ? 0x41 : 0x01;
        if (c >= 'a' && c <= 'z')
            return c <= 'f' ? 0x42 : 0x02;
        if (c == ' ')
            return 0x18;
        if (c >= 9 && c <= 13)
            return 0x08;
        if (c < 0x20 || c == 0x7F)
            return 0x20;
        return 0x10;
    }

    uint32_t IopVarArgs::next()
    {
        if (m_used < m_registers.size())
            return m_registers[m_used++];
        m_next = (m_next + 3u) & ~3u;
        const uint32_t value = m_memory.read32(m_next);
        m_next += 4u;
        return value;
    }

    int32_t iopPrnt(IopMemory &memory, uint32_t format, IopVarArgs &args, const std::function<void(int32_t)> &emit)
    {
        if (format == 0u)
            return 0;
        constexpr uint32_t kLeft = 0x10, kZero = 0x20, kAlt = 0x08, kPrefix = 0x40, kLong = 0x01, kShort = 0x04, kLongDouble = 0x02;
        static const char lower[] = "0123456789abcdef";
        static const char upper[] = "0123456789ABCDEF";
        emit(0x200);
        int32_t count = 0;
        // %X switches the digit set; the firmware only switches back after it has generated
        // digits, so "%.0X" of 0 leaves the next %x upper-case.
        const char *digits = lower;
        uint32_t p = format;
        for (;; ++p)
        {
            int32_t c = s8(memory.read8(p));
            if (c == 0)
                break;
            if (c != '%')
            {
                emit(c);
                ++count;
                continue;
            }
            uint32_t flags = 0;
            int32_t width = 0, precision = -1, sign = 0, numberPrecision = 0;
            bool done = false;   // conversion produced output (or %n)
            bool literal = false; // unknown conversion: print the character itself
            bool end = false;     // format ended inside the conversion
            // Converted text: either the number buffer or a guest string.
            std::string text;
            for (;;)
            {
                ++p;
                c = s8(memory.read8(p));
                if (c == 0)
                {
                    end = true;
                    break;
                }
                if (static_cast<uint32_t>(c) >= 0x79u)
                {
                    literal = true;
                    break;
                }
                switch (c)
                {
                case ' ':
                    if (sign == 0)
                        sign = ' ';
                    continue;
                case '#':
                    flags |= kAlt;
                    continue;
                case '*':
                    width = static_cast<int32_t>(args.next());
                    if (width < 0)
                    {
                        width = -width;
                        flags |= kLeft;
                    }
                    continue;
                case '-':
                    flags |= kLeft;
                    continue;
                case '+':
                    sign = '+';
                    continue;
                case '.':
                {
                    ++p;
                    int32_t value = 0;
                    if (s8(memory.read8(p)) == '*')
                        value = static_cast<int32_t>(args.next());
                    else
                    {
                        while (iopCtype(s8(memory.read8(p))) & 0x04)
                            value = value * 10 + s8(memory.read8(p++)) - '0';
                        --p;
                    }
                    precision = value >= 0 ? value : -1;
                    continue;
                }
                case '0':
                    flags |= kZero;
                    continue;
                case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                {
                    int32_t value = 0;
                    do
                        value = value * 10 + s8(memory.read8(p++)) - '0';
                    while (iopCtype(s8(memory.read8(p))) & 0x04);
                    --p;
                    width = value;
                    continue;
                }
                case 'L':
                    flags |= kLongDouble;
                    continue;
                case 'h':
                    flags |= kShort;
                    continue;
                case 'l':
                    flags |= kLong;
                    continue;
                default:
                    break;
                }
                // Conversions.
                uint32_t value = 0, base = 0;
                switch (c)
                {
                case 'c':
                    text.assign(1, static_cast<char>(args.next() & 0xFFu));
                    sign = 0;
                    done = true;
                    break;
                case 'D':
                case 'd':
                case 'i':
                {
                    const int32_t v = static_cast<int32_t>(args.next());
                    base = 10;
                    value = static_cast<uint32_t>(v);
                    if (v < 0)
                    {
                        value = 0u - value;
                        sign = '-';
                    }
                    break;
                }
                case 'n':
                {
                    const uint32_t where = args.next();
                    if (!(flags & kLong) && (flags & kShort))
                        memory.write16(where, static_cast<uint16_t>(count));
                    else
                        memory.write32(where, static_cast<uint32_t>(count));
                    done = true;
                    text.clear();
                    break;
                }
                case 'O':
                case 'o':
                    value = args.next();
                    base = 8;
                    sign = 0;
                    break;
                case 'p':
                    value = args.next();
                    base = 16;
                    sign = 0;
                    break;
                case 's':
                {
                    uint32_t s = args.next();
                    std::string str;
                    if (s == 0u)
                        str = "(null)";
                    else
                    {
                        // len = min(strlen, precision) without reading past the precision
                        for (uint32_t i = 0; i < kStringLimit && (precision < 0 || i < static_cast<uint32_t>(precision)); ++i)
                        {
                            const uint8_t ch = memory.read8(s + i);
                            if (ch == 0)
                                break;
                            str += static_cast<char>(ch);
                        }
                    }
                    if (s == 0u && precision >= 0 && str.size() > static_cast<size_t>(precision))
                        str.resize(static_cast<size_t>(precision));
                    text = str;
                    sign = 0;
                    done = true;
                    break;
                }
                case 'U':
                case 'u':
                    value = args.next();
                    base = 10;
                    sign = 0;
                    break;
                case 'X':
                    digits = upper;
                    [[fallthrough]];
                case 'x':
                    value = args.next();
                    base = 16;
                    if ((flags & kAlt) && value != 0u)
                        flags |= kPrefix;
                    sign = 0;
                    break;
                default:
                    literal = true;
                    break;
                }
                if (literal)
                    break;
                if (c == 'n')
                {
                    done = false; // no output, no count
                    break;
                }
                if (base != 0)
                {
                    numberPrecision = precision;
                    if (precision >= 0)
                        flags &= ~kZero;
                    if (value != 0u || numberPrecision != 0)
                    {
                        std::string rev;
                        do
                        {
                            rev += digits[value % base];
                            value /= base;
                        } while (value != 0u);
                        digits = lower;
                        if ((flags & kAlt) && base == 8 && rev.back() != '0')
                            rev += '0';
                        text.assign(rev.rbegin(), rev.rend());
                    }
                    done = true;
                }
                break;
            }
            if (end)
                break;
            if (literal)
            {
                emit(c);
                ++count;
                continue;
            }
            if (!done)
                continue;

            const int32_t length = static_cast<int32_t>(text.size());
            int32_t needed = length + (sign ? 1 : 0) + ((flags & kPrefix) ? 2 : 0);
            const int32_t field = std::max(needed, numberPrecision);
            if ((flags & (kLeft | kZero)) == 0 && width != 0)
                for (int32_t i = field; i < width; ++i)
                    emit(' ');
            if (sign)
                emit(sign);
            if (flags & kPrefix)
            {
                emit('0');
                emit(c);
            }
            if ((flags & (kLeft | kZero)) == kZero)
                for (int32_t i = field; i < width; ++i)
                    emit('0');
            for (int32_t i = needed; i < numberPrecision; ++i)
                emit('0');
            for (char ch : text)
                emit(static_cast<int8_t>(ch));
            if (flags & kLeft)
                for (int32_t i = field; i < width; ++i)
                    emit(' ');
            count += std::max(field, width);
        }
        emit(0x201);
        return count;
    }

    IopSysclib::IopSysclib(IopMemory &memory) noexcept
        : m_memory(memory)
    {
    }

    void IopSysclib::reset() noexcept
    {
        m_strtokSave = 0;
        m_ctypeTable = 0;
    }

    bool IopSysclib::dispatchImport(uint16_t ordinal, IopCpuState &cpu, IopGuestExecutor *executor)
    {
        IopMemory &mem = m_memory;
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const uint32_t a2 = cpu.gpr[6];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        const auto strlen = [&](uint32_t s) -> uint32_t
        {
            uint32_t n = 0;
            while (n < kStringLimit && mem.read8(s + n) != 0u)
                ++n;
            return n;
        };
        const auto wordCopy = [&](uint32_t d, uint32_t s, uint32_t bytes)
        {
            // _wmemcopy: the remainder words one at a time, then groups of four (read all, write all).
            const int32_t words = static_cast<int32_t>(bytes >> 2);
            if (words <= 0)
                return;
            const int32_t single = words & 3;
            int32_t i = 0;
            for (; i < single; ++i, d += 4, s += 4)
                mem.write32(d, mem.read32(s));
            for (; i < words; i += 4, d += 16, s += 16)
            {
                const uint32_t w0 = mem.read32(s), w1 = mem.read32(s + 4), w2 = mem.read32(s + 8), w3 = mem.read32(s + 12);
                mem.write32(d, w0);
                mem.write32(d + 4, w1);
                mem.write32(d + 8, w2);
                mem.write32(d + 12, w3);
            }
        };
        const auto wordSet = [&](uint32_t d, uint32_t value, uint32_t bytes)
        {
            const int32_t words = static_cast<int32_t>(bytes >> 2);
            for (int32_t i = 0; i < words; ++i, d += 4)
                mem.write32(d, value);
        };
        const auto memmove = [&](uint32_t d, uint32_t s, uint32_t n) -> uint32_t
        {
            if (d == 0u)
                return 0u;
            const int32_t count = static_cast<int32_t>(n);
            if (d < s)
                for (int32_t i = 0; i < count; ++i)
                    mem.write8(d + i, mem.read8(s + i));
            else
                for (int32_t i = count - 1; i >= 0; --i)
                    mem.write8(d + i, mem.read8(s + i));
            return d;
        };
        const auto memset = [&](uint32_t d, uint32_t c, uint32_t n) -> uint32_t
        {
            if (d == 0u)
                return 0u;
            if (c == 0u && ((d | n) & 3u) == 0u)
            {
                wordSet(d, 0u, n);
                return d;
            }
            for (int32_t i = 0; i < static_cast<int32_t>(n); ++i)
                mem.write8(d + i, static_cast<uint8_t>(c));
            return d;
        };
        const auto memcmp = [&](uint32_t p, uint32_t q, uint32_t n) -> uint32_t
        {
            for (int32_t i = 0; i < static_cast<int32_t>(n); ++i)
            {
                const uint8_t x = mem.read8(p + i), y = mem.read8(q + i);
                if (x != y)
                    return x < y ? 0xFFFFFFFFu : 1u;
            }
            return 0u;
        };
        const auto strchr = [&](uint32_t s, uint32_t c) -> uint32_t
        {
            if (s == 0u)
                return 0u;
            const int8_t needle = s8(c);
            for (uint32_t i = 0; i < kStringLimit; ++i)
            {
                const int8_t ch = s8(mem.read8(s + i));
                if (ch == needle)
                    return s + i;
                if (ch == 0)
                    break;
            }
            return 0u;
        };
        const auto strrchr = [&](uint32_t s, uint32_t c) -> uint32_t
        {
            if (s == 0u)
                return 0u;
            const int8_t needle = s8(c);
            for (uint32_t p = s + strlen(s);; --p)
            {
                if (s8(mem.read8(p)) == needle)
                    return p;
                if (p == s)
                    return 0u;
            }
        };
        const auto nullCompare = [&](uint32_t p, uint32_t q) -> uint32_t
        {
            if (p == q)
                return 0u;
            return p == 0u ? 0xFFFFFFFFu : 1u;
        };
        const auto inSet = [&](int8_t ch, uint32_t set) -> bool
        {
            for (uint32_t i = 0; i < kStringLimit; ++i)
            {
                const int8_t d = s8(mem.read8(set + i));
                if (d == 0)
                    return false;
                if (d == ch)
                    return true;
            }
            return false;
        };
        const auto digitValue = [&](uint8_t raw) -> int32_t
        {
            const int8_t c = s8(raw);
            const uint8_t f = iopCtype(c);
            if (f & 0x04)
                return c - '0';
            if (f & 0x03)
                return s8(static_cast<uint32_t>((f & 0x01) ? c + 0x20 : c)) - 0x57;
            return 9999999;
        };
        // strtol/strtoul share everything but the sign and the digit comparison.
        const auto strtoAny = [&](uint32_t s, uint32_t *endOut, uint32_t base, bool isSigned) -> uint32_t
        {
            if (s == 0u)
                return 0u;
            int32_t signum = 1;
            while (iopCtype(s8(mem.read8(s))) & 0x08)
                ++s;
            if (isSigned && s8(mem.read8(s)) == '-')
            {
                do
                {
                    ++s;
                    signum = -signum;
                } while (s8(mem.read8(s)) == '-');
            }
            if (base - 2u >= 35u)
                base = 0;
            if (base == 0)
                base = 10;
            const int8_t c = s8(mem.read8(s));
            if (c == '0')
            {
                ++s;
                const int8_t c2 = s8(mem.read8(s));
                if (c2 == 'X' || c2 == 'x')
                {
                    ++s;
                    base = 16;
                }
                else if (c2 == 'B' || c2 == 'b')
                {
                    ++s;
                    base = 2;
                }
            }
            else if (c == 'O' || c == 'o')
            {
                ++s;
                base = 8;
            }
            uint32_t acc = 0;
            for (;;)
            {
                const int32_t d = digitValue(mem.read8(s++));
                const bool ok = isSigned ? d < static_cast<int32_t>(base) : static_cast<uint32_t>(d) < base;
                if (!ok)
                    break;
                acc = acc * base + static_cast<uint32_t>(d);
            }
            --s;
            *endOut = s;
            return isSigned ? acc * static_cast<uint32_t>(signum) : acc;
        };
        const auto strtok = [&](uint32_t s, uint32_t delim, uint32_t &lasts) -> uint32_t
        {
            if (s == 0u)
            {
                s = lasts;
                if (s == 0u)
                    return 0u;
            }
            for (;;) // skip leading delimiters
            {
                const int8_t ch = s8(mem.read8(s++));
                if (ch == 0)
                {
                    lasts = 0u;
                    return 0u;
                }
                if (!inSet(ch, delim))
                    break;
            }
            const uint32_t token = s - 1u;
            for (;;)
            {
                const int8_t ch = s8(mem.read8(s++));
                if (ch == 0)
                {
                    lasts = 0u;
                    return token;
                }
                if (inSet(ch, delim))
                {
                    mem.write8(s - 1u, 0u);
                    lasts = s;
                    return token;
                }
            }
        };
        const auto sprintfInto = [&](uint32_t buffer, uint32_t format, IopVarArgs &args) -> uint32_t
        {
            uint32_t cursor = buffer;
            const int32_t n = iopPrnt(mem, format, args, [&](int32_t ch)
                                      {
                                          if (ch < 0x100)
                                              mem.write8(cursor++, static_cast<uint8_t>(ch));
                                          else
                                              mem.write8(cursor, 0u); // 0x200 start / 0x201 end
                                      });
            return static_cast<uint32_t>(n);
        };

        switch (ordinal)
        {
        case 4: // setjmp: ra, sp, fp, s0-s7, gp
        {
            static const int regs[] = {31, 29, 30, 16, 17, 18, 19, 20, 21, 22, 23, 28};
            for (int i = 0; i < 12; ++i)
                mem.write32(a0 + 4u * static_cast<uint32_t>(i), cpu.gpr[regs[i]]);
            setV0(0);
            return true;
        }
        case 5: // longjmp: restore and "return" to the saved ra with v0 = val (0 stays 0)
        {
            static const int regs[] = {31, 29, 30, 16, 17, 18, 19, 20, 21, 22, 23, 28};
            for (int i = 0; i < 12; ++i)
                cpu.gpr[regs[i]] = mem.read32(a0 + 4u * static_cast<uint32_t>(i));
            setV0(a1);
            return true;
        }
        case 6: // _toupper
        case 7: // _tolower
        {
            const uint8_t f = iopCtype(s8(a0));
            uint32_t r = a0;
            if (ordinal == 6 && (f & 0x02))
                r = a0 - 0x20u;
            if (ordinal == 7 && (f & 0x01))
                r = a0 + 0x20u;
            setV0(static_cast<uint32_t>(static_cast<int32_t>(s8(r))));
            return true;
        }
        case 8: // look_ctype_table(c): the table is indexed with the full argument
        {
            const int32_t c = static_cast<int32_t>(a0);
            setV0(static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(iopCtype(c)))));
            return true;
        }
        case 9: // get_ctype_table: pointer to the entry for 0; entry -1 (EOF) precedes it
            if (m_ctypeTable == 0u)
            {
                m_ctypeTable = mem.allocate(132u, 4u);
                if (m_ctypeTable != 0u)
                    for (int32_t c = -1; c < 128; ++c)
                        mem.write8(m_ctypeTable + static_cast<uint32_t>(c + 1), iopCtype(c));
            }
            setV0(m_ctypeTable != 0u ? m_ctypeTable + 1u : 0u);
            return true;
        case 10: // memchr
            if (a0 != 0u)
                for (int32_t i = 0; i < static_cast<int32_t>(a2); ++i)
                    if (mem.read8(a0 + i) == static_cast<uint8_t>(a1))
                    {
                        setV0(a0 + i);
                        return true;
                    }
            setV0(0);
            return true;
        case 11: // memcmp
        case 15: // bcmp
            setV0(memcmp(a0, a1, a2));
            return true;
        case 12: // memcpy
            if (a0 == 0u)
                setV0(0);
            else
            {
                if (((a0 | a1 | a2) & 3u) == 0u)
                    wordCopy(a0, a1, a2);
                else
                    for (int32_t i = 0; i < static_cast<int32_t>(a2); ++i)
                        mem.write8(a0 + i, mem.read8(a1 + i));
                setV0(a0);
            }
            return true;
        case 13:
            setV0(memmove(a0, a1, a2));
            return true;
        case 14:
            setV0(memset(a0, a1, a2));
            return true;
        case 16: // bcopy(src, dst, n) = memmove(dst, src, n)
            setV0(memmove(a1, a0, a2));
            return true;
        case 17: // bzero
            setV0(memset(a0, 0u, a1));
            return true;
        case 18: // prnt(callback, ctx, fmt, ap): callback(ctx, c) per character, 0x200 first, 0x201 last
        {
            if (executor == nullptr)
            {
                setV0(0);
                return true;
            }
            IopVarArgs args(mem, {}, cpu.gpr[7]);
            const uint32_t gp = cpu.gpr[28];
            setV0(static_cast<uint32_t>(iopPrnt(mem, a2, args, [&](int32_t ch)
                                                { (void)executor->executeGuestFunction(a0, a1, static_cast<uint32_t>(ch), 0u, 0u, gp); })));
            return true;
        }
        case 19: // sprintf(buf, fmt, ...)
        {
            IopVarArgs args(mem, {cpu.gpr[6], cpu.gpr[7]}, cpu.gpr[29] + 16u);
            setV0(sprintfInto(a0, a1, args));
            return true;
        }
        case 42: // vsprintf(buf, fmt, ap)
        {
            IopVarArgs args(mem, {}, a2);
            setV0(sprintfInto(a0, a1, args));
            return true;
        }
        case 20: // strcat
        {
            if (a0 == 0u || a1 == 0u)
            {
                setV0(0);
                return true;
            }
            const uint32_t end = a0 + strlen(a0);
            if (end == a1 + strlen(a1))
            {
                setV0(0);
                return true;
            }
            for (uint32_t i = 0; i < kStringLimit; ++i)
            {
                const uint8_t ch = mem.read8(a1 + i);
                mem.write8(end + i, ch);
                if (ch == 0u)
                    break;
            }
            setV0(a0);
            return true;
        }
        case 21: // strchr
        case 25: // index
            setV0(strchr(a0, a1));
            return true;
        case 22: // strcmp
            if (a0 == 0u || a1 == 0u)
                setV0(nullCompare(a0, a1));
            else
                for (uint32_t i = 0;; ++i)
                {
                    const int8_t x = s8(mem.read8(a0 + i)), y = s8(mem.read8(a1 + i));
                    if (x != y)
                    {
                        setV0(u(x - y));
                        break;
                    }
                    if (x == 0 || i >= kStringLimit)
                    {
                        setV0(0);
                        break;
                    }
                }
            return true;
        case 23: // strcpy
            if (a0 == 0u || a1 == 0u)
            {
                setV0(0);
                return true;
            }
            for (uint32_t i = 0; i < kStringLimit; ++i)
            {
                const uint8_t ch = mem.read8(a1 + i);
                mem.write8(a0 + i, ch);
                if (ch == 0u)
                    break;
            }
            setV0(a0);
            return true;
        case 24: // strcspn
        {
            uint32_t n = 0;
            for (; n < kStringLimit; ++n)
            {
                const int8_t ch = s8(mem.read8(a0 + n));
                if (ch == 0 || inSet(ch, a1))
                    break;
            }
            setV0(n);
            return true;
        }
        case 26: // rindex
        case 32: // strrchr
            setV0(strrchr(a0, a1));
            return true;
        case 27: // strlen
            setV0(a0 == 0u ? 0u : strlen(a0));
            return true;
        case 28: // strncat: copies at most n characters, then a NUL; n < 0 behaves as 0
        {
            if (a0 == 0u || a1 == 0u)
            {
                setV0(0);
                return true;
            }
            uint32_t e = a0 + strlen(a0), s = a1;
            int32_t n = static_cast<int32_t>(a2);
            for (;;)
            {
                const uint8_t ch = mem.read8(s++);
                mem.write8(e, ch);
                if (ch == 0u)
                    break;
                ++e;
                if (--n < 0)
                {
                    mem.write8(e - 1u, 0u);
                    break;
                }
            }
            setV0(a0);
            return true;
        }
        case 29: // strncmp
        {
            if (a0 == 0u || a1 == 0u)
            {
                setV0(nullCompare(a0, a1));
                return true;
            }
            int32_t n = static_cast<int32_t>(a2);
            uint32_t result = 0;
            for (uint32_t i = 0; --n >= 0; ++i)
            {
                const int8_t x = s8(mem.read8(a0 + i)), y = s8(mem.read8(a1 + i));
                if (x != y)
                {
                    result = u(x - y);
                    break;
                }
                if (x == 0)
                    break;
            }
            setV0(result);
            return true;
        }
        case 30: // strncpy
        {
            if (a0 == 0u || a1 == 0u)
            {
                setV0(0);
                return true;
            }
            const int32_t n = static_cast<int32_t>(a2);
            int32_t i = 0;
            bool ended = false;
            for (; i < n; ++i)
            {
                const uint8_t ch = ended ? 0u : mem.read8(a1 + i);
                mem.write8(a0 + i, ch);
                ended = ended || ch == 0u;
            }
            setV0(a0);
            return true;
        }
        case 31: // strpbrk
            for (uint32_t i = 0; i < kStringLimit; ++i)
            {
                const int8_t ch = s8(mem.read8(a0 + i));
                if (ch == 0)
                    break;
                if (inSet(ch, a1))
                {
                    setV0(a0 + i);
                    return true;
                }
            }
            setV0(0);
            return true;
        case 33: // strspn
        {
            uint32_t n = 0;
            for (; n < kStringLimit; ++n)
            {
                const int8_t ch = s8(mem.read8(a0 + n));
                if (ch == 0 || !inSet(ch, a1))
                    break;
            }
            setV0(n);
            return true;
        }
        case 34: // strstr
        {
            if (mem.read8(a1) == 0u)
            {
                setV0(a0);
                return true;
            }
            for (uint32_t h = a0; mem.read8(h) != 0u && h - a0 < kStringLimit; ++h)
            {
                for (uint32_t i = 0;; ++i)
                {
                    const uint8_t want = mem.read8(a1 + i);
                    if (want == 0u)
                    {
                        setV0(h);
                        return true;
                    }
                    const uint8_t have = mem.read8(h + i);
                    if (have == 0u)
                    {
                        setV0(0);
                        return true;
                    }
                    if (have != want)
                        break;
                }
            }
            setV0(0);
            return true;
        }
        case 35: // strtok
            setV0(strtok(a0, a1, m_strtokSave));
            return true;
        case 43: // strtok_r(s, delim, lasts)
        {
            uint32_t lasts = mem.read32(a2);
            const uint32_t token = strtok(a0, a1, lasts);
            mem.write32(a2, lasts);
            setV0(token);
            return true;
        }
        case 36: // strtol
        case 38: // strtoul
        {
            uint32_t end = 0u;
            setV0(strtoAny(a0, &end, a2, ordinal == 36));
            if (a0 != 0u && a1 != 0u)
                mem.write32(a1, end);
            return true;
        }
        case 37: // atob(s, &i): *i = strtol(s, &end, 10); return end (NULL s: end is never set)
        {
            uint32_t end = 0u;
            mem.write32(a1, strtoAny(a0, &end, 10u, true));
            setV0(end);
            return true;
        }
        case 39: // _retonly
            return true;
        case 40: // _wmemcopy(dst, src, bytes)
            wordCopy(a0, a1, a2);
            setV0(a0);
            return true;
        case 41: // _wmemset(dst, word, bytes)
            wordSet(a0, a1, a2);
            setV0(a0);
            return true;
        case 44: // unnamed slots: return -1
        case 45:
        case 46:
        case 47:
            setV0(0xFFFFFFFFu);
            return true;
        default:
            return false;
        }
    }
}
