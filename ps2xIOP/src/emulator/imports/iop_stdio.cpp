// STDIO as the IOPRP300 firmware implements it; checked against the original module by
// ps2_iop_ground_truth (tools/conformance/iop.sh). Firmware behaviour worth knowing:
//   - printf-family output goes through SYSCLIB prnt(); "\n" is written as "\r\n";
//   - puts() does NOT append a newline (it is fdputs(s, 1)) and returns 0;
//   - putchar/fdputc expand '\t' to the next 8-column stop and return their argument unchanged;
//   - input: the IOP console has no input here, so getchar/fdgetc return -1 and gets/fdgets store
//     an empty string and return NULL.
#include "iop_stdio.h"
#include "iop_sysclib.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

namespace ps2x::iop::detail
{
    IopStdio::IopStdio(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    void IopStdio::reset() noexcept
    {
        m_line.clear();
        m_column = 0;
    }

    void IopStdio::write(std::string_view bytes)
    {
        if (m_tap)
            m_tap->append(bytes);
        for (char ch : bytes)
        {
            if (ch != '\n')
            {
                m_line += ch;
                continue;
            }
            if (!m_line.empty() && m_line.back() == '\r')
                m_line.pop_back();
            m_host.log(LogLevel::Info, "[IOP stdout] " + m_line);
            m_line.clear();
        }
    }

    uint32_t IopStdio::putc(uint32_t c)
    {
        const int8_t ch = static_cast<int8_t>(c & 0xFFu);
        if (ch == '\t')
        {
            write(std::string_view("        ", 8u - (m_column & 7u)));
            m_column = (m_column & ~7u) + 8u;
        }
        else if (ch == '\n')
        {
            write("\r\n");
            m_column = 0;
        }
        else
        {
            if (iopCtype(ch) & 0x97u)
                ++m_column;
            const char byte = static_cast<char>(ch);
            write(std::string_view(&byte, 1));
        }
        return c;
    }

    bool IopStdio::dispatchImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        const auto print = [&](uint32_t format, IopVarArgs &args) -> uint32_t
        {
            std::string out;
            const int32_t n = iopPrnt(m_memory, format, args, [&](int32_t ch)
                                      {
                                          if (ch == 0x200 || ch == 0x201)
                                              return;
                                          if (ch == '\n')
                                              out += '\r';
                                          out += static_cast<char>(ch & 0xFF);
                                      });
            write(out);
            return static_cast<uint32_t>(n);
        };
        const auto puts = [&](uint32_t s)
        {
            if (s == 0u)
            {
                for (char ch : std::string_view("<NULL>"))
                    putc(static_cast<uint8_t>(ch));
                return;
            }
            for (uint32_t i = 0; i < (1u << 20); ++i)
            {
                const uint8_t ch = m_memory.read8(s + i);
                if (ch == 0u)
                    break;
                putc(ch);
            }
        };

        switch (ordinal)
        {
        case 4: // printf(fmt, ...)
        {
            IopVarArgs args(m_memory, {cpu.gpr[5], cpu.gpr[6], cpu.gpr[7]}, cpu.gpr[29] + 16u);
            setV0(print(a0, args));
            return true;
        }
        case 9: // fdprintf(fd, fmt, ...)
        {
            IopVarArgs args(m_memory, {cpu.gpr[6], cpu.gpr[7]}, cpu.gpr[29] + 16u);
            setV0(print(a1, args));
            return true;
        }
        case 14: // vfdprintf(fd, fmt, ap)
        {
            IopVarArgs args(m_memory, {}, cpu.gpr[6]);
            setV0(print(a1, args));
            return true;
        }
        case 5:  // getchar
        case 10: // fdgetc(fd)
            setV0(0xFFFFFFFFu);
            return true;
        case 6: // putchar(c)
            setV0(putc(a0));
            return true;
        case 11: // fdputc(c, fd)
            setV0(putc(a0));
            return true;
        case 7:  // puts(s): fdputs(s, 1), no newline
        case 12: // fdputs(s, fd)
            puts(a0);
            setV0(0u);
            return true;
        case 8:  // gets(buf)
        case 13: // fdgets(buf, fd): end of input before any character
            m_memory.write8(a0, 0u);
            setV0(0u);
            return true;
        default:
            return false;
        }
    }
}
