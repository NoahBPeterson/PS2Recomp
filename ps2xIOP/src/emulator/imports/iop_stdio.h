#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ps2x::iop
{
    class IopHost;
}

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopMemory;

    // STDIO as the IOPRP300 firmware implements it (checked against the original module by
    // ps2_iop_ground_truth). Every console byte the firmware would write(2) goes to the host log,
    // one line per log entry ("[IOP stdout] ...").
    class IopStdio
    {
    public:
        IopStdio(IopHost &host, IopMemory &memory) noexcept;

        void reset() noexcept;
        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu);

        // Test seam: every console byte is also appended to *sink (nullptr disables).
        void tapOutput(std::string *sink) noexcept { m_tap = sink; }

    private:
        void write(std::string_view bytes);
        uint32_t putc(uint32_t c);

        IopHost &m_host;
        IopMemory &m_memory;
        std::string *m_tap = nullptr;
        std::string m_line;   // console bytes since the last newline
        uint32_t m_column = 0; // fdputc's tab-stop column (a static in the firmware)
    };
}
