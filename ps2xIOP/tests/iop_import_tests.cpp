#include "emulator/core/iop_cpu.h"
#include "emulator/core/iop_kernel.h"
#include "emulator/core/iop_memory.h"
#include "emulator/imports/iop_cdvd.h"
#include "emulator/imports/iop_imports.h"
#include "emulator/imports/iop_intrman.h"
#include "emulator/imports/iop_loadcore.h"
#include "emulator/imports/iop_timrman.h"
#include "emulator/services/iop_rpc.h"
#include "ps2x/iop/iop_host.h"

#include <cstdint>
#include <chrono>
#include <ctime>
#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <string_view>

namespace
{
    using namespace ps2x::iop;
    using namespace ps2x::iop::detail;

    constexpr uint32_t kExportMagic = 0x41C00000u;
    constexpr int32_t kLibraryNotFound = -213;
    constexpr int32_t kIllegalLibrary = -214;

    class NullHost : public IopHost
    {
    public:
        bool readGuest(uint32_t, void *, size_t) const override { return false; }
        bool writeGuest(uint32_t, const void *, size_t) override { return false; }
        bool zeroGuest(uint32_t, size_t) override { return false; }
        bool normalizeGuestAddress(uint32_t, uint32_t &) const override { return false; }
        uint32_t allocateIopHandle(IopHandleKind) override { return 1u; }
        uint32_t allocateGuest(uint32_t, uint32_t) override { return 0u; }
        void freeGuest(uint32_t) override {}
        void audioCommand(uint32_t, uint32_t, GuestBuffer, GuestBuffer) override {}
        std::string hostPath(HostPathKind) const override { return {}; }
        std::string translateGuestPath(std::string_view path) const override { return std::string(path); }
        uint64_t openHostFile(std::string_view) override { return 0u; }
        bool hostFileSize(uint64_t, uint64_t &) const override { return false; }
        bool readHostFile(uint64_t, uint64_t, void *, size_t, size_t &) override { return false; }
        void closeHostFile(uint64_t) override {}
        int32_t memoryCard(const MemoryCardRequest &) override { return 0; }
        bool hasGuestFunction(uint32_t) const override { return false; }
        bool invokeGuestFunction(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t *) override { return false; }
        void log(LogLevel, std::string_view) override {}
    };

    class CdRootHost final : public NullHost
    {
    public:
        explicit CdRootHost(std::filesystem::path rootPath)
            : root(std::move(rootPath))
        {
        }

        std::string hostPath(HostPathKind kind) const override
        {
            return kind == HostPathKind::CdRoot ? root.string() : std::string{};
        }

    private:
        std::filesystem::path root;
    };

    class RecordingExecutor final : public IopGuestExecutor
    {
    public:
        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t,
                                      uint32_t,
                                      uint32_t,
                                      uint32_t gp) override
        {
            ++calls;
            lastAddress = address;
            lastArgument = a0;
            lastGp = gp;
            return callbackResult;
        }

        uint32_t callbackResult = 0u;
        uint32_t calls = 0u;
        uint32_t lastAddress = 0u;
        uint32_t lastArgument = 0u;
        uint32_t lastGp = 0u;
    };

    bool expect(bool condition, std::string_view message)
    {
        if (condition)
            return true;
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }

    bool testLoadcoreRebootLibraryMode()
    {
        IopMemory memory;
        IopImportRegistry imports(memory);
        IopLoadcore loadcore(memory, imports);

        IopCpuState cpu{};
        cpu.gpr[4] = 0u;
        cpu.gpr[5] = 2u;
        if (!expect(loadcore.dispatchImport(27u, cpu), "loadcore:27 was not handled") ||
            !expect(static_cast<int32_t>(cpu.gpr[2]) == kIllegalLibrary,
                    "loadcore:27 did not reject a null export table"))
            return false;

        constexpr uint32_t table = 0x1000u;
        memory.write32(table, kExportMagic);
        memory.write16(table + 8u, 0x0101u);
        memory.write16(table + 10u, 0x1234u);
        const char name[8] = {'t', 'e', 's', 't', 'l', 'i', 'b', '\0'};
        (void)memory.writeRam(table + 12u, name, sizeof(name));
        memory.write32(table + 20u, 0u);

        cpu = {};
        cpu.gpr[4] = table;
        cpu.gpr[5] = 2u;
        if (!expect(loadcore.dispatchImport(27u, cpu), "loadcore:27 rejected a valid export table") ||
            !expect(cpu.gpr[2] == 0u, "loadcore:27 returned an error for a valid export table") ||
            !expect(memory.read16(table + 10u) == 0x1232u,
                    "loadcore:27 did not replace only export mode bits 1 and 2"))
            return false;

        constexpr uint32_t invalidTable = 0x1100u;
        memory.write32(invalidTable, 0xDEADBEEFu);
        cpu = {};
        cpu.gpr[4] = invalidTable;
        cpu.gpr[5] = 6u;
        if (!expect(loadcore.dispatchImport(27u, cpu), "loadcore:27 did not consume an invalid-table call") ||
            !expect(static_cast<int32_t>(cpu.gpr[2]) == kLibraryNotFound,
                    "loadcore:27 returned the wrong invalid-table error"))
            return false;

        if (!expect(imports.registerExportTable(table), "test export table did not register"))
            return false;
        memory.write32(table, 0u);
        cpu = {};
        cpu.gpr[4] = table;
        cpu.gpr[5] = 6u;
        return expect(loadcore.dispatchImport(27u, cpu), "loadcore:27 rejected a registered table") &&
               expect(cpu.gpr[2] == 0u, "loadcore:27 returned an error for a registered table") &&
               expect(memory.read16(table + 10u) == 0x1236u,
                      "loadcore:27 did not update a registered table's mode");
    }

    bool pollEvent(IopKernel &kernel, int eventId, uint32_t bits, uint32_t resultAddress, int32_t expected)
    {
        IopCpuState cpu{};
        cpu.gpr[4] = static_cast<uint32_t>(eventId);
        cpu.gpr[5] = bits;
        cpu.gpr[6] = 0u; // WEF_AND
        cpu.gpr[7] = resultAddress;
        return expect(kernel.dispatchEventImport(11u, cpu), "PollEventFlag was not handled") &&
               expect(static_cast<int32_t>(cpu.gpr[2]) == expected, "PollEventFlag returned an unexpected result");
    }

    bool testCdvdSpecialControl()
    {
        NullHost host;
        IopMemory memory;
        IopKernel kernel(memory);
        kernel.reset();
        IopCdvd cdvd(host, memory, kernel);
        cdvd.reset();

        constexpr uint32_t param = 0x2000u;
        constexpr uint32_t eventResult = 0x2010u;
        IopCpuState cpu{};
        cpu.gpr[4] = static_cast<uint32_t>(-11); // sceCdSC: return cdvdman interrupt event flag
        cpu.gpr[5] = param;
        if (!expect(cdvd.dispatchImport(50u, cpu), "cdvdman:50 was not handled") ||
            !expect(static_cast<int32_t>(cpu.gpr[2]) > 0, "sceCdSC(-11) did not return a valid event flag"))
            return false;
        const int eventId = static_cast<int>(cpu.gpr[2]);

        if (!pollEvent(kernel, eventId, 0x29u, eventResult, 0) ||
            !expect(memory.read32(eventResult) == 0x29u, "cdvdman event flag did not start with bits 0x29"))
            return false;

        IopCpuState clear{};
        clear.gpr[4] = static_cast<uint32_t>(eventId);
        clear.gpr[5] = ~0x29u;
        if (!expect(kernel.dispatchEventImport(8u, clear), "ClearEventFlag was not handled") ||
            !pollEvent(kernel, eventId, 0x29u, eventResult, -418))
            return false;

        cpu = {};
        cpu.gpr[4] = 0x12345u;
        if (!expect(cdvd.dispatchImport(7u, cpu), "sceCdSeek was not handled") ||
            !pollEvent(kernel, eventId, 0x29u, eventResult, 0))
            return false;

        memory.write8(param, 0x30u);
        cpu = {};
        cpu.gpr[4] = static_cast<uint32_t>(-2);
        cpu.gpr[5] = param;
        if (!expect(cdvd.dispatchImport(50u, cpu), "sceCdSC(-2) was not handled") ||
            !expect(cpu.gpr[2] == 0x30u, "sceCdSC(-2) did not store the low-byte error"))
            return false;

        memory.write32(param, 0u);
        cpu = {};
        cpu.gpr[4] = static_cast<uint32_t>(-1);
        cpu.gpr[5] = param;
        if (!expect(cdvd.dispatchImport(50u, cpu), "sceCdSC(-1) was not handled") ||
            !expect(cpu.gpr[2] == 0u, "sceCdSC(-1) returned the wrong initial stream state") ||
            !expect(memory.read32(param) == 0x30u, "sceCdSC(-1) did not publish the last error"))
            return false;

        cpu = {};
        cpu.gpr[4] = 2u;
        cpu.gpr[5] = param;
        if (!expect(cdvd.dispatchImport(50u, cpu), "sceCdSC(2) was not handled") ||
            !expect(cpu.gpr[2] == 2u, "sceCdSC(2) did not update the stream state"))
            return false;

        cpu = {};
        cpu.gpr[4] = static_cast<uint32_t>(-1);
        cpu.gpr[5] = param;
        return expect(cdvd.dispatchImport(50u, cpu), "second sceCdSC(-1) was not handled") &&
               expect(cpu.gpr[2] == 2u, "sceCdSC(-1) did not preserve the stream state");
    }

    // sceCdReadClock (cdvdman:24, imported by MCMAN for save timestamps): sceCdCLOCK {stat, sec, min,
    // hour, pad, day, month, year} in BCD, from the host clock.
    bool testCdvdReadClock()
    {
        NullHost host;
        IopMemory memory;
        IopKernel kernel(memory);
        kernel.reset();
        IopCdvd cdvd(host, memory, kernel);
        cdvd.reset();

        constexpr uint32_t clock = 0x3000u;
        for (uint32_t i = 0; i < 8u; ++i)
            memory.write8(clock + i, 0xEEu);
        const std::time_t before = std::time(nullptr);
        IopCpuState cpu{};
        cpu.gpr[4] = clock;
        if (!expect(cdvd.dispatchImport(24u, cpu), "sceCdReadClock was not handled") ||
            !expect(cpu.gpr[2] == 1u, "sceCdReadClock did not return 1"))
            return false;
        const std::time_t after = std::time(nullptr);
        const auto fromBcd = [&](uint32_t offset, int lo, int hi, int &out)
        {
            const uint8_t b = memory.read8(clock + offset);
            out = (b >> 4) * 10 + (b & 0xF);
            return (b & 0xF) < 10 && (b >> 4) < 10 && out >= lo && out <= hi;
        };
        int sec = 0, min = 0, hour = 0, day = 0, month = 0, year = 0;
        if (!expect(memory.read8(clock) == 0u && memory.read8(clock + 4u) == 0u, "sceCdReadClock stat/pad not zero") ||
            !expect(fromBcd(1, 0, 59, sec) && fromBcd(2, 0, 59, min) && fromBcd(3, 0, 23, hour) &&
                        fromBcd(5, 1, 31, day) && fromBcd(6, 1, 12, month) && fromBcd(7, 0, 99, year),
                    "sceCdReadClock wrote a field that is not valid BCD in range"))
            return false;
        // the fields are the host's local time at the call (either side of it)
        bool matches = false;
        for (std::time_t t : {before, after})
        {
            std::tm local{};
#ifdef _WIN32
            localtime_s(&local, &t);
#else
            localtime_r(&t, &local);
#endif
            matches |= local.tm_sec == sec && local.tm_min == min && local.tm_hour == hour && local.tm_mday == day &&
                       local.tm_mon + 1 == month && local.tm_year % 100 == year;
        }
        if (!expect(matches, "sceCdReadClock is not the host's local time"))
            return false;
        cpu = {};
        return expect(cdvd.dispatchImport(24u, cpu), "sceCdReadClock(NULL) was not handled") &&
               expect(cpu.gpr[2] == 0u, "sceCdReadClock(NULL) did not fail");
    }

    bool testCdvdSearchFile()
    {
        const auto suffix = std::to_string(
            static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count()));
        const std::filesystem::path root =
            std::filesystem::temp_directory_path() / ("ps2x-iop-cdvd-search-" + suffix);
        const std::filesystem::path movieDirectory = root / "MOVIE";
        const std::filesystem::path moviePath = movieDirectory / "OPENING.PSS";
        std::error_code error;
        std::filesystem::create_directories(movieDirectory, error);
        if (!expect(!error, "could not create the temporary CD root"))
            return false;
        {
            std::ofstream movie(moviePath, std::ios::binary);
            movie.write("PSS!", 4);
        }

        CdRootHost host(root);
        IopMemory memory;
        IopKernel kernel(memory);
        kernel.reset();
        IopCdvd cdvd(host, memory, kernel);
        cdvd.reset();

        constexpr uint32_t resultAddress = 0x2400u;
        constexpr uint32_t pathAddress = 0x2480u;
        const char path[] = "cdrom0:\\movie\\opening.pss;1";
        (void)memory.writeRam(pathAddress, path, sizeof(path));

        IopCpuState cpu{};
        cpu.gpr[4] = resultAddress;
        cpu.gpr[5] = pathAddress;
        const bool handled = cdvd.dispatchImport(10u, cpu);
        const bool passed =
            expect(handled, "cdvdman:10 was not handled") &&
            expect(cpu.gpr[2] == 1u, "sceCdSearchFile did not find a case-insensitive ISO path") &&
            expect(memory.read32(resultAddress) >= 20u, "sceCdSearchFile returned an invalid LSN") &&
            expect(memory.read32(resultAddress + 4u) == 4u, "sceCdSearchFile returned the wrong size") &&
            expect(memory.readString(resultAddress + 8u, 16u) == "OPENING.PSS",
                   "sceCdSearchFile returned the wrong file name");

        std::filesystem::remove_all(root, error);
        return passed;
    }

    bool testTimrmanPeriodicCallback()
    {
        IopTimrman timrman;
        timrman.reset();
        IopCpuState cpu{};

        cpu.gpr[4] = 1u; // SYSCLK
        cpu.gpr[5] = 32u;
        cpu.gpr[6] = 1u;
        if (!expect(timrman.dispatchImport(4u, cpu, 100u), "AllocHardTimer was not handled") ||
            !expect(static_cast<int32_t>(cpu.gpr[2]) > 0, "AllocHardTimer did not allocate a 32-bit timer"))
            return false;
        const uint32_t timerId = cpu.gpr[2];

        cpu = {};
        cpu.gpr[4] = timerId;
        cpu.gpr[5] = 100u;
        cpu.gpr[6] = 0x12340u;
        cpu.gpr[7] = 0x45670u;
        cpu.gpr[28] = 0x89AB0u;
        if (!expect(timrman.dispatchImport(20u, cpu, 100u), "SetTimerHandler was not handled") ||
            !expect(cpu.gpr[2] == 0u, "SetTimerHandler failed"))
            return false;

        cpu = {};
        cpu.gpr[4] = timerId;
        cpu.gpr[5] = 1u;
        cpu.gpr[6] = 0u;
        cpu.gpr[7] = 1u;
        if (!expect(timrman.dispatchImport(22u, cpu, 100u), "SetupHardTimer was not handled") ||
            !expect(cpu.gpr[2] == 0u, "SetupHardTimer failed"))
            return false;

        cpu = {};
        cpu.gpr[4] = timerId;
        if (!expect(timrman.dispatchImport(23u, cpu, 100u), "StartHardTimer was not handled") ||
            !expect(cpu.gpr[2] == 0u, "StartHardTimer failed") ||
            !expect(timrman.nextEventCycle(1000u) == 200u, "timer compare was scheduled at the wrong cycle"))
            return false;

        RecordingExecutor executor;
        executor.callbackResult = 100u;
        timrman.serviceDue(199u, executor);
        if (!expect(executor.calls == 0u, "timer callback ran too early"))
            return false;
        timrman.serviceDue(200u, executor);
        return expect(executor.calls == 1u, "timer callback did not run") &&
               expect(executor.lastAddress == 0x12340u, "timer called the wrong handler") &&
               expect(executor.lastArgument == 0x45670u, "timer passed the wrong common argument") &&
               expect(executor.lastGp == 0x89AB0u, "timer callback lost the registering module GP") &&
               expect(timrman.nextEventCycle(1000u) == 300u, "timer callback return did not rearm compare");
    }

    // Return codes as Sony's INTRMAN gives them (ps2sdk kerr.h), measured by running the firmware module.
    bool testIntrmanReturnCodes()
    {
        constexpr int32_t kIllegalIntrcode = -101, kCpuDi = -102, kIntrDisable = -103;
        constexpr int32_t kFoundHandler = -104, kNotFoundHandler = -105;
        IopMemory memory;
        IopIntrman intrman(memory);
        intrman.reset();
        RecordingExecutor executor;
        IopCpuState cpu{};
        auto call = [&](uint16_t ordinal, uint32_t a0, uint32_t a1 = 0u, uint32_t a2 = 0u, uint32_t a3 = 0u)
        {
            cpu = {};
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            if (!intrman.dispatchImport(ordinal, cpu, executor))
                return 0x7FFFFFFF;
            return static_cast<int32_t>(cpu.gpr[2]);
        };

        if (!expect(call(4u, 9u, 0u, 0x1000u, 0u) == 0, "RegisterIntrHandler on a free line") ||
            !expect(call(4u, 9u, 0u, 0x1000u, 0u) == kFoundHandler, "RegisterIntrHandler twice") ||
            !expect(call(4u, 50u, 0u, 0x1000u, 0u) == kIllegalIntrcode, "RegisterIntrHandler on line 50") ||
            !expect(call(5u, 10u) == kNotFoundHandler, "ReleaseIntrHandler with no handler") ||
            !expect(call(5u, 9u) == 0, "ReleaseIntrHandler"))
            return false;

        constexpr uint32_t kRes = 0x1000u;
        if (!expect(call(6u, 9u) == 0, "EnableIntr(9)") ||
            !expect(call(7u, 9u, kRes) == 0, "DisableIntr(9) when enabled") ||
            !expect(memory.read32(kRes) == 9u, "DisableIntr stores the line in *res") ||
            !expect(call(7u, 9u, kRes) == kIntrDisable, "DisableIntr(9) when already disabled") ||
            !expect(static_cast<int32_t>(memory.read32(kRes)) == kIntrDisable, "*res when already disabled") ||
            !expect(call(6u, 39u) == kIllegalIntrcode, "EnableIntr on DMA line 39 (no such channel)"))
            return false;

        // DMA line 33 (channel 1) with its DICR (0x100) and DICR2 (0x200) completion-mode bits.
        if (!expect(call(6u, 33u | 0x300u) == 0, "EnableIntr(DMA line 33)") ||
            !expect(call(7u, 33u, kRes) == 0, "DisableIntr(DMA line 33)") ||
            !expect(memory.read32(kRes) == (33u | 0x300u), "DisableIntr keeps the DMA mode bits in *res"))
            return false;

        // CpuSuspendIntr reports KE_CPUDI when interrupts were already off.
        memory.setInterruptControl(1u);
        if (!expect(call(17u, kRes) == 0, "CpuSuspendIntr with interrupts on") ||
            !expect(memory.read32(kRes) == 1u, "CpuSuspendIntr stores the previous I_CTRL") ||
            !expect(call(17u, kRes) == kCpuDi, "CpuSuspendIntr with interrupts off") ||
            !expect(call(18u, 1u) == 0 && memory.interruptControl() == 1u, "CpuResumeIntr restores I_CTRL"))
            return false;

        // Reading I_CTRL returns it and clears it (PCSX2 IopHwRead).
        return expect(memory.read32(0x1F801078u) == 1u, "I_CTRL reads 1") &&
               expect(memory.read32(0x1F801078u) == 0u, "the read cleared I_CTRL");
    }
}

int main()
{
    if (!testLoadcoreRebootLibraryMode() || !testCdvdSpecialControl() || !testCdvdReadClock() || !testCdvdSearchFile() ||
        !testTimrmanPeriodicCallback() || !testIntrmanReturnCodes())
        return 1;
    std::cout << "ps2xIOP import tests passed\n";
    return 0;
}
