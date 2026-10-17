// The runtime as a whole: ELF loading, the slow memory paths and exception delivery, and a headless
// boot of a synthetic ELF through initialize(), loadELF() and run() (loader, IOP reset, EE kernel
// state, scheduler, present loop and shutdown working together).
#include "MiniTest.h"
#include "game_overrides.h"
#include "ps2_iop_host.h"
#include "ps2_recomp_abi.h"
#include "ps2_runtime.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "Kernel/Stubs/Pad.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

extern std::atomic<uint32_t> g_ps2xHostFrame;

namespace
{
    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;

    constexpr uint32_t kEntry = 0x00200000u;
    constexpr uint32_t kCodeWord = 0x12345678u;

    uint32_t excCode(const R5900Context &ctx) { return (ctx.cop0_cause & COP0_CAUSE_EXCCODE_MASK) >> 2; }

    void put16(std::vector<uint8_t> &b, size_t at, uint16_t v) { std::memcpy(b.data() + at, &v, 2); }
    void put32(std::vector<uint8_t> &b, size_t at, uint32_t v) { std::memcpy(b.data() + at, &v, 4); }

    struct Segment
    {
        uint32_t type = 1u; // PT_LOAD
        uint32_t vaddr = 0u;
        std::vector<uint8_t> data;
        uint32_t memsz = 0u;
        uint32_t flags = 0x5u; // PF_R | PF_X
    };

    // A 32-bit little-endian MIPS ET_EXEC image: header, program headers, then each segment's bytes.
    std::vector<uint8_t> makeElf(uint32_t entry, const std::vector<Segment> &segments)
    {
        const size_t phoff = 52u, phsize = 32u;
        size_t dataAt = phoff + phsize * segments.size();
        std::vector<uint8_t> b(dataAt, 0u);
        const uint8_t ident[] = {0x7F, 'E', 'L', 'F', 1, 1, 1};
        std::memcpy(b.data(), ident, sizeof(ident));
        put16(b, 16, 2u); // ET_EXEC
        put16(b, 18, 8u); // EM_MIPS
        put32(b, 20, 1u);
        put32(b, 24, entry);
        put32(b, 28, static_cast<uint32_t>(phoff));
        put16(b, 40, 52u);
        put16(b, 42, static_cast<uint16_t>(phsize));
        put16(b, 44, static_cast<uint16_t>(segments.size()));
        for (size_t i = 0; i < segments.size(); ++i)
        {
            const Segment &s = segments[i];
            const size_t ph = phoff + i * phsize;
            put32(b, ph + 0, s.type);
            put32(b, ph + 4, static_cast<uint32_t>(dataAt));
            put32(b, ph + 8, s.vaddr);
            put32(b, ph + 12, s.vaddr);
            put32(b, ph + 16, static_cast<uint32_t>(s.data.size()));
            put32(b, ph + 20, s.memsz);
            put32(b, ph + 24, s.flags);
            put32(b, ph + 28, 16u);
            b.insert(b.end(), s.data.begin(), s.data.end());
            dataAt += s.data.size();
        }
        return b;
    }

    std::vector<uint8_t> word(uint32_t v)
    {
        std::vector<uint8_t> b(4u);
        std::memcpy(b.data(), &v, 4u);
        return b;
    }

    // A scratch directory for ELF files; the runtime's I/O paths (which loadELF repoints at the ELF's
    // directory) are restored on exit.
    struct ElfDir
    {
        std::filesystem::path dir;
        PS2Runtime::IoPaths savedPaths = PS2Runtime::getIoPaths();

        ElfDir()
        {
            dir = std::filesystem::temp_directory_path() /
                  ("ps2x_boot_tests_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(dir);
        }
        ~ElfDir()
        {
            PS2Runtime::setIoPaths(savedPaths);
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
        std::string write(const std::string &name, const std::vector<uint8_t> &bytes) const
        {
            const std::filesystem::path p = dir / name;
            std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char *>(bytes.data()),
                                                     static_cast<std::streamsize>(bytes.size()));
            return p.string();
        }
    };

    void setEnv(const char *name, const char *value)
    {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    // The boot test's entry function: what it saw, for the test to check after run() returns.
    struct BootProbe
    {
        std::atomic<bool> entered{false};
        uint32_t sp = 0u, a0 = 1u, a1 = 1u;
        uint32_t codeWord = 0u;
        int threadId = 0;
        uint32_t framesSeen = 0u;
        bool storeVisible = false;
    };
    BootProbe *gBoot = nullptr;

    // Override descriptors that count their applications. Their ELF names match only the paths the
    // override test passes, so they never apply to another test's loadELF.
    int gOverrideHits[4] = {};
    void overrideByName(PS2Runtime &) { ++gOverrideHits[0]; }
    void overrideWrongEntry(PS2Runtime &) { ++gOverrideHits[1]; }
    void overrideByCrc(PS2Runtime &) { ++gOverrideHits[2]; }
    void overrideWrongCrc(PS2Runtime &) { ++gOverrideHits[3]; }

    void bootEntry(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        BootProbe &p = *gBoot;
        p.entered.store(true);
        p.sp = static_cast<uint32_t>(_mm_cvtsi128_si32(ctx->r[29]));
        p.a0 = static_cast<uint32_t>(_mm_cvtsi128_si32(ctx->r[4]));
        p.a1 = static_cast<uint32_t>(_mm_cvtsi128_si32(ctx->r[5]));
        p.codeWord = runtime->Load32(rdram, ctx, kEntry);
        runtime->Store32(rdram, ctx, 0x00300000u, 0xCAFEF00Du);
        uint32_t stored = 0u;
        std::memcpy(&stored, rdram + 0x00300000u, 4u);
        p.storeVisible = stored == 0xCAFEF00Du;
        ps2_syscalls::GetThreadId(rdram, ctx, runtime);
        p.threadId = _mm_cvtsi128_si32(ctx->r[2]);
        // Let the host loop present a few frames, then end the run as a window close would.
        const uint32_t start = g_ps2xHostFrame.load();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (g_ps2xHostFrame.load() < start + 3u && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        p.framesSeen = g_ps2xHostFrame.load() - start;
        runtime->requestStop();
    }
}

void register_ps2_runtime_boot_tests()
{
    MiniTest::Case("PS2RuntimeBoot", [](TestCase &tc)
    {
        tc.Run("an exception taken with EXL set keeps EPC and BD and goes to the common vector", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x1000u;
            ctx.cop0_status = COP0_STATUS_EXL;
            ctx.cop0_epc = 0x0BADu;
            ctx.cop0_cause = COP0_CAUSE_BD;
            runtime.SignalException(&ctx, EXCEPTION_TLB_REFILL);
            t.Equals(ctx.cop0_epc, 0x0BADu, "EPC is not rewritten while EXL is set");
            t.IsTrue((ctx.cop0_cause & COP0_CAUSE_BD) != 0u, "Cause.BD is not rewritten while EXL is set");
            t.Equals(excCode(ctx), static_cast<uint32_t>(EXCEPTION_TLB_REFILL), "Cause.ExcCode is still updated");
            t.Equals(ctx.pc, 0x80000180u, "a nested TLB refill takes the common vector, not the refill vector");
        });

        tc.Run("TLB refill vectors: 0x80000000, and 0xBFC00200 with BEV", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};
            ctx.pc = 0x1000u;
            runtime.SignalException(&ctx, EXCEPTION_TLB_REFILL);
            t.Equals(ctx.pc, 0x80000000u, "refill vector with BEV clear");
            t.Equals(ctx.cop0_epc, 0x1000u, "EPC = faulting pc");
            ctx = R5900Context{};
            ctx.pc = 0x1000u;
            ctx.cop0_status = COP0_STATUS_BEV;
            runtime.SignalException(&ctx, EXCEPTION_TLB_REFILL);
            t.Equals(ctx.pc, 0xBFC00200u, "refill vector with BEV set");
        });

        tc.Run("unaligned loads/stores raise AdEL/AdES with BadVAddr; aligned ones round-trip", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "memory initialize");
            uint8_t *rdram = runtime.memory().getRDRAM();

            struct Case
            {
                const char *name;
                uint32_t addr;
                bool store;
                int width;
            };
            const Case cases[] = {
                {"LH", 0x00100001u, false, 16}, {"LW", 0x00100002u, false, 32}, {"LD", 0x00100004u, false, 64},
                {"LQ", 0x00100008u, false, 128}, {"SH", 0x00100003u, true, 16}, {"SW", 0x00100001u, true, 32},
                {"SD", 0x00100004u, true, 64}, {"SQ", 0x00100004u, true, 128}};
            for (const Case &c : cases)
            {
                R5900Context ctx{};
                ctx.pc = 0x00101000u;
                if (c.store)
                {
                    switch (c.width)
                    {
                    case 16: runtime.Store16(rdram, &ctx, c.addr, 0x1234u); break;
                    case 32: runtime.Store32(rdram, &ctx, c.addr, 0x12345678u); break;
                    case 64: runtime.Store64(rdram, &ctx, c.addr, 0x1122334455667788ull); break;
                    default: runtime.Store128(rdram, &ctx, c.addr, _mm_set1_epi32(7)); break;
                    }
                }
                else
                {
                    uint64_t got = 1u;
                    switch (c.width)
                    {
                    case 16: got = runtime.Load16(rdram, &ctx, c.addr); break;
                    case 32: got = runtime.Load32(rdram, &ctx, c.addr); break;
                    case 64: got = runtime.Load64(rdram, &ctx, c.addr); break;
                    default: got = static_cast<uint64_t>(_mm_cvtsi128_si64(runtime.Load128(rdram, &ctx, c.addr))); break;
                    }
                    t.Equals(got, uint64_t{0}, std::string(c.name) + ": a faulting load returns 0");
                }
                const uint32_t want = c.store ? static_cast<uint32_t>(EXCEPTION_ADDRESS_ERROR_STORE)
                                              : static_cast<uint32_t>(EXCEPTION_ADDRESS_ERROR_LOAD);
                t.Equals(excCode(ctx), want, std::string(c.name) + ": Cause.ExcCode");
                t.Equals(ctx.cop0_badvaddr, c.addr, std::string(c.name) + ": BadVAddr = the faulting address");
                t.Equals(ctx.cop0_epc, 0x00101000u, std::string(c.name) + ": EPC = the faulting instruction");
                t.Equals(ctx.pc, 0x80000180u, std::string(c.name) + ": common vector");
                t.IsTrue((ctx.cop0_status & COP0_STATUS_EXL) != 0u, std::string(c.name) + ": EXL set");
            }

            R5900Context ctx{};
            ctx.pc = 0x00101000u;
            runtime.Store8(rdram, &ctx, 0x00100011u, 0xA5u);
            runtime.Store16(rdram, &ctx, 0x00100012u, 0xBEEFu);
            runtime.Store32(rdram, &ctx, 0x00100014u, 0xDEADBEEFu);
            runtime.Store64(rdram, &ctx, 0x00100018u, 0x0123456789ABCDEFull);
            runtime.Store128(rdram, &ctx, 0x00100020u, _mm_set_epi32(4, 3, 2, 1));
            t.Equals(static_cast<uint32_t>(runtime.Load8(rdram, &ctx, 0x00100011u)), 0xA5u, "SB/LB");
            t.Equals(static_cast<uint32_t>(runtime.Load16(rdram, &ctx, 0x00100012u)), 0xBEEFu, "SH/LH");
            t.Equals(runtime.Load32(rdram, &ctx, 0x00100014u), 0xDEADBEEFu, "SW/LW");
            t.Equals(runtime.Load64(rdram, &ctx, 0x00100018u), 0x0123456789ABCDEFull, "SD/LD");
            alignas(16) uint32_t q[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(q), runtime.Load128(rdram, &ctx, 0x00100020u));
            t.IsTrue(q[0] == 1u && q[1] == 2u && q[2] == 3u && q[3] == 4u, "SQ/LQ");
            t.Equals(runtime.Load32(rdram, &ctx, 0x20100014u), 0xDEADBEEFu, "the uncached mirror reads the same RAM");
            t.Equals(ctx.pc, 0x00101000u, "aligned accesses raise nothing");
            t.Equals(ctx.cop0_status & COP0_STATUS_EXL, 0u, "EXL stays clear");
        });

        tc.Run("the generated code's ps2x_abi entry points reach the runtime: loads, stores, exceptions", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "memory initialize");
            uint8_t *rdram = runtime.memory().getRDRAM();
            R5900Context ctx{};
            ctx.pc = 0x00101000u;
            PS2Runtime *rt = &runtime;
            ps2x_abi::Store8(rt, rdram, &ctx, 0x00100100u, 0x5Au);
            ps2x_abi::Store16(rt, rdram, &ctx, 0x00100102u, 0x1234u);
            ps2x_abi::Store32(rt, rdram, &ctx, 0x00100104u, 0x89ABCDEFu);
            ps2x_abi::Store64(rt, rdram, &ctx, 0x00100108u, 0x0011223344556677ull);
            ps2x_abi::Store128(rt, rdram, &ctx, 0x00100110u, _mm_set_epi32(4, 3, 2, 1));
            t.Equals(static_cast<uint32_t>(ps2x_abi::Load8(rt, rdram, &ctx, 0x00100100u)), 0x5Au, "8-bit");
            t.Equals(static_cast<uint32_t>(ps2x_abi::Load16(rt, rdram, &ctx, 0x00100102u)), 0x1234u, "16-bit");
            t.Equals(ps2x_abi::Load32(rt, rdram, &ctx, 0x00100104u), 0x89ABCDEFu, "32-bit");
            t.Equals(ps2x_abi::Load64(rt, rdram, &ctx, 0x00100108u), 0x0011223344556677ull, "64-bit");
            t.Equals(static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(ps2x_abi::Load128(rt, rdram, &ctx, 0x00100110u), 12))), 4u,
                     "128-bit");
            t.IsTrue(ps2x_abi::vu0Data(rt) == runtime.memory().getVU0Data(), "VU0 data memory");

            ps2x_abi::Load32(rt, rdram, &ctx, 0x00100101u);
            t.Equals(excCode(ctx), static_cast<uint32_t>(EXCEPTION_ADDRESS_ERROR_LOAD), "an unaligned load raises AdEL");
            R5900Context trap{};
            trap.pc = 0x00102000u;
            ps2x_abi::handleTrap(rt, rdram, &trap);
            t.Equals(excCode(trap), static_cast<uint32_t>(EXCEPTION_TRAP), "handleTrap raises Tr");
            R5900Context brk{};
            ps2x_abi::handleBreak(rt, rdram, &brk);
            t.Equals(excCode(brk), static_cast<uint32_t>(EXCEPTION_BREAKPOINT), "handleBreak raises Bp");
            R5900Context sig{};
            ps2x_abi::SignalException(rt, &sig, EXCEPTION_RESERVED_INSTRUCTION);
            t.Equals(excCode(sig), static_cast<uint32_t>(EXCEPTION_RESERVED_INSTRUCTION), "SignalException raises RI");
        });

        tc.Run("loadELF rejects malformed images", [](TestCase &t)
        {
            ElfDir dir;
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "memory initialize");
            const std::vector<Segment> ok = {{1u, kEntry, word(kCodeWord), 4u}};
            const std::vector<uint8_t> good = makeElf(kEntry, ok);

            auto mutate = [&](size_t at, uint8_t v)
            {
                std::vector<uint8_t> b = good;
                b[at] = v;
                return b;
            };
            struct Bad
            {
                const char *why;
                std::vector<uint8_t> bytes;
            };
            std::vector<Bad> bad = {
                {"shorter than an ELF header", std::vector<uint8_t>(good.begin(), good.begin() + 20)},
                {"bad magic", mutate(1, 'X')},
                {"64-bit class", mutate(4, 2)},
                {"big-endian", mutate(5, 2)},
                {"not MIPS", mutate(18, 3)},
                {"not ET_EXEC", mutate(16, 3)},
                {"program-header entries too small", mutate(42, 16)},
                {"program-header table past the end", mutate(44, 40)},
                {"filesz > memsz", makeElf(kEntry, {{1u, kEntry, word(1u), 2u}})},
                {"segment past RDRAM", makeElf(kEntry, {{1u, 0x01FFFFFEu, word(1u), 4u}})},
                {"no PT_LOAD segment", makeElf(kEntry, {{4u, kEntry, word(1u), 4u}})},
            };
            std::vector<uint8_t> truncated = good;
            truncated.resize(truncated.size() - 2u); // the segment's bytes run past the file
            bad.push_back({"segment bytes past the end of the file", truncated});
            t.IsFalse(runtime.loadELF((dir.dir / "missing.elf").string()), "a missing file is rejected");
            for (const Bad &b : bad)
                t.IsFalse(runtime.loadELF(dir.write("bad.elf", b.bytes)), std::string("rejected: ") + b.why);
            t.IsTrue(runtime.loadELF(dir.write("good.elf", good)), "the unmodified image loads");
        });

        tc.Run("loadELF places segments, zero-fills BSS, sets the entry pc, I/O paths and heap base", [](TestCase &t)
        {
            ElfDir dir;
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "memory initialize");
            uint8_t *rdram = runtime.memory().getRDRAM();
            std::memset(rdram + kEntry, 0xAA, 0x40);
            std::vector<uint8_t> text = word(kCodeWord);
            const std::vector<uint8_t> more = word(0x9ABCDEF0u);
            text.insert(text.end(), more.begin(), more.end());
            const std::vector<Segment> segs = {
                {1u, kEntry, text, 0x20u, 0x5u},
                {6u, 0x00280000u, word(0xFFFFFFFFu), 4u, 0x4u}, // PT_PHDR: not loaded
                {1u, 0x70000100u, word(0x55AA55AAu), 4u, 0x6u}, // scratchpad
                {1u, 0x00240000u, {}, 0x100u, 0x6u},            // pure BSS
            };
            std::memset(rdram + 0x00240000u, 0xAA, 0x100);
            const std::string path = dir.write("SLUS_000.00", makeElf(kEntry, segs));
            t.IsTrue(runtime.loadELF(path), "loads");

            uint32_t w0 = 0u, w1 = 0u, sp = 0u;
            std::memcpy(&w0, rdram + kEntry, 4u);
            std::memcpy(&w1, rdram + kEntry + 4u, 4u);
            std::memcpy(&sp, runtime.memory().getScratchpad() + 0x100u, 4u);
            t.Equals(w0, kCodeWord, "segment bytes copied");
            t.Equals(w1, 0x9ABCDEF0u, "segment bytes copied (second word)");
            bool zero = true;
            for (uint32_t i = 8u; i < 0x20u; ++i)
                zero = zero && rdram[kEntry + i] == 0u;
            t.IsTrue(zero, "memsz past filesz is zero-filled");
            t.Equals(static_cast<uint32_t>(rdram[kEntry + 0x20u]), 0xAAu, "nothing past memsz is touched");
            t.Equals(sp, 0x55AA55AAu, "a scratchpad segment lands in the scratchpad");
            uint32_t phdr = 0u;
            std::memcpy(&phdr, rdram + 0x00280000u, 4u);
            t.Equals(phdr, 0u, "non-PT_LOAD segments are not loaded");
            bool bss = true;
            for (uint32_t i = 0u; i < 0x100u; ++i)
                bss = bss && rdram[0x00240000u + i] == 0u;
            t.IsTrue(bss, "a segment with no file bytes is all zero");
            t.Equals(runtime.cpu().pc, kEntry, "pc = the ELF entry");
            const PS2Runtime::IoPaths &io = PS2Runtime::getIoPaths();
            t.Equals(io.elfDirectory, std::filesystem::path(path).parent_path(), "the ELF's directory");
            t.Equals(io.cdRoot, io.elfDirectory, "cdrom0: is served from the ELF's directory");
            t.Equals(io.mcRoot, io.elfDirectory / "mc0", "mc0: is the ELF directory's mc0");
            // RDRAM segments end at 0x240100; the guest heap starts past them plus a 4 KiB pad.
            t.Equals(runtime.guestHeapBase(), 0x00241100u, "the guest heap begins after the loaded image");
        });

        tc.Run("a synthetic ELF boots headless: entry thread runs, frames present, run() returns on stop", [](TestCase &t)
        {
            setEnv("PS2X_HEADLESS", "1");
            ElfDir dir;
            BootProbe probe;
            gBoot = &probe;
            {
                PS2Runtime runtime;
                t.IsTrue(runtime.initialize("ps2x_tests boot"), "initialize (headless)");
                const std::string path = dir.write("BOOT.ELF", makeElf(kEntry, {{1u, kEntry, word(kCodeWord), 0x100u}}));
                t.IsTrue(runtime.loadELF(path), "loadELF");
                t.IsTrue(runtime.registerFunction(kEntry, bootEntry), "register the entry function");
                runtime.run();
            }
            gBoot = nullptr;
            t.IsTrue(probe.entered.load(), "the entry function ran on the game thread");
            t.Equals(probe.sp, static_cast<uint32_t>(PS2_RAM_SIZE - 0x10u), "sp starts at the top of RAM");
            t.Equals(probe.a0, 0u, "a0 = 0 (argc)");
            t.Equals(probe.a1, 0u, "a1 = 0 (argv)");
            t.Equals(probe.codeWord, kCodeWord, "the loaded image is visible to guest loads");
            t.IsTrue(probe.storeVisible, "guest stores land in RDRAM");
            t.IsTrue(probe.threadId > 0, "GetThreadId gives the main thread's id");
            t.IsTrue(probe.framesSeen >= 3u, "the host loop presented frames while the guest ran");
        });
    });

    MiniTest::Case("PS2GameOverrides", [](TestCase &tc)
    {
        tc.Run("applyMatching filters on ELF name (any case), entry and CRC32", [](TestCase &t)
        {
            using namespace ps2_game_overrides;
            static bool registered = false;
            if (!registered)
            {
                registered = true;
                registerDescriptor({"by-name", "PS2X_OVR_TEST.ELF", 0u, 0u, overrideByName});
                registerDescriptor({"wrong-entry", "PS2X_OVR_TEST.ELF", 0x00300000u, 0u, overrideWrongEntry});
                static const AutoRegister byCrc({"by-crc", "PS2X_OVR_TEST.ELF", 0x00100008u, 0xABCD1234u, overrideByCrc});
                registerDescriptor({"wrong-crc", "", 0u, 0x0BADC0DEu, overrideWrongCrc});
                registerDescriptor({"no-apply", "PS2X_OVR_TEST.ELF", 0u, 0u, nullptr}); // ignored
            }
            PS2Runtime runtime;
            std::fill(std::begin(gOverrideHits), std::end(gOverrideHits), 0);
            applyMatching(runtime, "/games/x/ps2x_ovr_test.elf", 0x00100008u, 0xABCD1234u, true);
            t.Equals(gOverrideHits[0], 1, "a name-only descriptor matches the ELF basename in any case");
            t.Equals(gOverrideHits[1], 0, "a descriptor with another entry point does not apply");
            t.Equals(gOverrideHits[2], 1, "name + entry + CRC all matching applies (AutoRegister)");
            t.Equals(gOverrideHits[3], 0, "a CRC-only descriptor with another CRC does not apply");

            std::fill(std::begin(gOverrideHits), std::end(gOverrideHits), 0);
            applyMatching(runtime, "PS2X_OVR_TEST.ELF", 0x00100008u, 0xABCD1234u, false);
            t.Equals(gOverrideHits[0], 1, "no CRC needed: still applies");
            t.Equals(gOverrideHits[2], 0, "a CRC descriptor never applies when the CRC could not be computed");

            std::fill(std::begin(gOverrideHits), std::end(gOverrideHits), 0);
            applyMatching(runtime, "/games/x/OTHER.ELF", 0x00100008u, 0x0BADC0DEu, true);
            t.Equals(gOverrideHits[0] + gOverrideHits[1] + gOverrideHits[2], 0, "other ELF names match nothing named");
            t.Equals(gOverrideHits[3], 1, "a nameless CRC descriptor matches on CRC alone");
        });

        tc.Run("bindAddressHandler binds syscalls and stubs by name and rejects unknown names", [](TestCase &t)
        {
            using namespace ps2_game_overrides;
            PS2Runtime runtime;
            t.IsTrue(bindAddressHandler(runtime, 0x00110000u, "ExitThread"), "a syscall name binds");
            t.IsTrue(runtime.lookupFunction(0x00110000u) == &ps2_syscalls::ExitThread, "to the syscall handler");
            t.IsTrue(bindAddressHandler(runtime, 0x00110010u, "memcpy"), "a stub name binds");
            t.IsTrue(runtime.lookupFunction(0x00110010u) == &ps2_stubs::memcpy, "to the stub");
            t.IsFalse(bindAddressHandler(runtime, 0x00110020u, "NoSuchHandler"), "an unknown name is rejected");
            t.IsFalse(runtime.hasFunction(0x00110020u), "and binds nothing");
            t.IsFalse(bindAddressHandler(runtime, 0x02000000u, "memcpy"), "an address outside the function table is rejected");
        });
    });

    MiniTest::Case("PS2IopHostAdapter", [](TestCase &tc)
    {
        tc.Run("host paths follow the runtime I/O paths; guest paths map onto their device roots", [](TestCase &t)
        {
            using ps2x::iop::HostPathKind;
            ElfDir dir;
            PS2Runtime runtime;
            PS2IopHostAdapter host(runtime);
            PS2Runtime::IoPaths paths;
            paths.elfPath = dir.dir / "GAME.ELF";
            paths.hostRoot = dir.dir / "host";
            paths.cdRoot = dir.dir / "cd";
            paths.mcRoot = dir.dir / "mc";
            paths.cdImage = dir.dir / "game.iso";
            PS2Runtime::setIoPaths(paths);
            const PS2Runtime::IoPaths &io = PS2Runtime::getIoPaths();
            t.Equals(host.hostPath(HostPathKind::CdRoot), io.cdRoot.string(), "cd root");
            t.Equals(host.hostPath(HostPathKind::CdImage), io.cdImage.string(), "cd image");
            t.Equals(host.hostPath(HostPathKind::HostRoot), io.hostRoot.string(), "host root");
            t.Equals(host.hostPath(HostPathKind::MemoryCardRoot), io.mcRoot.string(), "memory card root");
            t.Equals(host.hostPath(HostPathKind::ElfDirectory), io.elfDirectory.string(), "ELF directory");
            t.Equals(io.elfDirectory, io.elfPath.parent_path(), "setIoPaths derives the ELF directory");

            t.Equals(host.translateGuestPath("cdrom0:\\DATA\\FILE.BIN;1"), (io.cdRoot / "DATA" / "FILE.BIN").string(),
                     "cdrom0: path, backslashes and version suffix");
            t.Equals(host.translateGuestPath("host0:sub/x.dat"), (io.hostRoot / "sub" / "x.dat").string(), "host0: path");
            t.Equals(host.translateGuestPath("mc0:/SAVE/icon.sys"), (io.mcRoot / "SAVE" / "icon.sys").string(), "mc0: path");
            t.Equals(host.translateGuestPath("rom0:IOPBTCONF"), std::string{}, "rom0: has no host file");
            t.Equals(host.translateGuestPath("host0:../../etc/passwd"), std::string{}, "paths may not climb out of their root");
        });

        tc.Run("host files: open, size, positioned reads, short reads at the end, close", [](TestCase &t)
        {
            ElfDir dir;
            PS2Runtime runtime;
            PS2IopHostAdapter host(runtime);
            const std::string path = dir.write("data.bin", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
            t.Equals(host.openHostFile(""), uint64_t{0}, "an empty path opens nothing");
            t.Equals(host.openHostFile((dir.dir / "missing.bin").string()), uint64_t{0}, "a missing file opens nothing");
            const uint64_t h = host.openHostFile(path);
            const uint64_t h2 = host.openHostFile(path);
            t.IsTrue(h != 0u && h2 != 0u && h != h2, "each open gets its own non-zero handle");
            uint64_t size = 0u;
            t.IsTrue(host.hostFileSize(h, size) && size == 10u, "size of the open file");
            uint8_t buf[8] = {};
            size_t got = 0u;
            t.IsTrue(host.readHostFile(h, 2u, buf, 4u, got), "a positioned read");
            t.IsTrue(got == 4u && buf[0] == 2u && buf[3] == 5u, "reads bytes 2..5");
            t.IsTrue(host.readHostFile(h, 7u, buf, 8u, got), "a read crossing the end succeeds");
            t.IsTrue(got == 3u && buf[0] == 7u && buf[2] == 9u, "with the bytes that exist");
            t.IsTrue(host.readHostFile(h, 0u, nullptr, 0u, got) && got == 0u, "a zero-size read with no buffer is fine");
            t.IsFalse(host.readHostFile(h, 0u, nullptr, 4u, got), "a sized read with no buffer fails");
            t.IsFalse(host.readHostFile(h + h2 + 100u, 0u, buf, 4u, got), "an unknown handle fails");
            host.closeHostFile(h);
            t.IsFalse(host.hostFileSize(h, size), "a closed handle is gone");
            t.IsFalse(host.readHostFile(h, 0u, buf, 1u, got), "reads on a closed handle fail");
            t.IsTrue(host.readHostFile(h2, 9u, buf, 1u, got) && got == 1u && buf[0] == 9u, "the other handle is unaffected");
            host.closeHostFile(h);  // closing twice is harmless
            host.closeHostFile(h2); // the destructor closes anything left otherwise
        });
    });

    MiniTest::Case("PS2CdAndLibcStubs", [](TestCase &t0)
    {
        auto setA = [](R5900Context &ctx, int reg, uint32_t v) { ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(v)); };
        auto v0 = [](const R5900Context &ctx) { return static_cast<int32_t>(_mm_cvtsi128_si32(ctx.r[2])); };

        t0.Run("sceCdIntToPos / sceCdPosToInt: LSN <-> BCD minute:second:sector, 150 sectors of lead-in", [&](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            const struct { uint32_t lsn; uint8_t m, s, f; } cases[] = {
                {0u, 0x00, 0x02, 0x00}, {74u, 0x00, 0x02, 0x74}, {75u, 0x00, 0x03, 0x00},
                {4350u, 0x01, 0x00, 0x00}, {123456u, 0x27, 0x28, 0x06}, {359849u, 0x79, 0x59, 0x74}};
            for (const auto &c : cases)
            {
                R5900Context ctx{};
                setA(ctx, 4, c.lsn);
                setA(ctx, 5, 0x1000u);
                ps2_stubs::sceCdIntToPos(rdram.data(), &ctx, &runtime);
                t.Equals(v0(ctx), 1, "sceCdIntToPos succeeds");
                const uint8_t *p = rdram.data() + 0x1000u;
                t.IsTrue(p[0] == c.m && p[1] == c.s && p[2] == c.f && p[3] == 0u,
                         "LSN " + std::to_string(c.lsn) + " -> BCD position");
                R5900Context back{};
                setA(back, 4, 0x1000u);
                ps2_stubs::sceCdPosToInt(rdram.data(), &back, &runtime);
                t.Equals(v0(back), static_cast<int32_t>(c.lsn), "sceCdPosToInt inverts it");
            }
        });

        t0.Run("sceCdReadClock: sceCdCLOCK in BCD (stat, second, minute, hour, pad, day, month, year) = local time", [&](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0xEEu);
            R5900Context ctx{};
            setA(ctx, 4, 0x2000u);
            const std::time_t before = std::time(nullptr);
            ps2_stubs::sceCdReadClock(rdram.data(), &ctx, &runtime);
            t.Equals(v0(ctx), 1, "succeeds");
            const uint8_t *c = rdram.data() + 0x2000u;
            auto bcd = [](uint8_t b) { return (b >> 4) * 10 + (b & 0xF); };
            bool validBcd = true;
            for (int i = 1; i < 8; ++i)
                validBcd = validBcd && (c[i] & 0xF) < 10 && (c[i] >> 4) < 10;
            t.IsTrue(validBcd, "every field is BCD");
            t.Equals(static_cast<uint32_t>(c[0]), 0u, "stat 0");
            t.Equals(static_cast<uint32_t>(c[4]), 0u, "pad 0");
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &before);
#else
            localtime_r(&before, &tm);
#endif
            t.Equals(bcd(c[7]), (tm.tm_year + 1900) % 100, "year");
            t.Equals(bcd(c[6]), tm.tm_mon + 1, "month");
            t.IsTrue(bcd(c[3]) < 24 && bcd(c[2]) < 60 && bcd(c[1]) < 60 && bcd(c[5]) >= 1 && bcd(c[5]) <= 31, "time fields in range");
            t.Equals(static_cast<uint32_t>(c[8]), 0xEEu, "nothing past the 8-byte record");
        });

        t0.Run("printf returns the rendered length; a null format is an error", [&](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            const char fmt[] = "%s=%d [%04x]";
            const char str[] = "lives";
            std::memcpy(rdram.data() + 0x3000u, fmt, sizeof(fmt));
            std::memcpy(rdram.data() + 0x3100u, str, sizeof(str));
            R5900Context ctx{};
            setA(ctx, 4, 0x3000u);
            setA(ctx, 5, 0x3100u);
            setA(ctx, 6, static_cast<uint32_t>(-3));
            setA(ctx, 7, 0xBEEFu);
            ps2_stubs::printf(rdram.data(), &ctx, &runtime);
            t.Equals(v0(ctx), static_cast<int32_t>(std::strlen("lives=-3 [beef]")), "length of \"lives=-3 [beef]\"");
            R5900Context bad{};
            ps2_stubs::printf(rdram.data(), &bad, &runtime);
            t.IsTrue(v0(bad) < 0, "printf(NULL) fails");
        });
    });

    MiniTest::Case("PS2PadScript", [](TestCase &tc)
    {
        tc.Run("PS2X_PAD_SCRIPT syntax: name@frame[+dur[*period]], unknown and malformed entries dropped", [](TestCase &t)
        {
            using namespace ps2_stubs;
            const auto script = parsePadScript("start@100+30,cross@200,l1@5+2*10,bogus@1+1,@7,triangle,select@0+0");
            t.Equals(script.size(), size_t{4}, "four valid entries");
            if (script.size() != 4u)
                return;
            t.IsTrue(script[0].mask == padButtonByName("start") && script[0].frame == 100u && script[0].duration == 30u && script[0].period == 0u,
                     "start@100+30");
            t.IsTrue(script[1].mask == padButtonByName("cross") && script[1].frame == 200u && script[1].duration == 1u,
                     "cross@200: one frame");
            t.IsTrue(script[2].mask == padButtonByName("l1") && script[2].period == 10u, "l1@5+2*10 repeats");
            t.IsTrue(script[3].mask == padButtonByName("select") && script[3].duration == 1u, "a zero duration means one frame");
            t.Equals(static_cast<uint32_t>(padButtonByName("nope")), 0u, "unknown names map to no button");
            uint16_t all = 0u;
            for (const char *name : {"select", "l3", "r3", "start", "up", "right", "down", "left", "l2", "r2", "l1", "r1",
                                     "triangle", "circle", "cross", "square"})
                all = static_cast<uint16_t>(all | padButtonByName(name));
            t.Equals(static_cast<uint32_t>(all), 0xFFFFu, "the sixteen names cover the sixteen button bits");
        });

        tc.Run("PS2X_PAD_SCRIPT timing: held from frame for dur frames, every period", [](TestCase &t)
        {
            using namespace ps2_stubs;
            const auto script = parsePadScript("start@100+30,l1@5+2*10");
            const uint16_t start = padButtonByName("start"), l1 = padButtonByName("l1");
            auto at = [&](uint32_t f) { return padScriptButtonsAt(script, f); };
            t.Equals(static_cast<uint32_t>(at(99) & start), 0u, "not before its frame");
            t.Equals(static_cast<uint32_t>(at(100) & start), static_cast<uint32_t>(start), "from its frame");
            t.Equals(static_cast<uint32_t>(at(129) & start), static_cast<uint32_t>(start), "for dur frames");
            t.Equals(static_cast<uint32_t>(at(130) & start), 0u, "and no longer");
            t.Equals(static_cast<uint32_t>(at(4) & l1), 0u, "l1 not before 5");
            t.IsTrue((at(5) & l1) && (at(6) & l1) && !(at(7) & l1) && (at(15) & l1) && (at(16) & l1) && !(at(17) & l1) && (at(1005) & l1),
                     "l1 held 2 of every 10 frames from frame 5, forever");
            t.Equals(static_cast<uint32_t>(at(115)), static_cast<uint32_t>(start | l1), "overlapping presses combine");
        });
    });
}
