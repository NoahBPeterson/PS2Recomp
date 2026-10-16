// Ground-truth harness support: run the original Sony IOP modules in our IOP interpreter.
//
// The firmware images are read at run time from the user's disc / BIOS dump and never committed:
//   - disc/MODULES/IOPRP300.IMG  (the IOP replacement image TC2 reboots the IOP with)
//   - a PS2 BIOS dump            (ROMDIR inside; INTRMAN, HEAPLIB, ...)
// Both are ROMDIR images: a table of {name[10], extinfo size u16, file size u32} entries starting
// with "RESET", each file following the previous one at a 16-byte boundary, offsets from file start.
#pragma once

#include "emulator/core/iop_cpu.h"
#include "emulator/core/iop_memory.h"
#include "emulator/imports/iop_imports.h"
#include "emulator/iop_emulator_const.h"
#include "emulator/services/iop_module_loader.h"
#include "ps2x/iop/iop_host.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace gt
{
    using namespace ps2x::iop;
    using namespace ps2x::iop::detail;

    // splitmix64: small, deterministic and identical on every platform.
    struct Rng
    {
        uint64_t state;
        explicit Rng(uint64_t seed) : state(seed) {}
        uint64_t next()
        {
            uint64_t z = (state += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }
        uint32_t u32() { return static_cast<uint32_t>(next() >> 32); }
        // uniform in [lo, hi]
        uint32_t range(uint32_t lo, uint32_t hi) { return lo + static_cast<uint32_t>(next() % (uint64_t(hi) - lo + 1u)); }
        bool chance(uint32_t percent) { return range(0, 99) < percent; }
        template <typename T>
        const T &pick(const std::vector<T> &v) { return v[range(0, static_cast<uint32_t>(v.size() - 1))]; }
    };

    inline uint64_t hashName(std::string_view s)
    {
        uint64_t h = 1469598103934665603ull;
        for (char c : s)
            h = (h ^ static_cast<uint8_t>(c)) * 1099511628211ull;
        return h;
    }

    inline std::vector<uint8_t> readFile(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return {};
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
    }

    // name -> file bytes (first occurrence wins; the BIOS has unnamed "-" padding entries).
    inline std::map<std::string, std::vector<uint8_t>> parseRomdir(const std::vector<uint8_t> &image)
    {
        std::map<std::string, std::vector<uint8_t>> files;
        static const uint8_t reset[6] = {'R', 'E', 'S', 'E', 'T', 0};
        size_t table = std::string::npos;
        for (size_t i = 0; i + 16 <= image.size() && i < 0x100000; i += 16)
            if (std::memcmp(image.data() + i, reset, 6) == 0)
            {
                table = i;
                break;
            }
        if (table == std::string::npos)
            return files;
        size_t offset = 0;
        for (size_t entry = table; entry + 16 <= image.size(); entry += 16)
        {
            if (image[entry] == 0)
                break;
            char name[11]{};
            std::memcpy(name, image.data() + entry, 10);
            uint32_t size = 0;
            std::memcpy(&size, image.data() + entry + 12, 4);
            if (offset + size > image.size())
                break;
            files.emplace(name, std::vector<uint8_t>(image.begin() + offset, image.begin() + offset + size));
            offset += (size + 15u) & ~size_t{15};
        }
        return files;
    }

    class CaptureHost final : public IopHost
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
        void log(LogLevel, std::string_view message) override { logs.emplace_back(message); }

        std::vector<std::string> logs;
    };

    constexpr uint32_t kSentinel = kCallReturnSentinel;

    // A bare IOP: memory + interpreter, modules loaded with our IRX loader, imports resolved
    // against the export tables the loaded modules registered themselves (loadcore:6), and
    // everything else routed to a stub callback that records the hit.
    class LleMachine
    {
    public:
        struct Library
        {
            std::string name;
            uint16_t version = 0;
            uint32_t table = 0;
            std::vector<uint32_t> functions;
        };
        struct Module
        {
            std::string name;
            uint32_t base = 0, size = 0, entry = 0, gp = 0;
            uint32_t startResult = 0;
        };
        enum class Exit
        {
            Return,
            Budget,
            Exception,
            BadPc,
        };
        struct RunResult
        {
            Exit exit = Exit::Return;
            uint32_t cause = 0;
            uint32_t epc = 0;
            uint32_t badAddress = 0;
            uint64_t instructions = 0;
        };
        // Returns true when it served the import (it sets $v0 etc.); the machine then returns to $ra.
        using StubHandler = std::function<bool(const IopImportCall &, IopCpuState &)>;

        IopMemory memory;
        IopCpuCore core{memory};
        IopImportRegistry decoder{memory};
        std::vector<Library> libraries;
        std::vector<Module> modules;
        std::map<std::string, uint64_t> stubHits; // unserved imports: "lib:ord" -> count
        StubHandler stub;
        uint32_t cursor = kModuleLoadBase;
        // Full-kernel mode: exceptions other than BREAK go to the guest's vector (the Sony kernel
        // handles syscalls itself); BREAK stops the run.
        bool deliverExceptions = false;
        std::map<uint32_t, uint64_t> exceptionCounts; // cause -> count (delivered ones)
        // Stop when the word at watchAddress becomes watchValue (0 address = off).
        uint32_t watchAddress = 0, watchValue = 0;

        static std::string key(std::string_view lib, uint32_t ord) { return std::string(lib) + ':' + std::to_string(ord); }

        const Library *findLibrary(std::string_view name) const
        {
            const Library *best = nullptr;
            for (const auto &lib : libraries)
                if (lib.name == name && (!best || lib.version > best->version))
                    best = &lib;
            return best;
        }

        uint32_t resolve(std::string_view name, uint32_t ordinal) const
        {
            const Library *lib = findLibrary(name);
            return lib && ordinal < lib->functions.size() ? lib->functions[ordinal] : 0u;
        }

        bool registerTable(uint32_t address)
        {
            const uint32_t phys = IopMemory::physicalAddress(address);
            if (memory.read32(phys) != 0x41C00000u)
                return false;
            Library lib;
            lib.table = phys;
            lib.version = memory.read16(phys + 8);
            lib.name = memory.readString(phys + 12, 8);
            for (uint32_t p = phys + 20; memory.read32(p) != 0u && lib.functions.size() < 1024; p += 4)
                lib.functions.push_back(memory.read32(p));
            libraries.push_back(std::move(lib));
            return true;
        }

        // Loads the module and runs its start entry point (argc=0, argv=NULL).
        std::optional<Module> loadAndStart(const std::string &name, const std::vector<uint8_t> &image, uint64_t budget = 5'000'000)
        {
            const IopImageLoadResult loaded = IopModuleLoader::load(image, memory, cursor);
            if (!loaded)
                return std::nullopt;
            cursor = loaded.nextModuleCursor;
            Module module{name, loaded.base, loaded.size, loaded.entry, loaded.gp, 0};
            IopCpuState cpu = initialCpu(module.entry, module.gp, kStackTop);
            const RunResult r = run(cpu, budget);
            if (r.exit != Exit::Return)
                return std::nullopt;
            module.startResult = cpu.gpr[2];
            modules.push_back(module);
            return module;
        }

        static constexpr uint32_t kStackTop = 0x001FE000u;

        static IopCpuState initialCpu(uint32_t pc, uint32_t gp, uint32_t sp)
        {
            IopCpuState cpu{};
            cpu.pc = pc;
            cpu.gpr[28] = gp;
            cpu.gpr[29] = sp;
            cpu.gpr[31] = kSentinel;
            cpu.cop0[12] = 0; // interrupts off: the harness never delivers any
            return cpu;
        }

        static void flushPendingLoad(IopCpuState &cpu)
        {
            if (cpu.pendingLoad)
            {
                if (cpu.pendingLoadReg != 0)
                    cpu.gpr[cpu.pendingLoadReg] = cpu.pendingLoadValue;
                cpu.pendingLoad = false;
            }
        }

        RunResult run(IopCpuState &cpu, uint64_t budget = 2'000'000)
        {
            RunResult result;
            while (true)
            {
                // Any address in the sentinel block is a harness return address.
                if ((cpu.pc & ~0xFu) == (kSentinel & ~0xFu) && !cpu.branchPending)
                {
                    flushPendingLoad(cpu);
                    result.exit = Exit::Return;
                    return result;
                }
                if (result.instructions >= budget)
                {
                    result.exit = Exit::Budget;
                    result.epc = cpu.pc;
                    return result;
                }
                if (IopMemory::physicalAddress(cpu.pc) >= IopMemory::RamSize || (cpu.pc & 3u))
                {
                    result.exit = Exit::BadPc;
                    result.epc = cpu.pc;
                    return result;
                }
                ++result.instructions;
                if (!cpu.branchPending)
                {
                    if (const auto call = decoder.decode(cpu.pc))
                    {
                        // An import stub. The jal's delay slot has executed; a load it issued is
                        // visible to the callee's first instructions on hardware.
                        flushPendingLoad(cpu);
                        const uint32_t target = resolve(call->library, call->ordinal);
                        if (target != 0u)
                        {
                            cpu.pc = target;
                            continue;
                        }
                        if (call->library == "loadcore" && call->ordinal == 6)
                        {
                            cpu.gpr[2] = registerTable(cpu.gpr[4]) ? 0u : 0xFFFFFFFFu;
                        }
                        else
                        {
                            if (!stub || !stub(*call, cpu))
                            {
                                ++stubHits[key(call->library, call->ordinal)]; // nothing serves it
                                cpu.gpr[2] = 0;
                            }
                        }
                        cpu.pc = cpu.gpr[31];
                        continue;
                    }
                }
                (void)core.executeInstruction(cpu);
                if (cpu.exception && deliverExceptions && ((cpu.cop0[13] >> 2) & 31u) != 9u)
                {
                    ++exceptionCounts[(cpu.cop0[13] >> 2) & 31u];
                    cpu.exception = false;
                }
                if (watchAddress != 0u && memory.read32(watchAddress) == watchValue)
                {
                    result.exit = Exit::Return;
                    result.epc = cpu.pc;
                    return result;
                }
                if (cpu.exception)
                {
                    result.exit = Exit::Exception;
                    result.cause = (cpu.cop0[13] >> 2) & 31u;
                    result.epc = cpu.cop0[14];
                    result.badAddress = cpu.cop0[8];
                    return result;
                }
            }
        }

        // Calls a guest function: a0..a3 in registers, the rest at sp+16 (o32).
        RunResult call(uint32_t function, const std::vector<uint32_t> &args, IopCpuState &cpu, uint64_t budget = 2'000'000)
        {
            for (size_t i = 0; i < args.size(); ++i)
            {
                if (i < 4)
                    cpu.gpr[4 + i] = args[i];
                else
                    memory.write32(cpu.gpr[29] + 16u + 4u * static_cast<uint32_t>(i - 4), args[i]);
            }
            cpu.pc = function;
            cpu.branchPending = false;
            return run(cpu, budget);
        }
    };

    inline const char *exitName(LleMachine::Exit e)
    {
        switch (e)
        {
        case LleMachine::Exit::Return:
            return "return";
        case LleMachine::Exit::Budget:
            return "instruction budget exhausted";
        case LleMachine::Exit::Exception:
            return "exception";
        case LleMachine::Exit::BadPc:
            return "pc left RAM";
        }
        return "?";
    }
}
