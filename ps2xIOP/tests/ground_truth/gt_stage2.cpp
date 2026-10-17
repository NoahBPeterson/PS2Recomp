// Stage 2 of the IOP ground-truth suite: the kernel modules. The original Sony kernel is booted in
// our interpreter the way IOPBOOT boots it (SYSMEM, then LOADCORE, which loads, links and starts
// every other module from a module list), with the IOPRP300 modules replacing their BIOS versions
// as UDNL does on TC2's IOP reboot. A generated test IRX goes last in the list: its start routine
// calls a sequence of exports and records the results. The same IRX is then loaded into our HLE
// IopEmulator and the two result sets are compared.
#include <cstdlib>
#include "gt_stage1.h"
#include "gt_diff.h"

#include "emulator/iop_emulator.h"

#include <iostream>
#include <memory>
#include <set>

namespace gt
{
    namespace
    {
        uint32_t rd32(const std::vector<uint8_t> &b, size_t o)
        {
            uint32_t v = 0;
            std::memcpy(&v, b.data() + o, 4);
            return v;
        }
        uint16_t rd16(const std::vector<uint8_t> &b, size_t o)
        {
            uint16_t v = 0;
            std::memcpy(&v, b.data() + o, 2);
            return v;
        }

        // IOPBOOT's job for SYSMEM and LOADCORE: copy the LOAD segment to textStart, relocate it
        // the way LOADCORE's loadrelelf does, and return the entry point and gp.
        bool placeIrx(IopMemory &mem, const std::vector<uint8_t> &elf, uint32_t textStart, uint32_t &entry, uint32_t &gp, uint32_t &end)
        {
            if (elf.size() < 52 || rd32(elf, 0) != 0x464C457Fu)
                return false;
            const uint32_t phoff = rd32(elf, 0x1C);
            const uint32_t shoff = rd32(elf, 0x20);
            const uint16_t shnum = rd16(elf, 0x30);
            const uint32_t iopmod = rd32(elf, phoff + 4);
            const uint32_t loadOff = rd32(elf, phoff + 32 + 4), fileSz = rd32(elf, phoff + 32 + 16), memSz = rd32(elf, phoff + 32 + 20);
            (void)mem.writeRam(textStart, elf.data() + loadOff, fileSz & ~3u);
            if (memSz > fileSz)
                (void)mem.zeroRam(textStart + fileSz, memSz - fileSz);
            entry = rd32(elf, iopmod + 4) + textStart;
            gp = rd32(elf, iopmod + 8) + textStart;
            end = textStart + memSz;
            for (uint16_t s = 1; s < shnum; ++s)
            {
                const size_t sh = shoff + 40u * s;
                if (rd32(elf, sh + 4) != 9u) // SHT_REL
                    continue;
                const uint32_t off = rd32(elf, sh + 16), size = rd32(elf, sh + 20), entsize = rd32(elf, sh + 36) ? rd32(elf, sh + 36) : 8u;
                const uint32_t n = size / entsize;
                for (uint32_t i = 0; i < n; ++i)
                {
                    const uint32_t roff = rd32(elf, off + i * entsize), type = rd32(elf, off + i * entsize + 4) & 0xFFu;
                    const uint32_t at = textStart + roff;
                    const uint32_t word = mem.read32(at);
                    switch (type)
                    {
                    case 2: // R_MIPS_32
                        mem.write32(at, word + textStart);
                        break;
                    case 4: // R_MIPS_26
                    {
                        const uint32_t target = textStart + ((roff & 0xF0000000u) | (4u * (word & 0x3FFFFFFu)));
                        mem.write32(at, (word & 0xFC000000u) | (target << 4 >> 6));
                        break;
                    }
                    case 5: // R_MIPS_HI16 (paired with the next entry's LO16)
                    {
                        const uint32_t lo = mem.read32(textStart + rd32(elf, off + (i + 1) * entsize));
                        const uint32_t full = textStart + static_cast<uint32_t>(static_cast<int16_t>(lo & 0xFFFFu)) + (word << 16);
                        mem.write32(at, (word & 0xFFFF0000u) | ((((full >> 15) + 1) >> 1) & 0xFFFFu));
                        break;
                    }
                    case 6: // R_MIPS_LO16
                        mem.write32(at, (word & 0xFFFF0000u) | ((textStart + word) & 0xFFFFu));
                        break;
                    default:
                        return false;
                    }
                }
            }
            return true;
        }
    }

    // The booted Sony kernel. Layout: SYSMEM at 0x830 (ModuleInfo at 0x800, "@800" in IOPBTCONF),
    // LOADCORE at the first free block SYSMEM reports, the module images in an IOPRP-style buffer
    // high in RAM (LOADCORE reserves it while booting), the boot stack at the top of RAM.
    struct KernelBoot
    {
        static constexpr uint32_t kBuffer = 0x00160000u;
        static constexpr uint32_t kBufferSize = 0x00060000u;
        static constexpr uint32_t kMailbox = kBuffer + kBufferSize - 16u; // test IRX: "done" + results address

        LleMachine m;
        std::vector<std::pair<std::string, uint32_t>> images; // name, address in the buffer
        std::string log;

        bool boot(const std::map<std::string, std::vector<uint8_t>> &bios, const std::map<std::string, std::vector<uint8_t>> &ioprp,
                  const std::vector<std::string> &order, const std::vector<uint8_t> *testIrx, uint64_t budget, bool devices = false)
        {
            m.deliverExceptions = true;
            // Root counters, INTC and interrupt delivery (T-0115 step 1): for tests that need time to
            // pass (delays, alarms), or for every test with GT_DEVICES=1.
            static const bool withDevices = std::getenv("GT_DEVICES") != nullptr;
            if (withDevices || devices)
                m.enableDevices();
            auto pick = [&](const std::string &name) -> const std::vector<uint8_t> *
            {
                if (auto it = ioprp.find(name); it != ioprp.end())
                    return &it->second;
                if (auto it = bios.find(name); it != bios.end())
                    return &it->second;
                return nullptr;
            };
            uint32_t cursor = kBuffer;
            std::vector<uint32_t> modData;
            for (const auto &name : order)
            {
                const auto *img = pick(name);
                if (!img)
                {
                    log += "no image for " + name + "\n";
                    return false;
                }
                (void)m.memory.writeRam(cursor, img->data(), img->size());
                images.emplace_back(name + (ioprp.count(name) ? " (IOPRP300)" : " (BIOS)"), cursor);
                modData.push_back(cursor);
                cursor = (cursor + static_cast<uint32_t>(img->size()) + 15u) & ~15u;
            }
            if (testIrx)
            {
                (void)m.memory.writeRam(cursor, testIrx->data(), testIrx->size());
                images.emplace_back("TESTIRX", cursor);
                modData.push_back(cursor);
                cursor = (cursor + static_cast<uint32_t>(testIrx->size()) + 15u) & ~15u;
            }
            modData.push_back(0);
            const uint32_t modDataAt = cursor;
            (void)m.memory.writeRam(modDataAt, modData.data(), modData.size() * 4);
            cursor += static_cast<uint32_t>(modData.size()) * 4u;
            if (cursor + 64u > kMailbox)
            {
                log += "module images do not fit the boot buffer\n";
                return false;
            }

            // SYSMEM
            uint32_t entry = 0, gp = 0, end = 0;
            const uint32_t sysmemText = 0x830u;
            if (!placeIrx(m.memory, *pick(order[0]), sysmemText, entry, gp, end))
            {
                log += "cannot place SYSMEM\n";
                return false;
            }
            IopCpuState cpu = LleMachine::initialCpu(entry, gp, 0x001FFF00u);
            cpu.gpr[4] = IopMemory::RamSize;
            auto r = m.run(cpu, 1'000'000);
            if (r.exit != LleMachine::Exit::Return || cpu.gpr[2] == 0u)
            {
                log += "SYSMEM start failed: " + std::string(exitName(r.exit)) + " v0=" + hex(cpu.gpr[2]) + "\n";
                return false;
            }
            const uint32_t firstFree = cpu.gpr[2];
            log += "SYSMEM at " + hex(sysmemText) + ", first free block " + hex(firstFree) + "\n";

            // LOADCORE: boot parameters, then it loads everything else and never returns.
            const uint32_t loadcoreText = firstFree + 0x30u;
            if (!placeIrx(m.memory, *pick(order[1]), loadcoreText, entry, gp, end))
            {
                log += "cannot place LOADCORE\n";
                return false;
            }
            const uint32_t params = cursor;
            const uint32_t words[8] = {2u /* MB */, 0u /* boot mode */, 0u, sysmemText, kBuffer, kBufferSize,
                                       static_cast<uint32_t>(modData.size() - 1), modDataAt};
            (void)m.memory.writeRam(params, words, sizeof(words));
            log += "LOADCORE at " + hex(loadcoreText) + "\n";
            cpu = LleMachine::initialCpu(entry, gp, 0x001FFF00u);
            cpu.gpr[4] = params;
            m.watchAddress = testIrx ? kMailbox : 0u;
            m.watchValue = 0x444F4E45u; // "DONE"
            r = m.run(cpu, budget);
            finalCpu = cpu;
            lastRun = r;
            if (withDevices && std::getenv("GT_DEVICES_REPORT"))
            {
                std::fprintf(stderr, "[gt-devices] cycle=%llu interrupts=%llu I_STAT=%x I_MASK=%x I_CTRL=%x SR=%x exit=%s\n",
                             static_cast<unsigned long long>(m.devices.cycle()), static_cast<unsigned long long>(m.interruptsTaken),
                             m.devices.iStat(), m.devices.iMask(), m.devices.iCtrl(), cpu.cop0[12], exitName(r.exit));
                for (int i = 0; i < 6; ++i)
                {
                    const auto &c = m.devices.counter(i);
                    std::fprintf(stderr, "[gt-devices]   counter %d mode=%04x rate=%u count=%llx target=%llx\n", i, c.mode, c.rate,
                                 static_cast<unsigned long long>(c.count), static_cast<unsigned long long>(c.target));
                }
            }
            return true;
        }

        IopCpuState finalCpu{};
        LleMachine::RunResult lastRun;

        // LOADCORE's resident-module list (ModuleInfo_t chain from SYSMEM's at 0x800).
        std::string modules() const
        {
            std::string out;
            uint32_t mi = 0x800u;
            for (int n = 0; mi != 0u && n < 64; ++n)
            {
                const uint32_t namePtr = m.memory.read32(mi + 4);
                out += "    module id=" + std::to_string(m.memory.read16(mi + 12)) + " " +
                       (namePtr ? m.memory.readString(namePtr, 64) : std::string("?")) + " text=" + hex(m.memory.read32(mi + 24)) +
                       " flags=" + hex(m.memory.read16(mi + 10)) + "\n";
                mi = m.memory.read32(mi);
            }
            return out;
        }
    };


    namespace
    {
        // ------------------------------------------------------------ test programs
        // One step = one import call. Arguments are constants, addresses in the program's data
        // area, or the v0 of an earlier step (object IDs).
        struct Arg
        {
            enum Kind : uint8_t
            {
                Const,
                Data,
                Result,
                DataWord, // the word at a data-area offset (a value an earlier call stored there)
            } kind = Const;
            uint32_t value = 0;
        };
        struct Step
        {
            std::string lib;
            uint16_t ord = 0;
            std::vector<Arg> args;
            bool returnsId = false;
            std::string text;
            // Stored into the program's data area right before the call: {data offset, value}.
            // Used for pointers inside parameter blocks (CreateThread's entry address).
            std::vector<std::pair<uint32_t, Arg>> stores = {};
            bool ignoreV0 = false; // void function: v0 is leftover
            // Run by the program's helper thread rather than the step thread (see Program::helperParam).
            bool helper = false;
        };
        // A case: steps plus its data area (inputs and out-buffers). Several cases share one IRX.
        struct Program
        {
            std::vector<Step> steps;
            std::vector<uint8_t> data;
            std::string note;
            // Out-buffer words that hold world-specific values: 0 ignore (stack/gp/context
            // pointers), 1 object ID (compared through the ID mapping), 2 an address in the probe
            // module (compared relative to the module base).
            std::map<uint32_t, uint8_t> wordKinds;
            // Data offset of a CreateThread parameter block (5 words; the entry is filled in at run
            // time) for a helper thread that runs the steps marked helper, or kNoHelper. The step
            // thread creates and starts it before the program's first step. Give it a lower priority
            // than the step thread (32): it then runs only while the step thread is blocked, which
            // is how a program makes a blocking call and has it released.
            uint32_t helperParam = kNoHelper;
            static constexpr uint32_t kNoHelper = 0xFFFFFFFFu;
        };

        struct Lib
        {
            const char *name;
            uint16_t version;
        };
        const Lib kLibs[] = {{"thbase", 0x101}, {"thevent", 0x101}, {"thsemap", 0x101}, {"intrman", 0x102}};

        Arg C(uint32_t v) { return {Arg::Const, v}; }
        Arg D(uint32_t off) { return {Arg::Data, off}; }
        Arg R(uint32_t step) { return {Arg::Result, step}; }
        Arg W(uint32_t off) { return {Arg::DataWord, off}; }

        // Encoders for the few instructions the test IRX needs.
        uint32_t iI(uint32_t op, uint32_t rs, uint32_t rt, uint32_t imm) { return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFFu); }
        uint32_t addiu(uint32_t rt, uint32_t rs, int32_t imm) { return iI(9, rs, rt, static_cast<uint32_t>(imm)); }
        uint32_t lui(uint32_t rt, uint32_t imm) { return iI(15, 0, rt, imm); }
        uint32_t ori(uint32_t rt, uint32_t rs, uint32_t imm) { return iI(13, rs, rt, imm); }
        uint32_t lw(uint32_t rt, int32_t off, uint32_t base) { return iI(0x23, base, rt, static_cast<uint32_t>(off)); }
        uint32_t sw(uint32_t rt, int32_t off, uint32_t base) { return iI(0x2B, base, rt, static_cast<uint32_t>(off)); }
        uint32_t jalr(uint32_t rs) { return (rs << 21) | (31u << 11) | 9u; }
        uint32_t jr(uint32_t rs) { return (rs << 21) | 8u; }
        uint32_t move(uint32_t rd, uint32_t rs) { return (rs << 21) | (rd << 11) | 0x21u; }
        constexpr uint32_t NOP = 0, BGEZAL_NEXT = 0x04110001u;

        // Builds an IRX (ELF type 0xFF80, .iopmod + one LOAD segment, no relocations: the code is
        // position independent and finds itself with bgezal). Layout of the LOAD segment:
        //   code | import tables | results (one word per step of every program) | data areas
        // At the end the start routine stores the results address at kMailbox+4 and "DONE" at
        // kMailbox, then returns 0 (resident).
        struct Irx
        {
            std::vector<uint8_t> file;
            uint32_t resultsOffset = 0;            // in the LOAD segment
            std::vector<uint32_t> stepBase;        // first result index of each program
            std::vector<uint32_t> dataOffset;      // data area of each program (LOAD offset)
            uint32_t totalSteps = 0;
        };

        Irx buildIrx(const std::vector<Program> &programs, uint32_t mailbox)
        {
            Irx irx;
            for (const auto &p : programs)
            {
                irx.stepBase.push_back(irx.totalSteps);
                irx.totalSteps += static_cast<uint32_t>(p.steps.size());
            }
            // import tables: every library, every ordinal 0..63 (simple and fixed)
            constexpr uint32_t kOrdinals = 64;
            const uint32_t tableWords = 5 + 2 * kOrdinals + 2;
            // The start routine creates and starts a thread (attr TH_C, priority 32, 4 KB stack)
            // whose entry runs the steps: the calls are made from thread context, as TC2's IRXs
            // make them, not from the boot-time module start.
            auto stubOffset = [&](uint32_t importsOff, const std::string &lib, uint32_t ord)
            {
                uint32_t libIndex = 0;
                while (lib != kLibs[libIndex].name)
                    ++libIndex;
                return importsOff + libIndex * tableWords * 4 + 20 + 8 * ord;
            };
            // Helper-thread entry offsets, taken from the previous (sizing) pass: code size does not
            // depend on them.
            std::vector<uint32_t> helperEntry(programs.size(), 0);
            auto emitCode = [&](uint32_t importsOff, uint32_t resultsOff, uint32_t threadParamOff, const std::vector<uint32_t> &dataOff,
                                std::vector<uint32_t> &code, uint32_t &threadEntryOff)
            {
                code.clear();
                auto here = [&]() { return static_cast<int32_t>(code.size() * 4); };
                auto call = [&](int32_t anchor, uint32_t stub)
                {
                    code.push_back(addiu(25, 16, static_cast<int32_t>(stub) - anchor));
                    code.push_back(jalr(25));
                    code.push_back(NOP);
                };
                // start(argc, argv)
                code.push_back(addiu(29, 29, -32));
                code.push_back(sw(31, 28, 29));
                code.push_back(sw(16, 24, 29));
                code.push_back(sw(17, 20, 29));
                code.push_back(BGEZAL_NEXT);
                code.push_back(NOP);
                int32_t anchor = here(); // s0 = address of this instruction
                code.push_back(move(16, 31));
                code.push_back(addiu(4, 16, static_cast<int32_t>(threadParamOff) - anchor));
                const size_t entryFixup = code.size();
                code.push_back(0); // addiu t0, s0, threadEntry - anchor (patched below)
                code.push_back(sw(8, 8, 4));
                call(anchor, stubOffset(importsOff, "thbase", 4)); // CreateThread(&param)
                code.push_back(move(4, 2));
                code.push_back(addiu(5, 0, 0));
                call(anchor, stubOffset(importsOff, "thbase", 6)); // StartThread(thid, 0)
                code.push_back(lw(31, 28, 29));
                code.push_back(lw(16, 24, 29));
                code.push_back(lw(17, 20, 29));
                code.push_back(addiu(2, 0, 0));
                code.push_back(jr(31));
                code.push_back(addiu(29, 29, 32));
                // thread entry
                threadEntryOff = static_cast<uint32_t>(here());
                code[entryFixup] = addiu(8, 16, static_cast<int32_t>(threadEntryOff) - anchor);
                code.push_back(addiu(29, 29, -32));
                code.push_back(sw(31, 28, 29));
                code.push_back(sw(16, 24, 29));
                code.push_back(sw(17, 20, 29));
                code.push_back(BGEZAL_NEXT);
                code.push_back(NOP);
                anchor = here();
                code.push_back(move(16, 31));
                code.push_back(addiu(17, 16, static_cast<int32_t>(resultsOff) - anchor));
                // One import call; its v0 goes to result slot `index`.
                auto emitStep = [&](size_t pi, const Step &st, uint32_t index)
                {
                    auto load = [&](const Arg &arg, uint32_t reg)
                    {
                        if (arg.kind == Arg::Const)
                        {
                            code.push_back(lui(reg, arg.value >> 16));
                            code.push_back(ori(reg, reg, arg.value & 0xFFFFu));
                        }
                        else if (arg.kind == Arg::Data)
                            code.push_back(addiu(reg, 16, static_cast<int32_t>(dataOff[pi] + arg.value) - anchor));
                        else if (arg.kind == Arg::DataWord)
                        {
                            code.push_back(lw(reg, static_cast<int32_t>(dataOff[pi] + arg.value) - anchor, 16));
                            code.push_back(NOP);
                        }
                        else
                        {
                            code.push_back(lw(reg, static_cast<int32_t>(4 * (irx.stepBase[pi] + arg.value)), 17));
                            code.push_back(NOP);
                        }
                    };
                    for (const auto &[off, arg] : st.stores)
                    {
                        load(arg, 8);
                        code.push_back(sw(8, static_cast<int32_t>(dataOff[pi] + off) - anchor, 16));
                    }
                    for (size_t a = 0; a < st.args.size() && a < 4; ++a)
                        load(st.args[a], 4 + static_cast<uint32_t>(a));
                    call(anchor, stubOffset(importsOff, st.lib, st.ord));
                    code.push_back(sw(2, static_cast<int32_t>(4 * index), 17));
                };
                for (size_t pi = 0; pi < programs.size(); ++pi)
                {
                    const Program &prog = programs[pi];
                    if (prog.helperParam != Program::kNoHelper)
                    {
                        code.push_back(addiu(4, 16, static_cast<int32_t>(dataOff[pi] + prog.helperParam) - anchor));
                        code.push_back(addiu(8, 16, static_cast<int32_t>(helperEntry[pi]) - anchor));
                        code.push_back(sw(8, 8, 4));
                        call(anchor, stubOffset(importsOff, "thbase", 4)); // CreateThread(&param)
                        code.push_back(move(4, 2));
                        code.push_back(addiu(5, 0, 0));
                        call(anchor, stubOffset(importsOff, "thbase", 6)); // StartThread(thid, 0)
                    }
                    for (size_t k = 0; k < prog.steps.size(); ++k)
                        if (!prog.steps[k].helper)
                            emitStep(pi, prog.steps[k], irx.stepBase[pi] + static_cast<uint32_t>(k));
                }
                code.push_back(lui(8, mailbox >> 16));
                code.push_back(ori(8, 8, mailbox & 0xFFFFu));
                code.push_back(sw(17, 4, 8));
                code.push_back(lui(9, 0x444F));
                code.push_back(ori(9, 9, 0x4E45));
                code.push_back(sw(9, 0, 8));
                code.push_back(lw(31, 28, 29));
                code.push_back(lw(16, 24, 29));
                code.push_back(lw(17, 20, 29));
                code.push_back(jr(31));
                code.push_back(addiu(29, 29, 32));
                // helper thread entries: the same frame and s0/s1 setup, then the helper's steps
                for (size_t pi = 0; pi < programs.size(); ++pi)
                {
                    const Program &prog = programs[pi];
                    if (prog.helperParam == Program::kNoHelper)
                        continue;
                    helperEntry[pi] = static_cast<uint32_t>(here());
                    code.push_back(addiu(29, 29, -32));
                    code.push_back(sw(31, 28, 29));
                    code.push_back(sw(16, 24, 29));
                    code.push_back(sw(17, 20, 29));
                    code.push_back(BGEZAL_NEXT);
                    code.push_back(NOP);
                    anchor = here();
                    code.push_back(move(16, 31));
                    code.push_back(addiu(17, 16, static_cast<int32_t>(resultsOff) - anchor));
                    for (size_t k = 0; k < prog.steps.size(); ++k)
                        if (prog.steps[k].helper)
                            emitStep(pi, prog.steps[k], irx.stepBase[pi] + static_cast<uint32_t>(k));
                    code.push_back(lw(31, 28, 29));
                    code.push_back(lw(16, 24, 29));
                    code.push_back(lw(17, 20, 29));
                    code.push_back(addiu(2, 0, 0));
                    code.push_back(jr(31));
                    code.push_back(addiu(29, 29, 32));
                }
            };
            std::vector<uint32_t> code;
            std::vector<uint32_t> dataOff(programs.size(), 0);
            uint32_t threadEntryOff = 0;
            emitCode(0, 0, 0, dataOff, code, threadEntryOff); // size pass
            const uint32_t importsOff = static_cast<uint32_t>(code.size() * 4);
            const uint32_t resultsOff = importsOff + static_cast<uint32_t>(std::size(kLibs)) * tableWords * 4;
            const uint32_t threadParamOff = resultsOff + 4 * irx.totalSteps;
            uint32_t cursor = threadParamOff + 20;
            for (size_t pi = 0; pi < programs.size(); ++pi)
            {
                cursor = (cursor + 15u) & ~15u;
                dataOff[pi] = cursor;
                cursor += static_cast<uint32_t>(programs[pi].data.size());
            }
            const uint32_t loadSize = (cursor + 15u) & ~15u;
            emitCode(importsOff, resultsOff, threadParamOff, dataOff, code, threadEntryOff);
            std::vector<uint8_t> seg(loadSize, 0);
            std::memcpy(seg.data(), code.data(), code.size() * 4);
            const uint32_t threadParam[5] = {0x02000000u /* TH_C */, 0u, 0u /* entry, set at run time */, 0x1000u, 32u};
            std::memcpy(seg.data() + threadParamOff, threadParam, sizeof(threadParam));
            for (size_t li = 0; li < std::size(kLibs); ++li)
            {
                std::vector<uint32_t> t = {0x41E00000u, 0u, kLibs[li].version, 0u, 0u};
                char name[8] = {};
                std::memcpy(name, kLibs[li].name, std::strlen(kLibs[li].name));
                std::memcpy(&t[3], name, 8);
                for (uint32_t o = 0; o < kOrdinals; ++o)
                {
                    t.push_back(0x03E00008u);
                    t.push_back(0x24000000u | o);
                }
                t.push_back(0);
                t.push_back(0);
                std::memcpy(seg.data() + importsOff + li * tableWords * 4, t.data(), t.size() * 4);
            }
            for (size_t pi = 0; pi < programs.size(); ++pi)
                std::memcpy(seg.data() + dataOff[pi], programs[pi].data.data(), programs[pi].data.size());
            irx.resultsOffset = resultsOff;
            irx.dataOffset = dataOff;

            // ELF: header, 2 program headers, iopmod, segment.
            const char modName[] = "gt_kernel_probe";
            std::vector<uint8_t> iopmod(28 + sizeof(modName), 0);
            const uint32_t im[6] = {0xFFFFFFFFu, 0u /* entry */, 0u /* gp */, loadSize, 0u, 0u};
            std::memcpy(iopmod.data(), im, sizeof(im));
            const uint16_t version = 0x0101;
            std::memcpy(iopmod.data() + 24, &version, 2);
            std::memcpy(iopmod.data() + 26, modName, sizeof(modName));
            const uint32_t iopmodOff = 52 + 64;
            const uint32_t segOff = (iopmodOff + static_cast<uint32_t>(iopmod.size()) + 15u) & ~15u;
            irx.file.assign(segOff + loadSize, 0);
            uint8_t *f = irx.file.data();
            const uint8_t ident[16] = {0x7F, 'E', 'L', 'F', 1, 1, 1, 0};
            std::memcpy(f, ident, 16);
            const uint16_t type = 0xFF80, machine = 8;
            std::memcpy(f + 16, &type, 2);
            std::memcpy(f + 18, &machine, 2);
            const uint32_t one = 1, phoff = 52;
            std::memcpy(f + 20, &one, 4);
            std::memcpy(f + 28, &phoff, 4);
            const uint16_t ehsize = 52, phentsize = 32, phnum = 2, shentsize = 40;
            std::memcpy(f + 40, &ehsize, 2);
            std::memcpy(f + 42, &phentsize, 2);
            std::memcpy(f + 44, &phnum, 2);
            std::memcpy(f + 46, &shentsize, 2);
            const uint32_t ph0[8] = {0x70000080u, iopmodOff, 0, 0, static_cast<uint32_t>(iopmod.size()), 0, 4, 4};
            const uint32_t ph1[8] = {1u, segOff, 0, 0, loadSize, loadSize, 7, 16};
            std::memcpy(f + 52, ph0, 32);
            std::memcpy(f + 84, ph1, 32);
            std::memcpy(f + iopmodOff, iopmod.data(), iopmod.size());
            std::memcpy(f + segOff, seg.data(), seg.size());
            return irx;
        }

        // What one side observed: v0 of every step and the bytes of every data area.
        struct Observed
        {
            bool ok = false;
            std::string error;
            std::vector<uint32_t> v0;
            std::vector<std::vector<uint8_t>> data;
            uint32_t segment = 0; // load address of the probe's LOAD segment
        };

        Observed collect(const Irx &irx, const std::vector<Program> &programs, const std::function<uint32_t(uint32_t)> &read32,
                         const std::function<uint8_t(uint32_t)> &read8, uint32_t mailbox)
        {
            Observed o;
            if (read32(mailbox) != 0x444F4E45u)
            {
                o.error = "the probe IRX did not finish";
                return o;
            }
            const uint32_t results = read32(mailbox + 4);
            const uint32_t segment = results - irx.resultsOffset;
            o.segment = segment;
            for (uint32_t i = 0; i < irx.totalSteps; ++i)
                o.v0.push_back(read32(results + 4 * i));
            for (size_t pi = 0; pi < programs.size(); ++pi)
            {
                std::vector<uint8_t> d(programs[pi].data.size());
                for (size_t b = 0; b < d.size(); ++b)
                    d[b] = read8(segment + irx.dataOffset[pi] + static_cast<uint32_t>(b));
                o.data.push_back(std::move(d));
            }
            o.ok = true;
            return o;
        }

        // Our HLE side: the IopEmulator the runtime uses, loading the probe IRX from a fake host file.
        class FileHost final : public IopHost
        {
        public:
            std::vector<uint8_t> file;
            std::vector<std::string> logs;
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
            uint64_t openHostFile(std::string_view path) override { return path == "probe.irx" ? 1u : 0u; }
            bool hostFileSize(uint64_t, uint64_t &size) const override
            {
                size = file.size();
                return true;
            }
            bool readHostFile(uint64_t, uint64_t offset, void *dst, size_t size, size_t &got) override
            {
                got = offset < file.size() ? std::min(size, file.size() - static_cast<size_t>(offset)) : 0;
                std::memcpy(dst, file.data() + offset, got);
                return true;
            }
            void closeHostFile(uint64_t) override {}
            int32_t memoryCard(const MemoryCardRequest &) override { return 0; }
            bool hasGuestFunction(uint32_t) const override { return false; }
            bool invokeGuestFunction(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t *) override { return false; }
            void log(LogLevel, std::string_view message) override { logs.emplace_back(message); }
        };

        Observed runHle(const Irx &irx, const std::vector<Program> &programs, uint32_t mailbox)
        {
            FileHost host;
            host.file = irx.file;
            IopEmulator emu(host);
            const ModuleLoadResult r = emu.loadModule("probe.irx", nullptr, 0);
            if (r.moduleId < 0)
            {
                Observed o;
                o.error = "HLE could not load the probe IRX";
                return o;
            }
            auto read32 = [&](uint32_t a)
            {
                uint32_t v = 0;
                (void)emu.readMemory(a, &v, 4);
                return v;
            };
            // The steps run in the probe's thread: let the HLE scheduler run it.
            for (int i = 0; i < 200 && read32(mailbox) != 0x444F4E45u; ++i)
                emu.runEeCycles(8u * 100000u);
            auto read8 = [&](uint32_t a)
            {
                uint8_t v = 0;
                (void)emu.readMemory(a, &v, 1);
                return v;
            };
            return collect(irx, programs, read32, read8, mailbox);
        }

        // ------------------------------------------------------------ generators
        void putWords(std::vector<uint8_t> &data, uint32_t off, const std::vector<uint32_t> &w)
        {
            if (data.size() < off + w.size() * 4)
                data.resize(off + w.size() * 4);
            std::memcpy(data.data() + off, w.data(), w.size() * 4);
        }
        uint32_t garbageBlock(Rng &r, std::vector<uint8_t> &data, uint32_t words)
        {
            const uint32_t off = (static_cast<uint32_t>(data.size()) + 3u) & ~3u;
            std::vector<uint32_t> w(words);
            for (auto &x : w)
                x = r.u32();
            putWords(data, off, w);
            return off;
        }

        // Semaphores: create a few, then a random mix of the other calls on live, deleted and bogus
        // IDs. WaitSema is only issued when a simple model says the count is positive (a wait that
        // blocks would park the boot thread; see the stage-2 notes in T-0014).
        Program semaProgram(Rng &r, uint32_t focus)
        {
            Program p;
            struct Model
            {
                size_t step;
                int count, max;
                bool live;
            };
            std::vector<Model> semas;
            auto create = [&]()
            {
                // SA_THFIFO 0, SA_THPRI 1, SA_IHTHPRI 0x100; anything else is illegal
                static const std::vector<uint32_t> attrs = {0u, 1u, 0x100u, 0x101u, 0u, 1u, 2u, 0x200u};
                const uint32_t attr = r.chance(90) ? r.pick(attrs) : r.u32();
                const int max = r.chance(85) ? static_cast<int>(r.range(1, 5)) : static_cast<int>(r.range(0, 2)) - 1;
                const int init = r.chance(85) ? static_cast<int>(r.range(0, static_cast<uint32_t>(std::max(max, 0)))) : static_cast<int>(r.range(0, 8)) - 2;
                const uint32_t param = garbageBlock(r, p.data, 4);
                putWords(p.data, param, {attr, r.u32(), static_cast<uint32_t>(init), static_cast<uint32_t>(max)});
                p.steps.push_back({"thsemap", 4, {D(param)}, true, "CreateSema(attr=" + hex(attr) + " init=" + std::to_string(init) + " max=" + std::to_string(max) + ")"});
                const bool legal = (attr & ~0x101u) == 0u;
                semas.push_back({p.steps.size() - 1, init, max, legal});
            };
            const uint32_t n = r.range(1, 3);
            for (uint32_t i = 0; i < n; ++i)
                create();
            const uint32_t ops = r.range(2, 24);
            for (uint32_t i = 0; i < ops; ++i)
            {
                // i-variants only when they are the focus: from thread context the firmware rejects them
                // (KE_ILLEGAL_CONTEXT), a known HLE gap that would otherwise mask every other result.
                uint32_t ord = r.chance(40) ? focus : r.pick(std::vector<uint32_t>{6, 8, 9, 11, 5});
                if (ord == 4)
                {
                    create();
                    continue;
                }
                Model &s = semas[r.range(0, static_cast<uint32_t>(semas.size() - 1))];
                const bool bogus = r.chance(8);
                // bogus IDs that are invalid in both ID schemes (ours are small integers, Sony's pointer-derived)
                const Arg id = bogus ? C(r.pick(std::vector<uint32_t>{0u, 0xFFFFFFFFu, 0x7FFFFFF1u, r.u32() | 0x40000000u})) : R(static_cast<uint32_t>(s.step));
                const std::string idText = bogus ? "bogus" : "sema" + std::to_string(s.step);
                if (ord == 8 && (bogus || !s.live || s.count <= 0 || focus == 7 || focus == 12))
                    ord = 9; // never block (with an i-variant focus the HLE state diverges, see above)
                switch (ord)
                {
                case 5:
                    p.steps.push_back({"thsemap", 5, {id}, false, "DeleteSema(" + idText + ")"});
                    if (!bogus)
                        s.live = false;
                    break;
                case 6:
                case 7:
                    p.steps.push_back({"thsemap", static_cast<uint16_t>(ord), {id}, false, std::string(ord == 6 ? "SignalSema(" : "iSignalSema(") + idText + ")"});
                    if (!bogus && s.live && ord == 6 && s.count < s.max)
                        ++s.count;
                    break;
                case 8:
                case 9:
                    p.steps.push_back({"thsemap", static_cast<uint16_t>(ord), {id}, false, std::string(ord == 8 ? "WaitSema(" : "PollSema(") + idText + ")"});
                    if (!bogus && s.live && s.count > 0)
                        --s.count;
                    break;
                default: // 11, 12: Refer
                {
                    const uint32_t info = garbageBlock(r, p.data, 8);
                    p.steps.push_back({"thsemap", static_cast<uint16_t>(ord), {id, D(info)}, false, std::string(ord == 11 ? "ReferSemaStatus(" : "iReferSemaStatus(") + idText + ")"});
                    break;
                }
                }
            }
            for (const Model &s : semas)
                if (s.live && r.chance(80))
                    p.steps.push_back({"thsemap", 5, {R(static_cast<uint32_t>(s.step))}, false, "DeleteSema(sema" + std::to_string(s.step) + ")"});
            p.note = std::to_string(p.steps.size()) + " calls";
            return p;
        }

        // Event flags, same scheme. WaitEventFlag is only issued when the model says it is satisfied.
        Program eventProgram(Rng &r, uint32_t focus)
        {
            Program p;
            struct Model
            {
                size_t step;
                uint32_t bits;
                bool live, multi;
            };
            std::vector<Model> flags;
            auto create = [&]()
            {
                // EA_SINGLE 0, EA_MULTI 2; anything else is illegal
                static const std::vector<uint32_t> attrs = {0u, 2u, 0u, 2u, 0x200u, 1u};
                const uint32_t attr = r.chance(90) ? r.pick(attrs) : r.u32();
                const uint32_t bits = r.chance(50) ? r.u32() : r.range(0, 15);
                const uint32_t param = garbageBlock(r, p.data, 3);
                putWords(p.data, param, {attr, r.u32(), bits});
                p.steps.push_back({"thevent", 4, {D(param)}, true, "CreateEventFlag(attr=" + hex(attr) + " bits=" + hex(bits) + ")"});
                flags.push_back({p.steps.size() - 1, bits, (attr & ~2u) == 0u, (attr & 2u) != 0});
            };
            const uint32_t n = r.range(1, 3);
            for (uint32_t i = 0; i < n; ++i)
                create();
            const uint32_t ops = r.range(2, 24);
            for (uint32_t i = 0; i < ops; ++i)
            {
                uint32_t ord = r.chance(40) ? focus : r.pick(std::vector<uint32_t>{6, 8, 10, 11, 13, 5});
                if (ord == 4)
                {
                    create();
                    continue;
                }
                Model &f = flags[r.range(0, static_cast<uint32_t>(flags.size() - 1))];
                const bool bogus = r.chance(8);
                const Arg id = bogus ? C(r.pick(std::vector<uint32_t>{0u, 0xFFFFFFFFu, 0x7FFFFFF1u, r.u32() | 0x40000000u})) : R(static_cast<uint32_t>(f.step));
                const std::string idText = bogus ? "bogus" : "evf" + std::to_string(f.step);
                const uint32_t bits = r.chance(20) ? 0u : (r.chance(50) ? r.u32() : (1u << r.range(0, 4)) | r.range(0, 3));
                static const std::vector<uint32_t> modes = {0u, 1u, 0x10u, 0x11u, 0x20u, 0x21u};
                const uint32_t mode = r.chance(90) ? r.pick(modes) : r.u32();
                const bool and_ = (mode & 1u) == 0u;
                const bool legal = (mode & ~0x11u) == 0u; // WEF_OR | WEF_CLEAR
                const bool satisfied = legal && bits != 0 && (and_ ? (f.bits & bits) == bits : (f.bits & bits) != 0);
                if (ord == 10 && (bogus || !f.live || !satisfied || focus == 7 || focus == 9 || focus == 14))
                    ord = 11; // never block (with an i-variant focus the HLE state diverges)
                switch (ord)
                {
                case 5:
                    p.steps.push_back({"thevent", 5, {id}, false, "DeleteEventFlag(" + idText + ")"});
                    if (!bogus)
                        f.live = false;
                    break;
                case 6:
                case 7:
                    p.steps.push_back({"thevent", static_cast<uint16_t>(ord), {id, C(bits)}, false, std::string(ord == 6 ? "SetEventFlag(" : "iSetEventFlag(") + idText + ", " + hex(bits) + ")"});
                    if (!bogus && f.live && ord == 6)
                        f.bits |= bits;
                    break;
                case 8:
                case 9:
                    p.steps.push_back({"thevent", static_cast<uint16_t>(ord), {id, C(bits)}, false, std::string(ord == 8 ? "ClearEventFlag(" : "iClearEventFlag(") + idText + ", " + hex(bits) + ")"});
                    if (!bogus && f.live && ord == 8)
                        f.bits &= bits;
                    break;
                case 10:
                case 11:
                {
                    const uint32_t res = garbageBlock(r, p.data, 1);
                    p.steps.push_back({"thevent", static_cast<uint16_t>(ord), {id, C(bits), C(mode), r.chance(10) ? C(0) : D(res)}, false,
                                       std::string(ord == 10 ? "WaitEventFlag(" : "PollEventFlag(") + idText + ", " + hex(bits) + ", mode " + hex(mode) + ")"});
                    if (!bogus && f.live && satisfied && (mode & 0x10u))
                        f.bits = 0; // WEF_CLEAR clears every bit
                    break;
                }
                default: // 13, 14
                {
                    const uint32_t info = garbageBlock(r, p.data, 8);
                    p.steps.push_back({"thevent", static_cast<uint16_t>(ord), {id, D(info)}, false, std::string(ord == 13 ? "ReferEventFlagStatus(" : "iReferEventFlagStatus(") + idText + ")"});
                    break;
                }
                }
            }
            for (const Model &f : flags)
                if (f.live && r.chance(80))
                    p.steps.push_back({"thevent", 5, {R(static_cast<uint32_t>(f.step))}, false, "DeleteEventFlag(evf" + std::to_string(f.step) + ")"});
            p.note = std::to_string(p.steps.size()) + " calls";
            return p;
        }

        // Threads. The probe thread runs at priority 32; created threads mostly get a lower
        // priority (they stay READY and never run), sometimes a higher one (StartThread then
        // switches to them; their entry returns at once, so they exit back to DORMANT).
        // Blocking and time-dependent calls (SleepThread, DelayThread, alarms, GetSystemTime) are
        // not probed: the LLE kernel gets no timer interrupts (see T-0115).
        Program threadProgram(Rng &r, uint32_t focus)
        {
            Program p;
            const uint32_t code = garbageBlock(r, p.data, 2);
            putWords(p.data, code, {0x03E00008u, 0u}); // entry: jr ra; nop
            p.steps.push_back({"thbase", 20, {}, true, "GetThreadId()"});
            const uint32_t self = 0;
            struct Model
            {
                size_t step;
                bool live;
            };
            std::vector<Model> threads;
            auto create = [&]()
            {
                static const std::vector<uint32_t> attrs = {0x02000000u, 0x02000000u, 0x02000000u, 0x01000000u, 0x02000008u, 0x02100000u, 0x02200000u, 0u, 0x80000000u};
                const uint32_t attr = r.chance(95) ? r.pick(attrs) : r.u32();
                static const std::vector<uint32_t> stacks = {0x200u, 0x400u, 0x800u, 0x1000u, 0x130u, 0x100u, 0x10u, 0u, 0x333u};
                const uint32_t stack = r.chance(90) ? r.pick(stacks) : r.range(0, 0x3000);
                const uint32_t prio = r.chance(70) ? r.range(40, 100) : r.chance(50) ? r.range(8, 31) : r.pick(std::vector<uint32_t>{0u, 1u, 126u, 127u, 128u, 0xFFFFFFFFu});
                const uint32_t param = garbageBlock(r, p.data, 5);
                const bool entryOk = r.chance(95);
                putWords(p.data, param, {attr, r.u32(), 0u, stack, prio});
                // the entry address is the probe's code word: patched in through a Data argument
                p.steps.push_back({"thbase", 4, {D(param)}, true,
                                   "CreateThread(attr=" + hex(attr) + " stack=" + hex(stack) + " prio=" + std::to_string(int32_t(prio)) + (entryOk ? "" : " entry misaligned") + ")"});
                // a bad entry is misaligned (KE_ILLEGAL_ENTRY); entry 0 would be accepted and crash when started
                p.steps.back().stores = {{param + 8, entryOk ? D(code) : D(code + 2)}};
                p.wordKinds[param + 8] = 2; // the entry address lives in the probe module
                threads.push_back({p.steps.size() - 1, true});
            };
            const uint32_t n = r.range(1, 3);
            for (uint32_t i = 0; i < n; ++i)
                create();
            const uint32_t ops = r.range(2, 20);
            for (uint32_t i = 0; i < ops; ++i)
            {
                uint32_t ord = r.chance(40) ? focus : r.pick(std::vector<uint32_t>{6, 5, 10, 14, 22, 25, 27, 29, 31, 18, 42, 39, 40});
                if (ord == 4)
                {
                    create();
                    continue;
                }
                Model &t = threads[r.range(0, static_cast<uint32_t>(threads.size() - 1))];
                const bool bogus = r.chance(6);
                const bool useSelf = r.chance(10);
                Arg id = bogus ? C(r.pick(std::vector<uint32_t>{0xFFFFFFFFu, 0x7FFFFFF1u, r.u32() | 0x40000000u}))
                               : useSelf ? R(self) : R(static_cast<uint32_t>(t.step));
                if (r.chance(5))
                    id = C(0); // 0 = the calling thread for the calls that accept it
                const std::string idText = id.kind == Arg::Const ? hex(id.value) : useSelf ? "self" : "th" + std::to_string(t.step);
                switch (ord)
                {
                case 6:
                    p.steps.push_back({"thbase", 6, {id, C(r.u32())}, false, "StartThread(" + idText + ")"});
                    break;
                case 14:
                {
                    const uint32_t prio = r.chance(80) ? r.range(33, 110) : r.pick(std::vector<uint32_t>{0u, 1u, 20u, 126u, 127u, 0xFFFFFFFFu});
                    p.steps.push_back({"thbase", 14, {id, C(prio)}, false, "ChangeThreadPriority(" + idText + ", " + std::to_string(int32_t(prio)) + ")"});
                    break;
                }
                case 22:
                {
                    const uint32_t info = garbageBlock(r, p.data, 17);
                    p.steps.push_back({"thbase", 22, {id, D(info)}, false, "ReferThreadStatus(" + idText + ")"});
                    p.wordKinds[info + 3 * 4] = 2;  // entry
                    p.wordKinds[info + 4 * 4] = 0;  // stack
                    p.wordKinds[info + 6 * 4] = 0;  // gp
                    p.wordKinds[info + 10 * 4] = 1; // waitId
                    p.wordKinds[info + 12 * 4] = 0; // register context
                    break;
                }
                case 39:
                {
                    static const std::vector<uint32_t> usecs = {0u, 1u, 999u, 1000u, 1000000u, 0x7FFFFFFFu, 0xFFFFFFFFu};
                    const uint32_t usec = r.chance(40) ? r.pick(usecs) : r.range(0, 10000000);
                    const uint32_t clk = garbageBlock(r, p.data, 2);
                    p.steps.push_back({"thbase", 39, {C(usec), D(clk)}, false, "USec2SysClock(" + std::to_string(usec) + ")"});
                    p.steps.back().ignoreV0 = true; // void
                    break;
                }
                case 40:
                {
                    const uint32_t clk = garbageBlock(r, p.data, 4);
                    putWords(p.data, clk, {r.chance(50) ? r.range(0, 1000000) : r.u32(), r.chance(70) ? 0u : r.range(0, 100)});
                    const uint32_t sec = garbageBlock(r, p.data, 1), usec = garbageBlock(r, p.data, 1);
                    p.steps.push_back({"thbase", 40, {D(clk), D(sec), D(usec)}, false, "SysClock2USec(...)"});
                    p.steps.back().ignoreV0 = true; // void
                    break;
                }
                case 42:
                    p.steps.push_back({"thbase", 42, {}, false, "GetThreadCurrentPriority()"});
                    break;
                case 20:
                    p.steps.push_back({"thbase", 20, {}, true, "GetThreadId()"});
                    break;
                default: // 5, 10, 18, 25, 27, 29, 31: one thread ID argument
                {
                    static const std::map<uint32_t, const char *> names = {{5, "DeleteThread"}, {10, "TerminateThread"}, {18, "ReleaseWaitThread"},
                                                                           {25, "WakeupThread"}, {27, "CancelWakeupThread"}, {29, "SuspendThread"}, {31, "ResumeThread"}};
                    const auto nm = names.find(ord);
                    p.steps.push_back({"thbase", static_cast<uint16_t>(ord), {id}, false, std::string(nm != names.end() ? nm->second : "?") + "(" + idText + ")"});
                    if (ord == 5 && !bogus && !useSelf)
                        t.live = false;
                    break;
                }
                }
            }
            for (const Model &t : threads)
                if (t.live && r.chance(80))
                {
                    p.steps.push_back({"thbase", 10, {R(static_cast<uint32_t>(t.step))}, false, "TerminateThread(th" + std::to_string(t.step) + ")"});
                    p.steps.push_back({"thbase", 5, {R(static_cast<uint32_t>(t.step))}, false, "DeleteThread(th" + std::to_string(t.step) + ")"});
                }
            p.note = std::to_string(p.steps.size()) + " calls";
            return p;
        }

        // Interrupt manager: handler registration, IRQ enable/disable and the CPU interrupt state.
        // Handlers are only registered and lines only enabled on IRQs no device raises in either
        // world (a raised line would call the fake handler), and every program ends with the CPU
        // interrupts enabled and its handlers released.
        Program intrProgram(Rng &r, uint32_t focus)
        {
            Program p;
            static const std::vector<uint32_t> quiet = {12u, 13u, 22u, 23u};
            // DMA completion lines of channels nothing transfers on here (PIO, GPU OTC, DEV9, SIO2);
            // bits 8/9 of EnableIntr's argument pick the DICR/DICR2 completion modes.
            static const std::vector<uint32_t> quietDma = {37u, 38u, 41u, 44u, 45u};
            static const std::vector<uint32_t> bogusIrq = {46u, 63u, 64u, 0xFFu, 0x10Du, 0xFFFFFFFFu, 0x80000000u};
            std::set<uint32_t> registered;
            const uint32_t handler = garbageBlock(r, p.data, 2);
            auto irqArg = [&](bool allowBogus) { return allowBogus && r.chance(20) ? r.pick(bogusIrq) : r.pick(quiet); };
            const uint32_t ops = r.range(3, 20);
            for (uint32_t i = 0; i < ops; ++i)
            {
                const uint32_t ord = r.chance(40) ? focus : r.pick(std::vector<uint32_t>{4, 5, 6, 7, 8, 17, 23});
                switch (ord)
                {
                case 4:
                {
                    const uint32_t irq = irqArg(true), mode = r.range(0, 4);
                    p.steps.push_back({"intrman", 4, {C(irq), C(mode), D(handler), C(r.u32())}, false,
                                       "RegisterIntrHandler(" + std::to_string(static_cast<int32_t>(irq)) + ", mode " + std::to_string(mode) + ")"});
                    registered.insert(irq);
                    break;
                }
                case 5:
                {
                    const uint32_t irq = irqArg(true);
                    p.steps.push_back({"intrman", 5, {C(irq)}, false, "ReleaseIntrHandler(" + std::to_string(static_cast<int32_t>(irq)) + ")"});
                    registered.erase(irq);
                    break;
                }
                case 6:
                {
                    const uint32_t irq = r.chance(30) ? r.pick(quietDma) | (r.range(0, 3) << 8) : r.pick(quiet);
                    p.steps.push_back({"intrman", 6, {C(irq)}, false, "EnableIntr(" + hex(irq) + ")"});
                    break;
                }
                case 7:
                {
                    const uint32_t irq = r.chance(25) ? r.pick(quietDma) : irqArg(true), res = garbageBlock(r, p.data, 1);
                    p.steps.push_back({"intrman", 7, {C(irq), r.chance(15) ? C(0) : D(res)}, false,
                                       "DisableIntr(" + std::to_string(static_cast<int32_t>(irq)) + ", &res)"});
                    break;
                }
                case 8: // CpuDisableIntr, possibly twice, then CpuEnableIntr
                    p.steps.push_back({"intrman", 8, {}, false, "CpuDisableIntr()"});
                    if (r.chance(50))
                        p.steps.push_back({"intrman", 8, {}, false, "CpuDisableIntr() again"});
                    if (r.chance(50))
                        p.steps.push_back({"intrman", 23, {}, false, "QueryIntrContext()"});
                    p.steps.push_back({"intrman", 9, {}, false, "CpuEnableIntr()"});
                    break;
                case 17: // CpuSuspendIntr (nested or not), then CpuResumeIntr with the stored states
                {
                    const uint32_t s1 = garbageBlock(r, p.data, 1);
                    p.steps.push_back({"intrman", 17, {D(s1)}, false, "CpuSuspendIntr(&s1)"});
                    if (r.chance(50))
                    {
                        const uint32_t s2 = garbageBlock(r, p.data, 1);
                        p.steps.push_back({"intrman", 17, {D(s2)}, false, "CpuSuspendIntr(&s2) nested"});
                        p.steps.push_back({"intrman", 18, {W(s2)}, false, "CpuResumeIntr(s2)"});
                    }
                    if (r.chance(30))
                        p.steps.push_back({"intrman", 17, {C(0)}, false, "CpuSuspendIntr(NULL)"});
                    p.steps.push_back({"intrman", 18, {W(s1)}, false, "CpuResumeIntr(s1)"});
                    break;
                }
                default: // 23
                    p.steps.push_back({"intrman", 23, {}, false, "QueryIntrContext()"});
                    break;
                }
            }
            std::vector<uint32_t> lines = quiet;
            lines.insert(lines.end(), quietDma.begin(), quietDma.end());
            for (uint32_t irq : lines)
            {
                const uint32_t res = garbageBlock(r, p.data, 1);
                p.steps.push_back({"intrman", 7, {C(irq), D(res)}, false, "DisableIntr(" + std::to_string(irq) + ", &res) cleanup"});
            }
            for (uint32_t irq : registered)
                p.steps.push_back({"intrman", 5, {C(irq)}, false, "ReleaseIntrHandler(" + std::to_string(static_cast<int32_t>(irq)) + ") cleanup"});
            p.steps.push_back({"intrman", 9, {}, false, "CpuEnableIntr() cleanup"});
            p.note = std::to_string(p.steps.size()) + " calls";
            return p;
        }

        // Blocking calls and what releases them. The step thread blocks (WaitSema, WaitEventFlag,
        // SleepThread, DelayThread) and a helper thread, at a lower priority, runs only then: it takes
        // a turn from the "go" semaphore and performs the release (SignalSema, SetEventFlag,
        // WakeupThread, DeleteSema, ReleaseWaitThread). The step thread, being higher priority,
        // resumes at once. Alarms are set with short or long clocks and cancelled before or after
        // they fire (the handler is two instructions in the data area: return 0).
        Program waitProgram(Rng &r, uint32_t focus)
        {
            Program p;
            auto helper = [&](Step st)
            {
                st.helper = true;
                st.text = "[helper] " + st.text;
                p.steps.push_back(std::move(st));
            };
            auto sema = [&](uint32_t init, uint32_t max, const std::string &name)
            {
                const uint32_t param = garbageBlock(r, p.data, 4);
                putWords(p.data, param, {r.chance(50) ? 1u : 0u, r.u32(), init, max});
                p.steps.push_back({"thsemap", 4, {D(param)}, true, "CreateSema(" + name + ")"});
                return static_cast<uint32_t>(p.steps.size() - 1);
            };
            p.helperParam = garbageBlock(r, p.data, 5);
            putWords(p.data, p.helperParam, {0x02000000u /* TH_C */, 0u, 0u, 0x800u, 40u});
            p.wordKinds[p.helperParam + 8u] = 2; // the entry, stored at run time: an address in the probe
            p.steps.push_back({"thbase", 20, {}, true, "GetThreadId()"});
            const uint32_t me = 0;
            const uint32_t go = sema(0, 100, "go");
            const uint32_t done = sema(0, 1, "done");
            const uint32_t wait = sema(0, 1, "wait");
            const uint32_t efParam = garbageBlock(r, p.data, 3);
            const uint32_t efAttr = r.chance(50) ? 2u : 0u; // EA_MULTI or EA_SINGLE
            putWords(p.data, efParam, {efAttr, r.u32(), 0u});
            p.steps.push_back({"thevent", 4, {D(efParam)}, true, "CreateEventFlag(attr " + hex(efAttr) + ")"});
            const uint32_t ef = static_cast<uint32_t>(p.steps.size() - 1);
            const uint32_t handler = garbageBlock(r, p.data, 2);
            putWords(p.data, handler, {0x03E00008u /* jr ra */, 0x00001021u /* move v0, zero */});
            // ++*arg; re-arm with 16 cycles (raised to the 100 us floor) until the third call
            const uint32_t repeat = garbageBlock(r, p.data, 7);
            putWords(p.data, repeat, {0x8C880000u /* lw t0, 0(a0) */, 0u, 0x25080001u /* addiu t0, t0, 1 */, 0xAC880000u /* sw t0, 0(a0) */,
                                      0x2D090003u /* sltiu t1, t0, 3 */, 0x03E00008u /* jr ra */, 0x00091100u /* sll v0, t1, 4 */});
            auto turn = [&]()
            {
                p.steps.push_back({"thsemap", 6, {R(go)}, false, "SignalSema(go)"});
                helper({"thsemap", 8, {R(go)}, false, "WaitSema(go)"});
            };

            const std::vector<uint32_t> kinds = {0, 1, 2, 3, 4, 5, 6, 7, 8};
            const uint32_t rounds = r.range(1, 6);
            for (uint32_t i = 0; i < rounds; ++i)
            {
                uint32_t kind = r.pick(kinds);
                if (r.chance(50))
                    kind = focus == 24 ? r.pick(std::vector<uint32_t>{2, 3}) : focus == 33 ? 4u : (focus == 35 || focus == 37) ? r.pick(std::vector<uint32_t>{5, 6, 8}) : kind;
                switch (kind)
                {
                case 0: // WaitSema released by SignalSema
                    turn();
                    p.steps.push_back({"thsemap", 8, {R(wait)}, false, "WaitSema(wait) blocks"});
                    helper({"thsemap", 6, {R(wait)}, false, "SignalSema(wait)"});
                    break;
                case 1: // WaitSema released by DeleteSema, or by ReleaseWaitThread
                {
                    turn();
                    if (r.chance(50))
                    {
                        const uint32_t gone = sema(0, 1, "gone");
                        p.steps.push_back({"thsemap", 8, {R(gone)}, false, "WaitSema(gone) blocks"});
                        helper({"thsemap", 5, {R(gone)}, false, "DeleteSema(gone)"});
                    }
                    else
                    {
                        p.steps.push_back({"thsemap", 8, {R(wait)}, false, "WaitSema(wait) blocks"});
                        helper({"thbase", 18, {R(me)}, false, "ReleaseWaitThread(main)"});
                    }
                    break;
                }
                case 2: // SleepThread released by WakeupThread
                    turn();
                    p.steps.push_back({"thbase", 24, {}, false, "SleepThread() blocks"});
                    helper({"thbase", 25, {R(me)}, false, "WakeupThread(main)"});
                    break;
                case 3: // a wakeup while waiting on something else is counted; CancelWakeupThread(0) returns it
                    turn();
                    p.steps.push_back({"thsemap", 8, {R(wait)}, false, "WaitSema(wait) blocks"});
                    helper({"thbase", 25, {R(me)}, false, "WakeupThread(main) while it waits on a sema"});
                    if (r.chance(50))
                        helper({"thbase", 25, {R(me)}, false, "WakeupThread(main) again"});
                    helper({"thsemap", 6, {R(wait)}, false, "SignalSema(wait)"});
                    if (r.chance(50))
                        p.steps.push_back({"thbase", 24, {}, false, "SleepThread() with a wakeup pending"});
                    p.steps.push_back({"thbase", 27, {C(0)}, false, "CancelWakeupThread(self)"});
                    break;
                case 4: // DelayThread: the helper runs meanwhile
                {
                    turn();
                    const uint32_t usec = r.pick(std::vector<uint32_t>{0u, 1u, 100u, r.range(1, 3000)});
                    p.steps.push_back({"thbase", 33, {C(usec)}, false, "DelayThread(" + std::to_string(usec) + ")"});
                    if (r.chance(50))
                        helper({"thsemap", 9, {R(wait)}, false, "PollSema(wait) while main is delayed"});
                    break;
                }
                case 5: // an alarm that fires during a delay; cancelling it afterwards finds nothing
                {
                    const uint32_t clock = garbageBlock(r, p.data, 2);
                    putWords(p.data, clock, {r.range(1, 0x4000), 0u});
                    const uint32_t arg = r.u32();
                    p.steps.push_back({"thbase", 35, {D(clock), D(handler), C(arg)}, false, "SetAlarm(short)"});
                    if (r.chance(40))
                        p.steps.push_back({"thbase", 35, {D(clock), D(handler), C(arg)}, false, "SetAlarm(same handler, arg)"});
                    turn();
                    p.steps.push_back({"thbase", 33, {C(2000)}, false, "DelayThread(2000) lets it fire"});
                    p.steps.push_back({"thbase", 37, {D(handler), C(arg)}, false, "CancelAlarm(fired)"});
                    break;
                }
                case 8: // an alarm whose handler re-arms it twice, then lets it end
                {
                    const uint32_t clock = garbageBlock(r, p.data, 2), counter = garbageBlock(r, p.data, 1);
                    putWords(p.data, clock, {r.range(1, 0x2000), 0u});
                    putWords(p.data, counter, {0u});
                    p.steps.push_back({"thbase", 35, {D(clock), D(repeat), D(counter)}, false, "SetAlarm(repeating)"});
                    turn();
                    p.steps.push_back({"thbase", 33, {C(3000)}, false, "DelayThread(3000): three calls"});
                    p.steps.push_back({"thbase", 37, {D(repeat), D(counter)}, false, "CancelAlarm(ended)"});
                    break;
                }
                case 6: // an alarm cancelled before it fires
                {
                    const uint32_t clock = garbageBlock(r, p.data, 2);
                    putWords(p.data, clock, {0x10000000u, 0u});
                    const uint32_t arg = r.u32();
                    p.steps.push_back({"thbase", 35, {D(clock), D(handler), C(arg)}, false, "SetAlarm(long)"});
                    p.steps.push_back({"thbase", 37, {D(handler), C(arg ^ 1u)}, false, "CancelAlarm(other arg)"});
                    p.steps.push_back({"thbase", 37, {D(handler), C(arg)}, false, "CancelAlarm(pending)"});
                    break;
                }
                default: // WaitEventFlag released by SetEventFlag (maybe after a set that does not satisfy it)
                {
                    const uint32_t bits = r.chance(50) ? (1u << r.range(0, 31)) : (r.u32() | 1u);
                    const uint32_t mode = r.pick(std::vector<uint32_t>{0u, 1u, 0x10u, 0x11u});
                    const uint32_t res = garbageBlock(r, p.data, 1);
                    turn();
                    p.steps.push_back({"thevent", 10, {R(ef), C(bits), C(mode), D(res)}, false,
                                       "WaitEventFlag(ef, " + hex(bits) + ", mode " + hex(mode) + ") blocks"});
                    const uint32_t low = bits & (~bits + 1u);
                    if ((mode & 1u) == 0u && bits != low && r.chance(60))
                        helper({"thevent", 6, {R(ef), C(low)}, false, "SetEventFlag(ef, " + hex(low) + ") not enough"});
                    helper({"thevent", 6, {R(ef), C(bits | (r.chance(30) ? 0x80000000u : 0u))}, false, "SetEventFlag(ef, bits)"});
                    p.steps.push_back({"thevent", 8, {R(ef), C(0)}, false, "ClearEventFlag(ef, 0)"});
                    break;
                }
                }
            }
            // End: the helper reports its ID and signals done; once it has returned it is deleted.
            turn();
            helper({"thbase", 20, {}, true, "GetThreadId()"});
            const uint32_t helperId = static_cast<uint32_t>(p.steps.size() - 1);
            helper({"thsemap", 6, {R(done)}, false, "SignalSema(done)"});
            p.steps.push_back({"thsemap", 8, {R(done)}, false, "WaitSema(done) blocks"});
            p.steps.push_back({"thbase", 33, {C(100)}, false, "DelayThread(100): the helper returns"});
            p.steps.push_back({"thbase", 5, {R(helperId)}, false, "DeleteThread(helper)"});
            for (uint32_t id : {go, done, wait})
                p.steps.push_back({"thsemap", 5, {R(id)}, false, "DeleteSema"});
            p.steps.push_back({"thevent", 5, {R(ef)}, false, "DeleteEventFlag(ef)"});
            p.note = std::to_string(p.steps.size()) + " calls";
            return p;
        }

        struct KernelTest
        {
            const char *lib;
            uint16_t ord;
            const char *name;
            Program (*gen)(Rng &, uint32_t);
            bool devices = false; // boot the Sony side with timers and interrupts
        };
        const std::vector<KernelTest> &kernelTests()
        {
            static const std::vector<KernelTest> tests = {
                {"thsemap", 4, "CreateSema", semaProgram},
                {"thsemap", 5, "DeleteSema", semaProgram},
                {"thsemap", 6, "SignalSema", semaProgram},
                {"thsemap", 7, "iSignalSema", semaProgram},
                {"thsemap", 8, "WaitSema", semaProgram},
                {"thsemap", 9, "PollSema", semaProgram},
                {"thsemap", 11, "ReferSemaStatus", semaProgram},
                {"thsemap", 12, "iReferSemaStatus", semaProgram},
                {"thevent", 4, "CreateEventFlag", eventProgram},
                {"thevent", 5, "DeleteEventFlag", eventProgram},
                {"thevent", 6, "SetEventFlag", eventProgram},
                {"thevent", 7, "iSetEventFlag", eventProgram},
                {"thevent", 8, "ClearEventFlag", eventProgram},
                {"thevent", 9, "iClearEventFlag", eventProgram},
                {"thevent", 10, "WaitEventFlag", eventProgram},
                {"thevent", 11, "PollEventFlag", eventProgram},
                {"thevent", 13, "ReferEventFlagStatus", eventProgram},
                {"thevent", 14, "iReferEventFlagStatus", eventProgram},
                {"thbase", 4, "CreateThread", threadProgram},
                {"thbase", 5, "DeleteThread", threadProgram},
                {"thbase", 6, "StartThread", threadProgram},
                {"thbase", 10, "TerminateThread", threadProgram},
                {"thbase", 14, "ChangeThreadPriority", threadProgram},
                {"thbase", 18, "ReleaseWaitThread", threadProgram},
                {"thbase", 20, "GetThreadId", threadProgram},
                {"thbase", 22, "ReferThreadStatus", threadProgram},
                {"thbase", 25, "WakeupThread", threadProgram},
                {"thbase", 27, "CancelWakeupThread", threadProgram},
                {"thbase", 29, "SuspendThread", threadProgram},
                {"thbase", 31, "ResumeThread", threadProgram},
                {"thbase", 39, "USec2SysClock", threadProgram},
                {"thbase", 40, "SysClock2USec", threadProgram},
                {"thbase", 42, "GetThreadCurrentPriority", threadProgram},
                {"thbase", 24, "SleepThread", waitProgram, true},
                {"thbase", 33, "DelayThread", waitProgram, true},
                {"thbase", 35, "SetAlarm", waitProgram, true},
                {"thbase", 37, "CancelAlarm", waitProgram, true},
                {"thsemap", 8, "WaitSema (blocking)", waitProgram, true},
                {"thevent", 10, "WaitEventFlag (blocking)", waitProgram, true},
                {"intrman", 4, "RegisterIntrHandler", intrProgram},
                {"intrman", 5, "ReleaseIntrHandler", intrProgram},
                {"intrman", 6, "EnableIntr", intrProgram},
                {"intrman", 7, "DisableIntr", intrProgram},
                {"intrman", 8, "CpuDisableIntr", intrProgram},
                {"intrman", 17, "CpuSuspendIntr", intrProgram},
                {"intrman", 23, "QueryIntrContext", intrProgram},
            };
            return tests;
        }

        // Compares one program's observations. Object IDs are compared through a bijection
        // (Sony's are pointer-derived, ours are small integers); everything else bit-exact.
        std::string compareProgram(const Program &p, const Observed &lle, const Observed &hle, uint32_t base, size_t pi, bool verbose)
        {
            // One bijection per object kind (the HLE numbers threads, semaphores and event flags
            // separately; Sony's IDs are unique across kinds).
            std::map<std::string, std::map<uint32_t, uint32_t>> fwdByKind, backByKind;
            std::string diff;
            for (size_t k = 0; k < p.steps.size(); ++k)
            {
                const Step &st = p.steps[k];
                const uint32_t a = lle.v0[base + k], b = hle.v0[base + k];
                std::string bad;
                if (st.ignoreV0)
                {
                }
                else if (st.returnsId && static_cast<int32_t>(a) > 0 && static_cast<int32_t>(b) > 0)
                {
                    auto &fwd = fwdByKind[st.lib];
                    auto &back = backByKind[st.lib];
                    auto f = fwd.find(a);
                    auto g = back.find(b);
                    if ((f != fwd.end() && f->second != b) || (g != back.end() && g->second != a))
                        bad = "ID mapping broken";
                    fwd[a] = b;
                    back[b] = a;
                }
                else if (a != b)
                    bad = "v0 Sony=" + hex(a) + " (" + std::to_string(static_cast<int32_t>(a)) + ") HLE=" + hex(b) + " (" + std::to_string(static_cast<int32_t>(b)) + ")";
                if (verbose)
                    std::cout << "      " << k << ": " << st.text << " -> Sony " << static_cast<int32_t>(a) << ", HLE " << static_cast<int32_t>(b) << (bad.empty() ? "" : "  <-- " + bad) << "\n";
                if (!bad.empty() && diff.empty())
                    diff = "step " + std::to_string(k) + " " + st.text + ": " + bad;
            }
            const auto &da = lle.data[pi], &db = hle.data[pi];
            for (size_t i = 0; i < da.size() && diff.empty(); ++i)
            {
                if (const auto kind = p.wordKinds.find(static_cast<uint32_t>(i & ~size_t{3})); kind != p.wordKinds.end())
                {
                    if ((i & 3) != 0 || kind->second == 0)
                        continue;
                    uint32_t wa = 0, wb = 0;
                    std::memcpy(&wa, da.data() + i, 4);
                    std::memcpy(&wb, db.data() + i, 4);
                    bool same = wa == wb; // untouched (garbage) words are equal on both sides
                    if (same)
                    {
                    }
                    else if (kind->second == 1)
                        same = (wa == 0 && wb == 0) || std::any_of(fwdByKind.begin(), fwdByKind.end(), [&](const auto &kind)
                                                                               { const auto it = kind.second.find(wa); return it != kind.second.end() && it->second == wb; });
                    else
                        same = wa - lle.segment == wb - hle.segment;
                    if (!same)
                    {
                        char buf[128];
                        std::snprintf(buf, sizeof buf, "data[%zu] (%s) Sony=%08x HLE=%08x", i, kind->second == 1 ? "id" : "module address", wa, wb);
                        diff = buf;
                    }
                    continue;
                }
                if (da[i] != db[i])
                {
                    char buf[96];
                    std::snprintf(buf, sizeof buf, "data[%zu] Sony=%02x HLE=%02x", i, da[i], db[i]);
                    diff = buf;
                    // name the step whose buffer argument starts closest below this byte
                    size_t best = SIZE_MAX;
                    uint32_t bestStart = 0;
                    for (size_t k = 0; k < p.steps.size(); ++k)
                        for (const Arg &arg : p.steps[k].args)
                            if (arg.kind == Arg::Data && arg.value <= i && (best == SIZE_MAX || arg.value >= bestStart))
                            {
                                best = k;
                                bestStart = arg.value;
                            }
                    if (best != SIZE_MAX)
                        diff += " (" + p.steps[best].text + " buffer +" + std::to_string(i - bestStart) + ")";
                }
            }
            return diff;
        }
    }

    int runStage2(const Options &opt, Summary &summary)
    {
        if (opt.bios.empty())
        {
            std::cout << "stage2: SKIP no --bios image\n";
            return 2;
        }
        const auto bios = parseRomdir(readFile(opt.bios));
        const auto ioprp = parseRomdir(readFile(opt.ioprp));
        if (bios.empty() || ioprp.empty())
        {
            std::cout << "stage2: SKIP cannot read the BIOS / IOPRP images\n";
            return 2;
        }
        // The boot list of the BIOS (IOPBTCONF) cut down to THREADMAN and what it links against;
        // IOPRP300 versions replace the BIOS ones.
        const std::vector<std::string> order = {"SYSMEM", "LOADCORE", "EXCEPMAN", "INTRMANI", "TIMEMANI", "SYSCLIB", "HEAPLIB", "THREADMAN"};
        if (opt.verbose && opt.filter == "boot")
        {
            auto kb = std::make_unique<KernelBoot>();
            const bool ok = kb->boot(bios, ioprp, order, nullptr, 20'000'000);
            std::cout << kb->log << kb->modules();
            std::cout << "  run: " << exitName(kb->lastRun.exit) << " after " << kb->lastRun.instructions << " instructions, pc=" << hex(kb->finalCpu.pc) << "\n";
            return ok ? 0 : 1;
        }
        constexpr uint32_t kPerBoot = 25; // programs per probe IRX (one kernel boot each)
        for (const KernelTest &t : kernelTests())
        {
            const std::string id = std::string(t.lib) + ":" + std::to_string(t.ord);
            const std::string full = id + " " + t.name;
            if (!opt.filter.empty() && full.find(opt.filter) == std::string::npos && id != opt.filter)
                continue;
            ExportResult res;
            res.id = id;
            res.name = t.name;
            const uint32_t cases = opt.onlyCase >= 0 ? 1u : std::max<uint32_t>(1u, opt.cases / 10u);
            for (uint32_t first = 0; first < cases; first += kPerBoot)
            {
                std::vector<Program> programs;
                std::vector<uint32_t> caseIds;
                for (uint32_t k = first; k < std::min(cases, first + kPerBoot); ++k)
                {
                    const uint32_t caseId = opt.onlyCase >= 0 ? static_cast<uint32_t>(opt.onlyCase) : k;
                    Rng rng(opt.seed ^ hashName(std::string("k2:") + t.name) ^ (uint64_t(caseId) * 0x9E3779B97F4A7C15ull));
                    programs.push_back(t.gen(rng, t.ord));
                    caseIds.push_back(caseId);
                }
                // One kernel boot per batch; when a program derails its batch (a hang, a kernel
                // panic), every program of that batch is rerun on its own boot.
                auto runBatch = [&](const std::vector<Program> &batch, Observed &lleOut, Observed &hleOut, Irx &irxOut)
                {
                    irxOut = buildIrx(batch, KernelBoot::kMailbox);
                    auto kb = std::make_unique<KernelBoot>();
                    const bool booted = kb->boot(bios, ioprp, order, &irxOut.file, 50'000'000, t.devices);
                    lleOut = Observed{};
                    if (!booted)
                        lleOut.error = "kernel boot failed: " + kb->log;
                    else
                        lleOut = collect(irxOut, batch, [&](uint32_t a) { return kb->m.memory.read32(a); },
                                         [&](uint32_t a) { return kb->m.memory.read8(a); }, KernelBoot::kMailbox);
                    if (!lleOut.ok)
                        lleOut.error += std::string(" (Sony run: ") + exitName(kb->lastRun.exit) + " pc=" + hex(kb->finalCpu.pc) +
                                        (kb->lastRun.exit == LleMachine::Exit::Exception ? " cause=" + std::to_string(kb->lastRun.cause) + " epc=" + hex(kb->lastRun.epc) : "") + ")";
                    hleOut = runHle(irxOut, batch, KernelBoot::kMailbox);
                };
                std::vector<Observed> lleEach(programs.size()), hleEach(programs.size());
                std::vector<Irx> irxEach(programs.size());
                std::vector<size_t> indexEach(programs.size(), 0); // program index inside its IRX
                {
                    Observed l, h;
                    Irx irx;
                    runBatch(programs, l, h, irx);
                    if (l.ok && h.ok)
                        for (size_t pi = 0; pi < programs.size(); ++pi)
                        {
                            lleEach[pi] = l;
                            hleEach[pi] = h;
                            irxEach[pi] = irx;
                            indexEach[pi] = pi;
                        }
                    else
                        for (size_t pi = 0; pi < programs.size(); ++pi)
                            runBatch({programs[pi]}, lleEach[pi], hleEach[pi], irxEach[pi]);
                }
                for (size_t pi = 0; pi < programs.size(); ++pi)
                {
                    ++res.cases;
                    std::string why;
                    std::string kind;
                    const Observed &lle = lleEach[pi], &hle = hleEach[pi];
                    const size_t local = indexEach[pi];
                    if (!lle.ok)
                    {
                        kind = "lle-fault";
                        why = lle.error;
                    }
                    else if (!hle.ok)
                    {
                        kind = "hle-fault";
                        why = hle.error;
                    }
                    else
                    {
                        if (opt.verbose)
                            std::cout << "    case " << caseIds[pi] << ":\n";
                        why = compareProgram(programs[pi], lle, hle, irxEach[pi].stepBase[local], local, opt.verbose);
                        kind = why.rfind("data", 0) == 0 ? "out-buffer" : "v0";
                    }
                    if (why.empty())
                        ++res.matches;
                    else
                    {
                        ++res.mismatches;
                        ++res.kinds[kind];
                        if (res.examples.size() < opt.examples)
                            res.examples.push_back("case " + std::to_string(caseIds[pi]) + ": " + programs[pi].note + "\n      " + why);
                    }
                }
            }
            summary.add(res);
        }
        return 0;
    }
}
