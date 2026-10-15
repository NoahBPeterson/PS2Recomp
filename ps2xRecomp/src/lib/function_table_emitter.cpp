#include "ps2recomp/Emitters/function_table_emitter.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ps2_recompiler.h"
#include "ps2recomp/recompiler_reporter.h"
#include "ps2recomp/types.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ps2recomp
{
    FunctionTableEmitter::FunctionTableEmitter(CodeGenerator &codeGenerator)
        : m_codeGenerator(codeGenerator)
    {
    }

    std::string FunctionTableEmitter::emit(const std::vector<Function> &functions, const std::map<uint32_t, std::string> &stubs)
    {
        (void)stubs;

        CodeGenerator &cg = m_codeGenerator;
        std::vector<std::pair<uint32_t, std::string>> entries;
        std::unordered_set<uint32_t> registeredAddresses;

        auto addEntry = [&](uint32_t address, const std::string &name)
        {
            if (name.empty())
            {
                return;
            }
            if ((address & 3u) != 0u)
            {
                std::ostringstream oss;
                oss << "Unaligned function table entry for " << name << " at 0x" << std::hex << address;

                if (cg.m_reporter)
                {
                    cg.m_reporter->errorAt("function-table", name, address, oss.str());
                }

                throw std::runtime_error(oss.str());
            }
            if (!registeredAddresses.insert(address).second)
            {
                return;
            }
            entries.emplace_back(address, name);
        };

        std::vector<std::pair<uint32_t, std::string>> normalFunctions;
        std::vector<std::pair<uint32_t, std::string>> stubFunctions;
        std::vector<std::pair<uint32_t, std::string>> systemCallFunctions;
        std::vector<std::pair<uint32_t, std::string>> libraryFunctions;

        for (const auto &function : functions)
        {
            if (!function.isRecompiled && !function.isStub && !function.isSkipped)
                continue;

            std::string generatedName = cg.getFunctionName(function.start);

            if (function.isSkipped)
            {
                libraryFunctions.emplace_back(function.start, generatedName);
            }
            else if (function.isStub)
            {
                const auto target = PS2Recompiler::resolveStubTarget(function.name);
                if (target == StubTarget::Syscall)
                {
                    systemCallFunctions.emplace_back(function.start, generatedName);
                }
                else
                {
                    stubFunctions.emplace_back(function.start, generatedName);
                }
            }
            else
            {
                normalFunctions.emplace_back(function.start, generatedName);
            }
        }

        if (cg.m_bootstrapInfo.valid)
        {
            std::string entryTarget = cg.m_bootstrapInfo.entryName;
            if (entryTarget.empty())
            {
                entryTarget = cg.getFunctionName(cg.m_bootstrapInfo.entry);
            }
            if (entryTarget.empty())
            {
                throw std::runtime_error("No entry function name available for registration.");
            }
            addEntry(cg.m_bootstrapInfo.entry, entryTarget);
        }

        for (const auto &[address, name] : normalFunctions)
        {
            addEntry(address, name);
        }

        for (const auto &[address, name] : stubFunctions)
        {
            addEntry(address, name);
        }
        for (const auto &[address, name] : systemCallFunctions)
        {
            addEntry(address, name);
        }
        for (const auto &[address, name] : libraryFunctions)
        {
            addEntry(address, name);
        }

        // Explicit function starts, including configured runtime handlers,
        // own their addresses even when another wrapper resumes there.
        std::unordered_set<uint32_t> entryWrapperStarts;
        for (const auto &function : functions)
        {
            if (function.name.rfind("entry_", 0) == 0)
            {
                entryWrapperStarts.insert(function.start);
            }
        }
        std::vector<uint32_t> resumeOwners;
        resumeOwners.reserve(cg.m_resumeEntryTargetsByOwner.size());
        for (const auto &[ownerStart, targets] : cg.m_resumeEntryTargetsByOwner)
        {
            (void)targets;
            resumeOwners.push_back(ownerStart);
        }
        std::sort(resumeOwners.begin(), resumeOwners.end(), [&](uint32_t left, uint32_t right)
                  {
                      const bool leftEntry = entryWrapperStarts.contains(left);
                      const bool rightEntry = entryWrapperStarts.contains(right);
                      if (leftEntry != rightEntry)
                      {
                          return !leftEntry;
                      }
                      // Prefer the most specific owner, as containing-owner
                      // discovery does. Synthetic entries only fill gaps left
                      // by normal owners, regardless of unordered-map layout.
                      return left > right;
                  });
        for (uint32_t ownerStart : resumeOwners)
        {
            const auto &targets = cg.m_resumeEntryTargetsByOwner.at(ownerStart);
            const std::string ownerName = cg.getFunctionName(ownerStart);
            if (ownerName.empty())
            {
                continue;
            }

            for (uint32_t target : targets)
            {
                addEntry(target, ownerName);
            }
        }

        std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        uint32_t tableBase = 0u;
        uint32_t tableEnd = 0u;
        uint32_t slotCount = 0u;
        if (!entries.empty())
        {
            tableBase = entries.front().first & ~3u;
            tableEnd = (entries.back().first + 4u + 3u) & ~3u;
            slotCount = (tableEnd - tableBase) >> 2;
        }

        std::stringstream ss;
        ss << "#include \"ps2_runtime.h\"\n";
        ss << "#include <ps2_recompiled_functions.h>\n";
        ss << "#include \"ps2_stubs.h\"\n";
        ss << "#include <ps2_recompiled_stubs.h>\n";
        ss << "#include \"ps2_syscalls.h\"\n\n";

        ss << "extern const uint32_t g_ps2RecompiledFunctionTableBase = 0x" << std::hex << tableBase << "u;\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0x" << std::hex << tableEnd << "u;\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = " << std::dec << slotCount << "u;\n";
        ss << "PS2Runtime::RecompiledFunction g_ps2RecompiledFunctionTable[" << std::dec << (slotCount == 0u ? 1u : slotCount) << "u] = {};\n\n";

        // The entries are constant data and the initializer is one loop: a constructor with one
        // assignment per entry (~124k for TC2) took the optimizer ~9 minutes on a single core.
        ss << "namespace {\n";
        ss << "struct GeneratedFunctionTableEntry {\n";
        ss << "    uint32_t slot;\n";
        ss << "    PS2Runtime::RecompiledFunction function;\n";
        ss << "};\n";
        ss << "const GeneratedFunctionTableEntry kGeneratedFunctionTableEntries[] = {\n";
        for (const auto &[address, name] : entries)
        {
            const uint32_t slot = (address - tableBase) >> 2;
            ss << "    {" << std::dec << slot << "u, " << name << "}, // 0x" << std::hex << address << std::dec << "\n";
        }
        if (entries.empty())
            ss << "    {0u, nullptr},\n";
        ss << "};\n";
        ss << "struct GeneratedFunctionTableInitializer {\n";
        ss << "    GeneratedFunctionTableInitializer() {\n";
        ss << "        for (const GeneratedFunctionTableEntry &entry : kGeneratedFunctionTableEntries)\n";
        ss << "            if (entry.function)\n";
        ss << "                g_ps2RecompiledFunctionTable[entry.slot] = entry.function;\n";
        ss << "    }\n";
        ss << "};\n";
        ss << "static const GeneratedFunctionTableInitializer g_generatedFunctionTableInitializer;\n";
        ss << "}\n";

        return ss.str();
    }
}
