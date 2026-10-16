#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gt
{
    struct Options
    {
        std::string ioprp; // disc/MODULES/IOPRP300.IMG
        std::string bios;  // PS2 BIOS dump (optional)
        uint32_t cases = 300;
        uint64_t seed = 0x1F0B5EEDull;
        std::string filter; // substring of "lib:ord name", or exactly "lib:ord"
        int onlyCase = -1;
        bool verbose = false;
        size_t examples = 2;
        std::vector<std::string> stages = {"1"};
    };

    struct ExportResult
    {
        std::string id;   // lib:ord
        std::string name; // export name (with [variant])
        uint32_t cases = 0, matches = 0, mismatches = 0;
        std::map<std::string, uint32_t> kinds; // first failing aspect -> count
        std::vector<std::string> examples;
    };

    struct Summary
    {
        std::vector<ExportResult> results;
        void add(const ExportResult &r) { results.push_back(r); }
    };

    int runStage1(const Options &opt, Summary &summary);
    int runStage2(const Options &opt, Summary &summary);
}
