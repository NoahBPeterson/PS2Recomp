// ps2_iop_ground_truth: differential checks of our IOP HLE against the original Sony IOP modules.
//
//   ps2_iop_ground_truth --ioprp disc/MODULES/IOPRP300.IMG [--bios BIOS.BIN] [--cases N] [--seed S]
//                        [--only "sysclib:19"|substring] [--case K] [-v] [--examples N] [--stage 1,2]
//
// One line per export:  GT <lib:ord> <name> cases=<n> match=<n> mismatch=<n> [kind=count ...]
// followed by up to --examples minimal repros. Reproduce a single case with --only <lib:ord> --case K -v.
// Exit status: 0 when the run completed (mismatch counts are judged by tools/conformance/iop.sh
// against tools/conformance/iop_known_failures.txt), 2 when a firmware image is missing.
#include "gt_stage1.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>

int main(int argc, char **argv)
{
    gt::Options opt;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
            {
                std::cerr << "missing value for " << a << "\n";
                std::exit(64);
            }
            return argv[++i];
        };
        if (a == "--ioprp")
            opt.ioprp = value();
        else if (a == "--bios")
            opt.bios = value();
        else if (a == "--cases")
            opt.cases = static_cast<uint32_t>(std::stoul(value()));
        else if (a == "--seed")
            opt.seed = std::stoull(value(), nullptr, 0);
        else if (a == "--only")
            opt.filter = value();
        else if (a == "--case")
            opt.onlyCase = std::stoi(value());
        else if (a == "--examples")
            opt.examples = std::stoul(value());
        else if (a == "--stage")
        {
            opt.stages.clear();
            std::stringstream ss(value());
            for (std::string s; std::getline(ss, s, ',');)
                opt.stages.push_back(s);
        }
        else if (a == "-v")
            opt.verbose = true;
        else
        {
            std::cerr << "unknown argument " << a << "\n";
            return 64;
        }
    }
    if (opt.ioprp.empty())
    {
        std::cerr << "--ioprp <IOPRP300.IMG> is required\n";
        return 64;
    }
    const auto t0 = std::chrono::steady_clock::now();
    gt::Summary summary;
    int rc = 0;
    for (const auto &stage : opt.stages)
    {
        const int r = stage == "1" ? gt::runStage1(opt, summary) : stage == "2" ? gt::runStage2(opt, summary) : 64;
        rc = std::max(rc, r);
    }
    uint32_t cases = 0, mismatches = 0, exports = 0, clean = 0;
    for (const auto &r : summary.results)
    {
        std::cout << "GT " << r.id << ' ' << r.name << " cases=" << r.cases << " match=" << r.matches
                  << " mismatch=" << r.mismatches;
        for (const auto &[k, n] : r.kinds)
            std::cout << ' ' << k << '=' << n;
        std::cout << '\n';
        for (const auto &e : r.examples)
            std::cout << "    " << e << '\n';
        cases += r.cases;
        mismatches += r.mismatches;
        ++exports;
        clean += r.mismatches == 0;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "SUMMARY exports=" << exports << " clean=" << clean << " cases=" << cases
              << " mismatches=" << mismatches << " seconds=" << secs << '\n';
    return rc;
}
