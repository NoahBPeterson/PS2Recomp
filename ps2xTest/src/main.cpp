#include "MiniTest.h"
#include <cstdlib>
#include <iostream>

void register_code_generator_tests();
void register_r5900_decoder_tests();
void register_r5900_general_instruction_tests();
void register_elf_analyzer_tests();
void register_pad_input_tests();
void register_ps2_runtime_io_tests();
void register_ps2_runtime_kernel_tests();
void register_ps2_runtime_interrupt_tests();
void register_ps2_memory_tests();
void register_ps2_vu1_tests();
void register_ps2_vu_tests();
void register_ps2_gs_tests();
void register_ps2_iop_tests();
void register_ps2_sif_rpc_tests();
void register_ps2_sif_dma_tests();
void register_ps2_recompiler_tests();
void register_ps2_runtime_expansion_tests();
void register_ps2_float_semantics_tests();
void register_ps2_mmi_permutation_tests();
void reset_ps2_test_function_table();

#ifdef PS2X_COVERAGE
extern "C" int __llvm_profile_write_file(void);
#endif

int main()
{
    // The GS tests read the VRAM allocation right after submitting packets, so they need the
    // synchronous GS backend (the runtime's default is asynchronous). Respect an explicit setting.
    if (!std::getenv("PS2X_GS_ASYNC"))
    {
#ifdef _WIN32
        _putenv_s("PS2X_GS_ASYNC", "0");
#else
        setenv("PS2X_GS_ASYNC", "0", 0);
#endif
    }
    MiniTest::BeforeEach(reset_ps2_test_function_table);

    register_code_generator_tests();
    register_r5900_decoder_tests();
    register_r5900_general_instruction_tests();
    register_elf_analyzer_tests();
    register_pad_input_tests();
    register_ps2_runtime_io_tests();
    register_ps2_runtime_kernel_tests();
    register_ps2_runtime_interrupt_tests();
    register_ps2_memory_tests();
    register_ps2_vu1_tests();
    register_ps2_vu_tests();
    register_ps2_gs_tests();
    register_ps2_iop_tests();
    register_ps2_sif_rpc_tests();
    register_ps2_sif_dma_tests();
    register_ps2_recompiler_tests();
    register_ps2_runtime_expansion_tests();
    register_ps2_float_semantics_tests();
    register_ps2_mmi_permutation_tests();
    int res = MiniTest::Run();
    std::cout.flush();
    std::cerr.flush();
#ifdef PS2X_COVERAGE
    // std::_Exit skips the atexit hook that writes the coverage profile (tools/conformance/coverage.sh
    // builds with -fprofile-instr-generate -DPS2X_COVERAGE): write it here.
    __llvm_profile_write_file();
#endif
    std::_Exit(res);
}
