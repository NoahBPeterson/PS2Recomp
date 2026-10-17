# ps2x_ee_kernel_replay: the runtime's EE syscalls against the PS2 kernel (PCSX2's BIOS), on probes made
# by tools/conformance/ee_kernel/eek.py (see the comment at the top of ee_kernel_replay.cpp).
add_executable(ps2x_ee_kernel_replay
    ee_kernel_replay.cpp
    $<TARGET_OBJECTS:ps2_test_function_table>
)
target_include_directories(ps2x_ee_kernel_replay PRIVATE
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)
target_link_libraries(ps2x_ee_kernel_replay PRIVATE ps2_runtime)
