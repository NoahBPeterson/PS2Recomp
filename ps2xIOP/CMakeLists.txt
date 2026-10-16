cmake_minimum_required(VERSION 3.21)

project(ps2xIOP LANGUAGES CXX)

option(PS2X_IOP_BUILD_TESTS "Build ps2xIOP emulator smoke tests" OFF)

add_library(ps2_iop STATIC
    src/ps2_path.cpp
    src/iop_module_manager.cpp
    src/iop_subsystem.cpp
    src/emulator/iop_emulator.cpp
    src/emulator/core/iop_cpu.cpp
    src/emulator/core/iop_devices.cpp
    src/emulator/core/iop_kernel.cpp
    src/emulator/core/iop_memory.cpp
    src/emulator/services/iop_module_loader.cpp
    src/emulator/services/iop_rpc.cpp
    src/emulator/imports/iop_cdvd.cpp
    src/emulator/imports/iop_heaplib.cpp
    src/emulator/imports/iop_imports.cpp
    src/emulator/imports/iop_intrman.cpp
    src/emulator/imports/iop_ioman.cpp
    src/emulator/imports/iop_loadcore.cpp
    src/emulator/imports/iop_stdio.cpp
    src/emulator/imports/iop_sysclib.cpp
    src/emulator/imports/iop_sysmem.cpp
    src/emulator/imports/iop_timrman.cpp
    src/emulator/imports/iop_vblank.cpp
    src/modules/dbcman.cpp
    src/modules/libsd.cpp
    src/modules/mcserv.cpp
)

target_compile_features(ps2_iop PUBLIC cxx_std_20)

target_include_directories(ps2_iop
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(ps2x::iop ALIAS ps2_iop)

if(PS2X_IOP_BUILD_TESTS)
    enable_testing()
    add_executable(ps2_iop_emulator_tests tests/iop_emulator_tests.cpp)
    target_link_libraries(ps2_iop_emulator_tests PRIVATE ps2_iop)
    add_test(NAME ps2_iop_emulator_tests COMMAND ps2_iop_emulator_tests)

    add_executable(ps2_iop_import_tests tests/iop_import_tests.cpp)
    target_link_libraries(ps2_iop_import_tests PRIVATE ps2_iop)
    target_include_directories(ps2_iop_import_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ps2_iop_import_tests COMMAND ps2_iop_import_tests)

    add_executable(ps2_iop_compatibility_tests tests/iop_compatibility_tests.cpp)
    target_link_libraries(ps2_iop_compatibility_tests PRIVATE ps2_iop)
    add_test(NAME ps2_iop_compatibility_tests COMMAND ps2_iop_compatibility_tests)

    add_executable(ps2_iop_import_version_tests tests/iop_import_version_tests.cpp)
    target_link_libraries(ps2_iop_import_version_tests PRIVATE ps2_iop)
    target_include_directories(ps2_iop_import_version_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ps2_iop_import_version_tests COMMAND ps2_iop_import_version_tests)

    # Differential checks against the original Sony IOP modules (read at run time from the disc /
    # BIOS; see tools/conformance/iop.sh). Not an add_test: it needs the firmware images.
    add_executable(ps2_iop_ground_truth
        tests/ground_truth/gt_main.cpp
        tests/ground_truth/gt_stage1.cpp
        tests/ground_truth/gt_stage2.cpp)
    target_link_libraries(ps2_iop_ground_truth PRIVATE ps2_iop)
    target_include_directories(ps2_iop_ground_truth PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # Differential checks of the IOP device models against PCSX2's own implementation, compiled
    # from a PCSX2 source tree (GPL-3, read from outside the repo, never committed).
    set(PS2X_PCSX2_SRC "" CACHE PATH "PCSX2 source tree (tag v2.8.2) for the device-model differential tests")
    if(PS2X_PCSX2_SRC AND EXISTS "${PS2X_PCSX2_SRC}/pcsx2/IopCounters.cpp")
        # Copied so that its quoted includes find the stand-ins in tests/pcsx2_ref/shim, not PCSX2's headers.
        set(_ref ${CMAKE_CURRENT_BINARY_DIR}/pcsx2_ref)
        configure_file(${PS2X_PCSX2_SRC}/pcsx2/IopCounters.cpp ${_ref}/IopCounters.cpp COPYONLY)
        configure_file(${PS2X_PCSX2_SRC}/pcsx2/IopCounters.h ${_ref}/include/IopCounters.h COPYONLY)
        add_executable(ps2_iop_rcnt_diff tests/pcsx2_ref/rcnt_diff.cpp ${_ref}/IopCounters.cpp)
        target_link_libraries(ps2_iop_rcnt_diff PRIVATE ps2_iop)
        target_include_directories(ps2_iop_rcnt_diff PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/pcsx2_ref/shim ${_ref}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
    endif()
    if(PS2X_PCSX2_SRC AND EXISTS "${PS2X_PCSX2_SRC}/pcsx2/R3000AInterpreter.cpp")
        set(_cpu ${CMAKE_CURRENT_BINARY_DIR}/pcsx2_ref_cpu)
        foreach(f R3000AInterpreter.cpp R3000AOpcodeTables.cpp)
            configure_file(${PS2X_PCSX2_SRC}/pcsx2/${f} ${_cpu}/${f} COPYONLY)
        endforeach()
        configure_file(${PS2X_PCSX2_SRC}/pcsx2/R3000A.h ${_cpu}/include/R3000A.h COPYONLY)
        add_executable(ps2_iop_cpu_diff tests/pcsx2_ref/cpu_diff.cpp ${_cpu}/R3000AInterpreter.cpp ${_cpu}/R3000AOpcodeTables.cpp)
        target_link_libraries(ps2_iop_cpu_diff PRIVATE ps2_iop)
        target_include_directories(ps2_iop_cpu_diff PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/pcsx2_ref/shim_cpu ${_cpu}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_compile_options(ps2_iop_cpu_diff PRIVATE -w)
    endif()

endif()

install(TARGETS ps2_iop
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)
