# ps2x_ee_oracle: replay EE oracle vectors (tools/ee_oracle) against recompiled functions.
# The function list is the project's manifest; each listed function's generated runner source is
# compiled into this executable on its own (no full runner build needed).
set(PS2X_EE_ORACLE_MANIFEST "${CMAKE_SOURCE_DIR}/../tools/ee_oracle/functions.txt"
    CACHE FILEPATH "EE oracle manifest: '<name> <addr> <spec> [kind]' per line")
set(PS2X_EE_ORACLE_RUNNER_DIR "${CMAKE_SOURCE_DIR}/ps2xRuntime/src/runner")

if(NOT EXISTS "${PS2X_EE_ORACLE_MANIFEST}")
    message(STATUS "ps2x_ee_oracle: no manifest at ${PS2X_EE_ORACLE_MANIFEST}, target skipped")
    return()
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PS2X_EE_ORACLE_MANIFEST}")

file(STRINGS "${PS2X_EE_ORACLE_MANIFEST}" _lines)
set(_sources "")
set(_decls "")
set(_table "")
foreach(_line IN LISTS _lines)
    string(REGEX REPLACE "#.*" "" _line "${_line}")
    string(STRIP "${_line}" _line)
    if(_line STREQUAL "")
        continue()
    endif()
    string(REGEX MATCH "^([A-Za-z0-9_]+)[ \t]+(0x[0-9A-Fa-f]+)" _m "${_line}")
    if(NOT _m)
        message(FATAL_ERROR "ps2x_ee_oracle: bad manifest line: ${_line}")
    endif()
    set(_name "${CMAKE_MATCH_1}")
    set(_addr "${CMAKE_MATCH_2}")
    set(_src "${PS2X_EE_ORACLE_RUNNER_DIR}/${_name}.cpp")
    if(NOT EXISTS "${_src}")
        message(WARNING "ps2x_ee_oracle: ${_src} does not exist (function renamed or dropped?); skipped")
        continue()
    endif()
    list(APPEND _sources "${_src}")
    string(APPEND _decls "void ${_name}(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);\n")
    string(APPEND _table "    {${_addr}u, \"${_name}\", &${_name}},\n")
endforeach()

set(_inc "${CMAKE_CURRENT_BINARY_DIR}/ee_oracle_functions.inc")
file(CONFIGURE OUTPUT "${_inc}" CONTENT
"// Generated from ${PS2X_EE_ORACLE_MANIFEST}. Do not edit.
@_decls@
struct EeOracleFunction { uint32_t addr; const char *name; RecompiledFn fn; };
static const EeOracleFunction kFunctions[] = {
@_table@};
" @ONLY)

add_executable(ps2x_ee_oracle
    ee_oracle_main.cpp
    ${_sources}
    $<TARGET_OBJECTS:ps2_test_function_table>
)
target_include_directories(ps2x_ee_oracle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)
target_link_libraries(ps2x_ee_oracle PRIVATE ps2_runtime)

# ps2x_ee_oracle_wide: the same replay linked against every generated function (the ps2EntryRunner
# objects, reused as built), dispatching through the runtime's function table. Any TC2 function whose
# call graph stays in recompiled code can be checked; no per-function compile.
if(TARGET ps2EntryRunner AND APPLE)
    add_executable(ps2x_ee_oracle_wide
        ee_oracle_main.cpp
        $<TARGET_OBJECTS:ps2EntryRunner>
    )
    target_compile_definitions(ps2x_ee_oracle_wide PRIVATE EE_ORACLE_WIDE=1)
    target_include_directories(ps2x_ee_oracle_wide PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
    )
    target_link_libraries(ps2x_ee_oracle_wide PRIVATE ps2_runtime)
    foreach(_lib rlImGui imgui)
        if(TARGET ${_lib})
            target_link_libraries(ps2x_ee_oracle_wide PRIVATE ${_lib})
        endif()
    endforeach()
    # the runner's objects bring the game's main(); enter through the replay instead
    target_link_options(ps2x_ee_oracle_wide PRIVATE "-Wl,-e,_ee_oracle_entry")
endif()
