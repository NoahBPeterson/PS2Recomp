include(CheckIPOSupported)

# Whole-program LTO makes every link of a large recompiled game take tens of minutes.
option(PS2X_ENABLE_IPO "Enable IPO/LTO for release targets" ON)

check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

function(EnableFastReleaseMode TargetName)
    message("> Enabling optimization for: ${TargetName}")
    if(MSVC)
        target_compile_options(${TargetName} PRIVATE
            $<$<CONFIG:Release>:
                /O2 # speed
                /Ob2 # inline aggressively
                /Oi # intrinsics
                /GL # whole program opt
                /Gy # function-level linking
                /Gw # global data in COMDAT
                /GF # string pooling
                /Zc:inline # remove unreferenced inline
                /fp:fast # fast math (graphics friendly)
                /DNDEBUG
                /arch:AVX2 # Advanced Vector Extensions 2
                /GS- # Disable Buffer Security Check (faster)
                /Qspectre- # Disable Spectre mitigations (faster)
            >
        )

        if(TARGET ${TargetName})
            target_link_options(${TargetName} PRIVATE
                $<$<CONFIG:Release>:
                    /LTCG # link-time code generation
                    /OPT:REF # remove unreferenced
                    /OPT:ICF # fold identical COMDATs
                >
            )
        endif()
    endif()

    if(NOT PS2X_ENABLE_IPO)
        message("> IPO/LTO disabled for: ${TargetName}")
    elseif(IPO_SUPPORTED)
        set_property(TARGET ${TargetName} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        # ThinLTO cache: relinks only re-optimize modules that changed.
        if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_link_options(${TargetName} PRIVATE
                $<$<CONFIG:Release>:-Wl,-cache_path_lto,${CMAKE_BINARY_DIR}/thinlto-cache>)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
            target_link_options(${TargetName} PRIVATE
                $<$<CONFIG:Release>:-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/thinlto-cache>)
        endif()
    else()
        message(WARNING "Interprocedural optimization not supported: ${ipo_error}")
    endif()
endfunction()