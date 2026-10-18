# Patches paraLLEl-GS's shader precompilation so the runtime can compile every pipeline the game
# needs at startup, behind a progress bar, instead of mid-game (src/lib/gs/pgs_precompile.h):
#   gs/gs_renderer.cpp, GSRenderer::kick_compilation_tasks:
#     - also queues the pipelines recorded as compiled mid-game in earlier runs
#       (ParallelGS::ps2x_recorded_variants);
#     - counts compiled pipelines (ParallelGS::ps2x_precompile_done / _total);
#     - tasks_per_thread rounds up: upstream rounded down and never compiled the last
#       num_tasks % target_threads variants.
#   Granite vulkan/command_buffer.cpp, CommandBuffer::build_compute_pipeline: reports each compute
#     pipeline compiled synchronously (a precompile miss) to Vulkan::ps2x_compute_compile_hook.
set(ps2x_pgs_precompile_header_dir "${CMAKE_CURRENT_LIST_DIR}/../src/lib/gs")

function(ps2x_patch_source var before after)
    string(FIND "${${var}}" "${before}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "parallel-gs precompile patch does not match the pinned source:\n${before}")
    endif()
    string(REPLACE "${before}" "${after}" patched "${${var}}")
    set(${var} "${patched}" PARENT_SCOPE)
endfunction()

# Replaces `relative` in `target`'s sources with the patched copy `out_name`.
function(ps2x_use_patched_source target source_dir relative out_name content)
    set(out "${parallel_gs_BINARY_DIR}/${out_name}")
    file(CONFIGURE OUTPUT "${out}" CONTENT "${content}" @ONLY)
    get_target_property(sources ${target} SOURCES)
    list(REMOVE_ITEM sources ${relative})
    set_property(TARGET ${target} PROPERTY SOURCES "${sources}")
    target_sources(${target} PRIVATE "${out}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_dir}/${relative}")
endfunction()

# --- gs/gs_renderer.cpp ---
file(READ "${parallel_gs_SOURCE_DIR}/gs/gs_renderer.cpp" gs_renderer_source)
ps2x_patch_source(gs_renderer_source "#include \"gs_renderer.hpp\"\n"
"#include \"gs_renderer.hpp\"\n#include \"pgs_precompile.h\"\n")
ps2x_patch_source(gs_renderer_source "void GSRenderer::kick_compilation_tasks()\n{"
"std::atomic<uint32_t> ps2x_precompile_done{0};
std::atomic<uint32_t> ps2x_precompile_total{0};
std::atomic<uint32_t> ps2x_precompile_recorded{0};
std::vector<Ps2xPipelineVariant> ps2x_recorded_variants;

void GSRenderer::kick_compilation_tasks()
{")
ps2x_patch_source(gs_renderer_source "\tsize_t num_tasks = tasks.size();\n"
"\t{
\t\t// ps2x: pipelines an earlier run compiled mid-game (pgs_precompile.h).
\t\tVulkan::Program *programs[] = {
\t\t\tshaders.binning, shaders.upload[0], shaders.upload[1], shaders.clut_write, shaders.vram_copy,
\t\t\tshaders.triangle_setup, shaders.single_sample_heuristic, shaders.ubershader[0][0],
\t\t\tshaders.ubershader[1][0], shaders.ubershader[0][1], shaders.ubershader[1][1], shaders.extwrite,
\t\t\tshaders.qword_clear, shaders.sampler_feedback,
\t\t};
\t\tuint32_t recorded = 0;
\t\tauto cmd = device->request_command_buffer();
\t\tfor (auto &variant : ps2x_recorded_variants)
\t\t{
\t\t\tfor (auto *prog : programs)
\t\t\t{
\t\t\t\tif (!prog || prog->get_hash() != variant.program)
\t\t\t\t\tcontinue;
\t\t\t\tcmd->set_program(prog);
\t\t\t\tcmd->set_specialization_constant_mask(variant.mask);
\t\t\t\tfor (unsigned i = 0; i < 8; i++)
\t\t\t\t\tcmd->set_specialization_constant(i, variant.constants[i]);
\t\t\t\tcmd->set_robustness(variant.robustness != 0);
\t\t\t\tcmd->enable_subgroup_size_control(variant.subgroupControl != 0);
\t\t\t\tcmd->set_subgroup_size_log2(variant.subgroupFull != 0, variant.subgroupMinLog2, variant.subgroupMaxLog2);
\t\t\t\tVulkan::DeferredPipelineCompile deferred = {};
\t\t\t\tcmd->extract_pipeline_state(deferred);
\t\t\t\tif (deferred.hash == variant.pipeline)
\t\t\t\t{
\t\t\t\t\ttasks.push_back(deferred);
\t\t\t\t\trecorded++;
\t\t\t\t}
\t\t\t\tbreak;
\t\t\t}
\t\t}
\t\tdevice->submit_discard(cmd);
\t\tps2x_precompile_recorded = recorded;
\t}

\tsize_t num_tasks = tasks.size();
")
ps2x_patch_source(gs_renderer_source "size_t tasks_per_thread = num_tasks / target_threads;"
"size_t tasks_per_thread = (num_tasks + target_threads - 1) / target_threads;\n\tps2x_precompile_done = 0;\n\tps2x_precompile_total = uint32_t(num_tasks);")
ps2x_patch_source(gs_renderer_source "tasks.data() + thread_index * tasks_per_thread,"
"tasks.data() + std::min<size_t>(thread_index * tasks_per_thread, tasks.size()),")
ps2x_patch_source(gs_renderer_source "device, task, Vulkan::CommandBuffer::CompileMode::AsyncThread);"
"device, task, Vulkan::CommandBuffer::CompileMode::AsyncThread);\n\t\t\t\tps2x_precompile_done.fetch_add(1, std::memory_order_relaxed);")
ps2x_use_patched_source(parallel-gs "${parallel_gs_SOURCE_DIR}/gs" gs_renderer.cpp gs_renderer_precompile.cpp "${gs_renderer_source}")
target_include_directories(parallel-gs PRIVATE "${ps2x_pgs_precompile_header_dir}")

# --- Granite vulkan/command_buffer.cpp ---
set(granite_vulkan_dir "${parallel_gs_SOURCE_DIR}/Granite/vulkan")
file(READ "${granite_vulkan_dir}/command_buffer.cpp" granite_command_buffer_source)
ps2x_patch_source(granite_command_buffer_source "Pipeline CommandBuffer::build_compute_pipeline(Device *device, const DeferredPipelineCompile &compile,"
"void (*ps2x_compute_compile_hook)(const DeferredPipelineCompile &compile) = nullptr;

Pipeline CommandBuffer::build_compute_pipeline(Device *device, const DeferredPipelineCompile &compile,")
ps2x_patch_source(granite_command_buffer_source "\tauto &shader = *compile.program->get_shader(ShaderStage::Compute);\n"
"\tif (mode == CompileMode::Sync && ps2x_compute_compile_hook)
\t\tps2x_compute_compile_hook(compile);

\tauto &shader = *compile.program->get_shader(ShaderStage::Compute);\n")
ps2x_use_patched_source(granite-vulkan "${granite_vulkan_dir}" command_buffer.cpp granite_command_buffer_precompile.cpp "${granite_command_buffer_source}")
