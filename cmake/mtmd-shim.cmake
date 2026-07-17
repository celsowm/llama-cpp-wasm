# Wrapper-side mtmd target for the WASM build.
#
# The vendored llama.cpp gates tools/mtmd behind an `if (EMSCRIPTEN) else()`
# block in tools/CMakeLists.txt, so under Emscripten the upstream build does
# not produce the mtmd library. Rather than edit the pinned vendor tree (which
# `npm run vendor:llama` would discard), we declare the mtmd target here with
# the same source files and include paths as the upstream CMakeLists.txt
# (vendor/llama.cpp/tools/mtmd/CMakeLists.txt). The sources point at the
# vendored tree so no source files are duplicated.
#
# Divergences from the upstream CMakeLists are intentional for the WASM build:
#   - The MACHO_CURRENT_VERSION property is dropped (not recognized by CMake
#     4.2 on non-Apple platforms and breaks set_target_properties arg count).
#   - The install() rule is dropped (the wrapper copies outputs itself).
#   - The CLI binaries block is dropped (LLAMA_BUILD_TOOLS=OFF; image-only).
#   - MTMD_VIDEO is forced OFF by the wrapper CMakeLists.

find_package(Threads REQUIRED)

set(_lcw_mtmd_dir "${CMAKE_CURRENT_LIST_DIR}/../vendor/llama.cpp/tools/mtmd")
cmake_path(ABSOLUTE_PATH _lcw_mtmd_dir NORMALIZE OUTPUT_VARIABLE _lcw_mtmd_src_dir)

add_library(mtmd
            ${_lcw_mtmd_src_dir}/mtmd.cpp
            ${_lcw_mtmd_src_dir}/mtmd-audio.cpp
            ${_lcw_mtmd_src_dir}/mtmd-image.cpp
            ${_lcw_mtmd_src_dir}/mtmd.cpp
            ${_lcw_mtmd_src_dir}/mtmd-helper.cpp
            ${_lcw_mtmd_src_dir}/clip.cpp
            ${_lcw_mtmd_src_dir}/models/cogvlm.cpp
            ${_lcw_mtmd_src_dir}/models/conformer.cpp
            ${_lcw_mtmd_src_dir}/models/dotsocr.cpp
            ${_lcw_mtmd_src_dir}/models/exaone4_5.cpp
            ${_lcw_mtmd_src_dir}/models/gemma4a.cpp
            ${_lcw_mtmd_src_dir}/models/gemma4v.cpp
            ${_lcw_mtmd_src_dir}/models/gemma4ua.cpp
            ${_lcw_mtmd_src_dir}/models/gemma4uv.cpp
            ${_lcw_mtmd_src_dir}/models/glm4v.cpp
            ${_lcw_mtmd_src_dir}/models/granite-speech.cpp
            ${_lcw_mtmd_src_dir}/models/granite4-vision.cpp
            ${_lcw_mtmd_src_dir}/models/hunyuanvl.cpp
            ${_lcw_mtmd_src_dir}/models/internvl.cpp
            ${_lcw_mtmd_src_dir}/models/kimivl.cpp
            ${_lcw_mtmd_src_dir}/models/kimik25.cpp
            ${_lcw_mtmd_src_dir}/models/nemotron-v2-vl.cpp
            ${_lcw_mtmd_src_dir}/models/llama4.cpp
            ${_lcw_mtmd_src_dir}/models/llava.cpp
            ${_lcw_mtmd_src_dir}/models/minicpmv.cpp
            ${_lcw_mtmd_src_dir}/models/paddleocr.cpp
            ${_lcw_mtmd_src_dir}/models/pixtral.cpp
            ${_lcw_mtmd_src_dir}/models/qwen2vl.cpp
            ${_lcw_mtmd_src_dir}/models/qwen3vl.cpp
            ${_lcw_mtmd_src_dir}/models/mimovl.cpp
            ${_lcw_mtmd_src_dir}/models/qwen3a.cpp
            ${_lcw_mtmd_src_dir}/models/step3vl.cpp
            ${_lcw_mtmd_src_dir}/models/siglip.cpp
            ${_lcw_mtmd_src_dir}/models/whisper-enc.cpp
            ${_lcw_mtmd_src_dir}/models/deepseekocr.cpp
            ${_lcw_mtmd_src_dir}/models/deepseekocr2.cpp
            ${_lcw_mtmd_src_dir}/models/mobilenetv5.cpp
            ${_lcw_mtmd_src_dir}/models/youtuvl.cpp
            ${_lcw_mtmd_src_dir}/models/yasa2.cpp
            )

# The upstream CMakeLists sets VERSION/SOVERSION from LLAMA_INSTALL_VERSION;
# both are shared-library ABI props that have no effect under our static-lib
# build (BUILD_SHARED_LIBS=OFF). LLAMA_INSTALL_VERSION can also be empty when
# the upstream build-info.cmake did not populate BUILD_NUMBER, which the
# `set_target_properties ... VERSION <empty>` form rejects outright. We do
# not need either prop, so omit them.
# set_target_properties(mtmd PROPERTIES VERSION ${LLAMA_INSTALL_VERSION} SOVERSION 0)

target_link_libraries     (mtmd PUBLIC ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  ${_lcw_mtmd_src_dir})
target_include_directories(mtmd PRIVATE ${_lcw_mtmd_src_dir}/../..)
target_include_directories(mtmd PRIVATE ${_lcw_mtmd_src_dir}/../../vendor)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (MTMD_VIDEO)
    target_compile_definitions(mtmd PRIVATE MTMD_VIDEO)
endif()

# stb_image.h and miniaudio.h cast away const in a few places; the upstream
# CMakeLists silences this with -Wno-cast-qual on non-MSVC toolchains. Apply
# the same flag under clang (Emscripten).
target_compile_options(mtmd PRIVATE -Wno-cast-qual)

unset(_lcw_mtmd_dir)
unset(_lcw_mtmd_src_dir)
