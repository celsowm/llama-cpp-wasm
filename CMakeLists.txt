cmake_minimum_required(VERSION 3.20)
project(llama_cpp_wasm VERSION 0.0.4 LANGUAGES C CXX)

if (NOT EMSCRIPTEN)
    message(FATAL_ERROR "Use emcmake cmake to configure this project.")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR
        "Missing vendor/llama.cpp. Run: npm run vendor:llama")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_WASM_MEM64 OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_HTML OFF CACHE BOOL "" FORCE)

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/llama.cpp)

# Common Emscripten link flags shared by both single-thread and pthread
# builds. The list is factored so the per-targets only need to add their
# threading flags.
set(LCW_LINK_OPTIONS_COMMON
    -O3
    -msimd128
    "SHELL:-sNO_DISABLE_EXCEPTION_CATCHING=1"
    "SHELL:-sMODULARIZE=1"
    "SHELL:-sEXPORT_ES6=1"
    "SHELL:-sEXPORT_NAME=createLlamaCppWasmModule"
    "SHELL:-sENVIRONMENT=worker"
    "SHELL:-sNO_EXIT_RUNTIME=1"
    "SHELL:-sALLOW_MEMORY_GROWTH=1"
    "SHELL:-sINITIAL_MEMORY=268435456"
    "SHELL:-sMAXIMUM_MEMORY=4294967296"
    "SHELL:-sSTACK_SIZE=5242880"
    "SHELL:-sFORCE_FILESYSTEM=1"
    "SHELL:-sEXPORTED_FUNCTIONS=['_malloc','_free',\
'_lcw_load_model',\
'_lcw_format_chat_multi',\
'_lcw_start',\
'_lcw_generate_chunk',\
'_lcw_finished',\
'_lcw_reset_kv',\
'_lcw_unload',\
'_lcw_last_error',\
'_lcw_bench_prompt_ms',\
'_lcw_bench_generate_ms',\
'_lcw_bench_prompt_tokens',\
'_lcw_bench_generate_tokens',\
'_lcw_bench_prompt_batch',\
'_lcw_bench_load_ms']"
    "SHELL:-sEXPORTED_RUNTIME_METHODS=['FS','cwrap','HEAPU8','wasmMemory']"
)

# ---------------------------------------------------------------------------
# single-threaded build: llama-cpp-wasm-st.js / .wasm
# ---------------------------------------------------------------------------
add_executable(llama_cpp_wasm_st cpp/bridge.cpp)
target_link_libraries(llama_cpp_wasm_st PRIVATE llama)
target_compile_options(llama_cpp_wasm_st PRIVATE -O3 -msimd128 -fexceptions)

set_target_properties(llama_cpp_wasm_st PROPERTIES
    OUTPUT_NAME "llama-cpp-wasm-st"
    SUFFIX ".js"
)

target_link_options(llama_cpp_wasm_st PRIVATE ${LCW_LINK_OPTIONS_COMMON})

# ---------------------------------------------------------------------------
# multithreaded pthread build: llama-cpp-wasm-mt.js / .wasm
# ---------------------------------------------------------------------------
# The MT build links with -pthread and tells Emscripten to spawn a worker pool
# of size 4. With ALLOW_MEMORY_GROWTH, each pthread worker also has its own
# stack; -sSTACK_SIZE above applies to both the main and worker stacks.
# When no growth is allowed, INITIAL_MEMORY must be larger to accommodate the
# pthread stack pool; here growth absorbs the difference at runtime.
add_executable(llama_cpp_wasm_mt cpp/bridge.cpp)
target_link_libraries(llama_cpp_wasm_mt PRIVATE llama)
target_compile_options(llama_cpp_wasm_mt PRIVATE
    -O3
    -msimd128
    -fexceptions
    -pthread
)

set_target_properties(llama_cpp_wasm_mt PROPERTIES
    OUTPUT_NAME "llama-cpp-wasm-mt"
    SUFFIX ".js"
)

target_link_options(llama_cpp_wasm_mt PRIVATE
    ${LCW_LINK_OPTIONS_COMMON}
    -pthread
    "SHELL:-sUSE_PTHREADS=1"
    "SHELL:-sPTHREAD_POOL_SIZE=4"
)
