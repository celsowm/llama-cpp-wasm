cmake_minimum_required(VERSION 3.20)
project(llama_cpp_wasm VERSION 0.0.3 LANGUAGES C CXX)

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

add_executable(llama_cpp_wasm cpp/bridge.cpp)
target_link_libraries(llama_cpp_wasm PRIVATE llama)
target_compile_options(llama_cpp_wasm PRIVATE -O3 -msimd128 -fexceptions)

set_target_properties(llama_cpp_wasm PROPERTIES
    OUTPUT_NAME "llama-cpp-wasm"
    SUFFIX ".js"
)

target_link_options(llama_cpp_wasm PRIVATE
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
    "SHELL:-sEXPORTED_FUNCTIONS=['_malloc','_free','_lcw_load_model','_lcw_format_chat','_lcw_start','_lcw_next','_lcw_unload','_lcw_last_error']"
    "SHELL:-sEXPORTED_RUNTIME_METHODS=['FS','cwrap','HEAPU8']"
)
