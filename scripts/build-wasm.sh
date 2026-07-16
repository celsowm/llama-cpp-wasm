#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/wasm"
DIST_DIR="${ROOT_DIR}/dist/wasm"
DEMO_DIR="${ROOT_DIR}/demo/public/wasm"

command -v emcmake >/dev/null 2>&1 || {
  echo "emcmake was not found. Activate the Emscripten SDK environment first." >&2
  exit 1
}

command -v ninja >/dev/null 2>&1 || {
  echo "ninja was not found." >&2
  exit 1
}

if [[ ! -f "${ROOT_DIR}/vendor/llama.cpp/CMakeLists.txt" ]]; then
  echo "Missing vendor/llama.cpp. Run: npm run vendor:llama" >&2
  exit 1
fi

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${DIST_DIR}" "${DEMO_DIR}"

emcmake cmake \
  -S "${ROOT_DIR}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

# Build both targets. They share the llama.cpp pinned revision so the cost of
# building libllama once is amortised.
cmake --build "${BUILD_DIR}" --target llama_cpp_wasm_st llama_cpp_wasm_mt --parallel

copy_artifact() {
  local target_name="$1"
  local out_stem="$2"

  local js_file
  local wasm_file

  js_file="$(find "${BUILD_DIR}" -type f -name "${target_name}.js" -print -quit)"
  wasm_file="$(find "${BUILD_DIR}" -type f -name "${target_name}.wasm" -print -quit)"

  if [[ -z "${js_file}" || -z "${wasm_file}" ]]; then
    echo "Could not find output for ${target_name}." >&2
    exit 1
  fi

  cp "${js_file}"   "${DIST_DIR}/${out_stem}.js"
  cp "${wasm_file}" "${DIST_DIR}/${out_stem}.wasm"
  cp "${js_file}"   "${DEMO_DIR}/${out_stem}.js"
  cp "${wasm_file}" "${DEMO_DIR}/${out_stem}.wasm"
}

copy_artifact "llama-cpp-wasm-st" "llama-cpp-wasm-st"
copy_artifact "llama-cpp-wasm-mt" "llama-cpp-wasm-mt"

# Emscripten also emits a threaded worker (llama-cpp-wasm-mt.worker.js). Copy
# it next to the module so the runtime can spawn pthreads.
worker_file="$(find "${BUILD_DIR}" -type f -name 'llama-cpp-wasm-mt.worker.js' -print -quit)"
if [[ -n "${worker_file}" ]]; then
  cp "${worker_file}" "${DIST_DIR}/llama-cpp-wasm-mt.worker.js"
  cp "${worker_file}" "${DEMO_DIR}/llama-cpp-wasm-mt.worker.js"
fi

echo "WASM artifacts copied to:"
echo "  ${DIST_DIR}"
echo "  ${DEMO_DIR}"
