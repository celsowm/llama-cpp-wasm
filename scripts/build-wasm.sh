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

cmake --build "${BUILD_DIR}" --target llama_cpp_wasm --parallel

JS_FILE="$(find "${BUILD_DIR}" -type f -name 'llama-cpp-wasm.js' -print -quit)"
WASM_FILE="$(find "${BUILD_DIR}" -type f -name 'llama-cpp-wasm.wasm' -print -quit)"

if [[ -z "${JS_FILE}" || -z "${WASM_FILE}" ]]; then
  echo "The expected Emscripten output was not found." >&2
  exit 1
fi

cp "${JS_FILE}" "${DIST_DIR}/llama-cpp-wasm.js"
cp "${WASM_FILE}" "${DIST_DIR}/llama-cpp-wasm.wasm"
cp "${JS_FILE}" "${DEMO_DIR}/llama-cpp-wasm.js"
cp "${WASM_FILE}" "${DEMO_DIR}/llama-cpp-wasm.wasm"

echo "WASM artifacts copied to:"
echo "  ${DIST_DIR}"
echo "  ${DEMO_DIR}"
