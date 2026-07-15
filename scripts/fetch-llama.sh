#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor/llama.cpp"

# Current llama.cpp revision used while preparing llama-cpp-wasm 0.0.1.
#
# COMPATIBILITY PIN: cpp/bridge.cpp calls llama_chat_apply_template() with the
# post-refactor signature that takes NO `model` argument. If you bump this
# commit, verify the llama_chat_apply_template ABI has not changed, otherwise
# the WASM build will fail (or silently misbehave). Update cpp/bridge.cpp and
# the lcw_format_chat cwrap signature in src/worker.ts together with any bump.
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-a5822222909b785f23ddc74ce3c8f85bd0e38562}"

mkdir -p "${ROOT_DIR}/vendor"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "${VENDOR_DIR}"
fi

git -C "${VENDOR_DIR}" fetch --depth 1 origin "${LLAMA_CPP_COMMIT}"
git -C "${VENDOR_DIR}" checkout --detach "${LLAMA_CPP_COMMIT}"

echo "llama.cpp pinned at ${LLAMA_CPP_COMMIT}"
