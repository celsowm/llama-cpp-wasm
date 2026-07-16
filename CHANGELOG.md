# Changelog

## 0.0.4

Performance release focused on the three highest-impact opportunities
(pthread support, KV-cache preservation between turns, and chunked native
generation), plus several secondary improvements.

### Added

- Separate single-threaded (`llama-cpp-wasm-st`) and pthread
  (`llama-cpp-wasm-mt`) CMake targets; the build emits both artifacts plus
  the pthread worker (`.worker.js`).
- `LlamaCppWasm.createThreaded(assets)` that picks the right binary at
  runtime based on `crossOriginIsolated` and a reachable
  `SharedArrayBuffer`. The default thread count is `min(4,
  navigator.hardwareConcurrency)` when the MT build is selected.
- `LlamaCppWasm.resetKV()` for explicit KV-cache resets when switching
  the sidebar conversation to an unrelated thread.
- `lcw_generate_chunk` replaces `lcw_next`; up to 16 tokens per native
  call land in JavaScript as a single UTF-8 chunk, eliminating one JS/WASM
  cross per token.
- `lcw_bench_*` accessors and a `bench` payload on the `done` worker
  message, surfacing prompt-ms, generate-ms, prompt-tokens,
  generate-tokens, prompt-batch-size, and load-ms back to the runtime.
  The playground status line now reports prompt/generated tokens/s and
  time-to-first-token.
- `LFM25_230M_Q4_0`, `LFM25_230M_Q4_K_S`, `LFM25_230M_Q5_0`,
  `LFM25_230M_Q8_0` presets plus
  `LFM25_230M_QUANTIZATION_PRESETS` for the quantization benchmark matrix.

### Changed

- The native `llama_context` is created once on `lcw_load_model` and reused
  across turns. `lcw_start` now compares the new prompt's tokens with the
  previously evaluated prefix, rewinds only the divergent tail of the KV
  cache via `llama_memory_seq_rm`, and evaluates just the newly appended
  tokens. Multi-turn chat therefore pays for the new user message only,
  not the entire history.
- `lcw_generate_chunk` uses a stack-allocated `std::array<char, 256>` for
  the per-token piece, avoiding a per-token `std::vector<char>` heap
  allocation in the hot path.
- The MEMFS model file is unlinked immediately after
  `llama_model_load_from_file` succeeds, releasing ~150 MiB (230M) or
  ~731 MiB (1.2B) of heap pressure before inference begins.
- For known-size model sources, the worker pre-grows the WASM heap once
  via `wasmMemory.grow()` instead of triggering several incremental
  growth operations while copying the GGUF.
- The playground batches `renderMarkdown` updates with
  `requestAnimationFrame`, so a long answer no longer triggers O(n^2)
  DOM reflow work. Final Markdown is rendered once when the stream ends.
- The playground calls `resetKV()` when starting a new chat or switching
  sidebar conversations, so unrelated prompts don't force `lcw_start` to
  truncate the stale cache.
- The runtime threads default is computed up-front from the chosen binary;
  callers passing `load({...})` without `threads` no longer fall back to
  the historic single-threaded default.

### Removed

- `lcw_next` is gone; the worker drives generation through
  `lcw_generate_chunk` and `lcw_finished`.

## 0.0.3

GitHub Pages deployment release.

### Added

- `.github/workflows/deploy-pages.yml`.
- Automatic playground deployment on every push to `main`.
- Manual `workflow_dispatch` deployment.
- GitHub Pages build/deploy jobs and deployment environment.
- Dynamic Pages base-path configuration.
- Vite-recognizable worker factory.
- Production verification for the HTML and WASM artifacts.
- `DEPLOYMENT.md`.
- `.nojekyll` in the published playground.

### Fixed

- WASM URLs no longer point to the domain root when hosted at
  `/llama-cpp-wasm/`.
- The playground worker is now emitted through Vite's module-worker
  pipeline rather than relying on a raw TypeScript asset URL.

## 0.0.2

Targeted compatibility release for
`LiquidAI/LFM2.5-230M-GGUF`.

### Added

- Native GGUF chat-template formatting through
  `llama_model_chat_template()` and `llama_chat_apply_template()`.
- `engine.chat()` for one optional system message and one user message.
- `LFM25_230M_Q4_K_M` preset with the Hugging Face model URL.
- One-click remote-model loading in the demo.
- Type-check test command.

### Tested target

- Repository: `LiquidAI/LFM2.5-230M-GGUF`
- File: `LFM2.5-230M-Q4_K_M.gguf`
- Runtime path: HTTP streaming → MEMFS → llama.cpp → WASM SIMD

### Current limitations

- CPU-only, single-threaded WASM.
- One model and one active generation.
- One-turn chat in `0.0.2`; assistant-history replay is intentionally rejected.
- No OPFS cache, WebGPU, embeddings, grammar sampling, or pthread build.

## 0.0.1

Initial proof of concept with local/HTTP GGUF loading and streamed raw
completion.
