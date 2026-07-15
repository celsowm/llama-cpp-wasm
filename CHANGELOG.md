# Changelog

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
