# llama-cpp-wasm 0.0.3

A small proof-of-concept runtime that executes decoder-only GGUF language
models in a browser by compiling `llama.cpp` to WebAssembly.

Version 0.0.2 is tested around `LiquidAI/LFM2.5-230M-GGUF` and its embedded chat template. It establishes the complete path:

```text
GGUF file or URL
      ↓
Dedicated Web Worker
      ↓
Emscripten MEMFS
      ↓
llama.cpp / ggml CPU backend
      ↓
WASM SIMD
      ↓
streamed text
```

## Features

- Browser-local inference without an application server.
- GGUF loading from a local file or same-origin/CORS-enabled URL.
- WASM SIMD CPU execution.
- Streaming completion API.
- Temperature, top-k, top-p, maximum-token and seed controls.
- Cancellation between token evaluations.
- TypeScript API and a minimal Vite demo.

## Requirements

- Node.js 20 or newer.
- CMake and Ninja.
- Git.
- An activated Emscripten SDK environment with `emcmake` and `em++` in `PATH`.
- A modern browser with WebAssembly SIMD.
- A small quantized decoder-only GGUF model. Start below roughly 1 GB.

The model is copied into Emscripten's virtual filesystem and then loaded by
`llama.cpp`, so peak browser memory is higher than the GGUF file size.

## Build

```bash
npm install
npm run vendor:llama
npm run build
npm run dev
```

Open the displayed Vite URL, choose a GGUF file, load it, and enter a prompt.

The vendoring script pins `llama.cpp` to a known commit used by this release.
To test another revision, edit `LLAMA_CPP_COMMIT` in
`scripts/fetch-llama.sh`. The `llama.cpp` C API evolves, so unpinned upgrades
can require changes to `cpp/bridge.cpp`.

## TypeScript API

```ts
import { LlamaCppWasm } from "llama-cpp-wasm";

const engine = await LlamaCppWasm.create({
  moduleUrl: new URL(
    "llama-cpp-wasm/wasm/llama-cpp-wasm.js",
    import.meta.url
  ).href,
  wasmUrl: new URL(
    "llama-cpp-wasm/wasm/llama-cpp-wasm.wasm",
    import.meta.url
  ).href
});

await engine.load(
  { file: selectedFile },
  {
    contextSize: 2048,
    batchSize: 256,
    threads: 1
  },
  progress => {
    console.log(progress.loadedBytes, progress.totalBytes);
  }
);

for await (const text of engine.completion("Hello, my name is", {
  maxTokens: 64,
  temperature: 0.7,
  topK: 40,
  topP: 0.95
})) {
  output.textContent += text;
}

engine.terminate();
```

A package consumer may override `workerUrl` in `create()` when its bundler
does not preserve the default `./worker.js` URL.

## Public API

### `LlamaCppWasm.create(options)`

Creates the dedicated worker and loads the generated Emscripten module.

### `engine.load(source, options, onProgress?)`

Writes the GGUF into the worker's MEMFS and loads it with `llama.cpp`.

Sources:

```ts
{ file: File }
```

or:

```ts
{ url: string, headers?: Record<string, string> }
```

### `engine.completion(prompt, options?)`

Returns an async iterable of decoded text fragments.

### `engine.cancel()`

Requests cancellation of the active completion. Version `0.0.1` observes
cancellation between token evaluations, not during a single `llama_decode`.

### `engine.unload()`

Frees the context, sampler and model, then removes the model file from MEMFS.

### `engine.terminate()`

Terminates the worker immediately.

## Browser deployment

The `0.0.1` binary is single-threaded, so cross-origin isolation is not
strictly required. The demo already sends COOP/COEP headers to prepare for a
future pthread build.

Your model URL must be same-origin or return suitable CORS headers.

## Version boundaries

This release does **not** attempt to be a production replacement for native
`llama.cpp`. It is the smallest useful base for the next steps:

- `0.0.2`: chat templates, model metadata and better error reporting.
- `0.0.3`: OPFS model cache and URL resume support.
- `0.1.0`: separate single-thread and pthread binaries.
- later: WebGPU backend.

## License

The wrapper is MIT licensed. `llama.cpp` is fetched separately and retains
its own license and notices.


## LiquidAI LFM2.5 230M smoke test

The demo includes a one-click preset for:

```text
LiquidAI/LFM2.5-230M-GGUF
LFM2.5-230M-Q4_K_M.gguf
```

Programmatic use:

```ts
import {
  LFM25_230M_Q4_K_M,
  LlamaCppWasm
} from "llama-cpp-wasm";

await engine.load(
  { url: LFM25_230M_Q4_K_M.url },
  {
    contextSize: LFM25_230M_Q4_K_M.recommendedContextSize,
    batchSize: LFM25_230M_Q4_K_M.recommendedBatchSize,
    threads: 1
  }
);

for await (const text of engine.chat([
  { role: "system", content: "You are a concise assistant." },
  { role: "user", content: "Explain what GGUF is." }
])) {
  console.log(text);
}
```

`chat()` delegates formatting to the chat template embedded in the GGUF.
Version `0.0.2` intentionally accepts only one optional system message and
one user message. This avoids pretending to support multi-turn KV state before
that lifecycle is implemented correctly.


## GitHub Pages playground

Every push to `main` is built and deployed to:

```text
https://celsowm.github.io/llama-cpp-wasm/
```

The deployment workflow is:

```text
.github/workflows/deploy-pages.yml
```

Before the first deployment, set the repository's Pages source to
**GitHub Actions** under **Settings → Pages**. See `DEPLOYMENT.md` for the
complete flow.
