# AGENTS.md

Notes for AI agents working in this repository. This is the **llama-cpp-wasm**
wrapper repo (the directory containing this file). It is a separate project
from the vendored `vendor/llama.cpp/` snapshot.

## Repository layout

```text
cpp/bridge.cpp        C++ bridge compiled to WASM via Emscripten
scripts/              fetch-llama.sh, build-wasm.sh
src/                  TypeScript runtime, worker, presets, types
demo/                 Vite playground (main.ts, markdown.ts, index.html, public/)
dist/                 compiled library output (tsc + copied wasm/)
dist-demo/            compiled playground output (vite build)
vendor/llama.cpp/     PINNED upstream snapshot (do not edit; see below)
```

The vendored `vendor/llama.cpp/AGENTS.md` belongs to the upstream llama.cpp
project, not this wrapper. Treat it as a pinned third-party file: edits to it
do not survive `npm run vendor:llama`, and its AI-contribution policy
describes upstream PRs to llama.cpp, not work in this repository.

## Build

```bash
npm install
npm run vendor:llama    # check out the pinned llama.cpp revision
npm run build:wasm     # builds BOTH llama_cpp_wasm_st and llama_cpp_wasm_mt
npm run build:ts       # tsc emits dist/
npm run build          # build:wasm && build:ts
npm run dev            # Vite playground at http://localhost:5173
npm run build:pages    # build + vite build for GitHub Pages
```

`build:wasm` requires an activated Emscripten SDK (`emcmake` and `em++` in
`PATH`) and `ninja`. It emits four artifacts into `dist/wasm/` AND
`demo/public/wasm/`:

```text
llama-cpp-wasm-st.js / .wasm
llama-cpp-wasm-mt.js / .wasm
llama-cpp-wasm-mt.worker.js   (pthread worker; required by the MT build)
```

The MT build adds `-pthread`, `-sUSE_PTHREADS=1`, `-sPTHREAD_POOL_SIZE=4`.
To widen the pool cap for an 8-thread benchmark, edit `CMakeLists.txt`
(`PTHREAD_POOL_SIZE`) — do not assume default 8.

## Verify

```bash
npm test               # tsc --noEmit (no WASM build)
npm run typecheck      # same
```

There is no in-repo command that builds WASM AND typechecks together. To
gate a change end-to-end you must run `npm run build:wasm` manually with
Emscripten, then `npm test`, then `npm run dev` for a browser smoke pass.

The pre-existing `tests/browser/smoke.mjs` references element IDs
(`#generate`, `#generateRaw`, `#output`) that no longer exist in the current
playground. It is stale tech debt and is not a correctness gate for the
current UI. Do NOT treat it as authoritative; if you fix the playground,
consider fixing or deleting the smoke test in the same change.

## Runtime architecture (post-0.0.4)

- `llama_context` is created once in `lcw_load_model` and preserved across
  turns. `lcw_start` performs prefix comparison and rewinds only the
  divergent tail of the KV cache via `llama_memory_seq_rm`; only the newly
  appended user tokens are re-evaluated. Turning a fresh `lcw_start` into
  a full re-tokenization is a regression and must be avoided.
- `lcw_next` is gone. The worker drives `lcw_generate_chunk` (multiple
  tokens per native call) plus `lcw_finished` for completion detection.
  Do NOT reintroduce a per-token `lcw_next`; the chunked path is what
  removes one JS/WASM round trip per token.
- `lcw_reset_kv` clears the cache without unloading. The playground calls
  it when switching sidebar conversations. Keep this lifecycle; do not
  fold it back into `lcw_unload`.
- The MEMFS model file is unlinked in `src/worker.ts` immediately after
  `lcw_load_model` returns 0. Removing that unlink silently reintroduces
  ~150 MiB (230M) / ~731 MiB (1.2B) of avoidable heap pressure.
- `LlamaCppWasm.createThreaded()` picks ST or MT at runtime using
  `crossOriginIsolated` + `typeof SharedArrayBuffer`. The ST and MT URLs
  must both be passed; if you add a new artifact, update both
  `ThreadedRuntimeAssets` and the demo's `playgroundAssetUrl` calls.
- The deployed site (GitHub Pages) must serve
  `Cross-Origin-Opener-Policy: same-origin` AND
  `Cross-Origin-Embedder-Policy: require-corp`, otherwise the runtime
  silently falls back to the single-threaded build. The Vite dev server
  already sends these headers.

## C++ bridge pin

`vendor/llama.cpp` is pinned via `LLAMA_CPP_COMMIT` in
`scripts/fetch-llama.sh`. The bridge in `cpp/bridge.cpp` calls specific
post-refactor llama.cpp APIs (e.g. `llama_model_load_from_file`,
`llama_memory_seq_rm`, `llama_sampler_chain_*`). Bumping the pin without
verifying the llama.cpp ABI is correct WILL break the build or silently
misbehave. If you bump the pin:

1. Verify `llama.h` API signatures still match the calls in
   `cpp/bridge.cpp`.
2. Cross-check the `EXPORTED_FUNCTIONS` list in `CMakeLists.txt` against
   the `EMSCRIPTEN_KEEPALIVE` entries in `cpp/bridge.cpp`.
3. Run `npm run build:wasm` and `npm test`.

## Coding conventions

- Do not add comments explaining what code already tells you. Reserve
  comments for non-obvious invariants (e.g. why the headroom check makes
  the chunked decode loop invariant-preserving).
- No emdash `—`, unicode arrow `→`, or `…` characters in code or commit
  messages. Use ASCII `-`, `->`, `...`. The existing codebase already
  follows this convention.
- TypeScript strictness: `tsconfig.json` enables `strict`,
  `noUncheckedIndexedAccess`. New code must pass `npm run typecheck`.
  The `demo/` tree is compiled by Vite, not by `tsconfig.json`, so the
  demo cancompile with pre-existing strict errors that the production
  `npm test` does NOT enforce. Keep demo changes consistent with the
  existing demo style rather than adding type: any or non-null assertions
  to satisfy an unrelated tsc invocation.

## Git / PR hygiene

Do NOT commit or push without explicit user instruction. Earlier in this
project the user confirmed "implement all code changes; you verify" — that
scope does NOT include git commits or pushes. If the user asks for a
commit/push:

- Inspect `git status`, `git diff`, and `git log --oneline -10` first.
- Write a concise commit message in the repo's existing style (lowercase
  `perf:`, `feat:`, `fix:`, `refactor:` prefixes; ASCII only).
- Do NOT update git config, skip hooks, force-push, or create empty
  commits.
- Do NOT amend on the user's behalf if a commit fails; fix and create a
  new commit.
- For GitHub PRs, use `gh` and return the PR URL.

The `vendor/llama.cpp/AGENTS.md` policy on AI-authored PRs applies to
contributions to **upstream llama.cpp**, not to this wrapper. Work in this
repository is fine; do not submit AI-generated PRs to `ggml-org/llama.cpp`.
