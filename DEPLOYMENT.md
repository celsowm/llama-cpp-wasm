# GitHub Pages deployment

Version `0.0.3` deploys the browser playground after every push to
`main`.

Target URL:

```text
https://celsowm.github.io/llama-cpp-wasm/
```

## One-time repository setting

In the GitHub repository, open:

```text
Settings → Pages → Build and deployment → Source
```

Select:

```text
GitHub Actions
```

The workflow cannot enable Pages with the default `GITHUB_TOKEN`;
this one-time setting must exist before the first successful deploy.

## Workflow

```text
.github/workflows/deploy-pages.yml
```

On each push to `main`, it:

1. Checks out the repository.
2. Reads the GitHub Pages base path.
3. Installs Node.js and Emscripten.
4. Fetches the pinned `llama.cpp` revision.
5. Type-checks the TypeScript library.
6. Compiles the WASM runtime.
7. Builds the Vite playground with the Pages project path.
8. Uploads `dist-demo`.
9. Deploys the artifact to the `github-pages` environment.

The workflow can also be started manually through
**Actions → Deploy playground to GitHub Pages → Run workflow**.

## Why the base-path handling is necessary

This is a project Pages site, not a root user site. Its assets live
below `/llama-cpp-wasm/`. Version `0.0.3` obtains that path from
`actions/configure-pages` and supplies it to Vite through
`VITE_BASE_PATH`.

The playground also resolves the Emscripten JavaScript and WASM files
from `import.meta.env.BASE_URL`, preventing requests to the incorrect
origin-root paths:

```text
/wasm/llama-cpp-wasm.js
/wasm/llama-cpp-wasm.wasm
```
