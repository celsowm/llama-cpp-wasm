# v0.0.2 test procedure

## Target

- Repository: `LiquidAI/LFM2.5-230M-GGUF`
- Quant: `LFM2.5-230M-Q4_K_M.gguf`

## Automated source checks

```bash
npm install
npm test
```

## Browser smoke test

```bash
npm run vendor:llama
npm run build
npm run dev
```

1. Open the demo.
2. Click **Load LFM2.5 230M Q4_K_M**.
3. Confirm download progress reaches 100%.
4. Generate with the default prompt.
5. Confirm streamed output and no unsupported-architecture error.
6. Generate with temperature `0` and confirm deterministic greedy output.
7. Start a long generation, click **Cancel**, and confirm it stops.
8. Reload the model and generate again to exercise unload/reload cleanup.

## Expected failure tests

- Empty chat messages: rejected in TypeScript.
- Assistant history in v0.0.2: rejected explicitly.
- Prompt plus max tokens beyond context: rejected by the native bridge.
- Model without a supported chat template: rejected with a native error.
