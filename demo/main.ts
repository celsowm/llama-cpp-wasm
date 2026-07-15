import { LFM25_230M_Q4_K_M, LlamaCppWasm } from "../src/index.ts";

const modelInput = getElement<HTMLInputElement>("model");
const contextInput = getElement<HTMLInputElement>("context");
const batchInput = getElement<HTMLInputElement>("batch");
const maxTokensInput = getElement<HTMLInputElement>("maxTokens");
const temperatureInput = getElement<HTMLInputElement>("temperature");
const promptInput = getElement<HTMLTextAreaElement>("prompt");
const loadButton = getElement<HTMLButtonElement>("load");
const loadLfmButton = getElement<HTMLButtonElement>("loadLfm");
const generateButton = getElement<HTMLButtonElement>("generate");
const generateRawButton = getElement<HTMLButtonElement>("generateRaw");
const cancelButton = getElement<HTMLButtonElement>("cancel");
const progress = getElement<HTMLProgressElement>("progress");
const status = getElement<HTMLParagraphElement>("status");
const output = getElement<HTMLPreElement>("output");

let engine: LlamaCppWasm | undefined;

loadButton.addEventListener("click", async () => {
  const file = modelInput.files?.[0];
  if (!file) {
    status.textContent = "Choose a GGUF file first.";
    return;
  }

  await loadSource({ file });
});

loadLfmButton.addEventListener("click", async () => {
  contextInput.value = String(LFM25_230M_Q4_K_M.recommendedContextSize);
  batchInput.value = String(LFM25_230M_Q4_K_M.recommendedBatchSize);
  await loadSource({ url: LFM25_230M_Q4_K_M.url });
});

async function loadSource(
  source: { file: File } | { url: string }
): Promise<void> {
  setBusy(true);
  status.textContent = "Initializing WASM…";
  progress.value = 0;

  try {
    engine?.terminate();

    engine = await LlamaCppWasm.create({
      workerFactory: () =>
        new Worker(new URL("../src/worker.ts", import.meta.url), {
          type: "module",
          name: "llama-cpp-wasm-playground"
        }),
      moduleUrl: playgroundAssetUrl("wasm/llama-cpp-wasm.js"),
      wasmUrl: playgroundAssetUrl("wasm/llama-cpp-wasm.wasm")
    });

    status.textContent = "Copying GGUF into the worker…";

    const info = await engine.load(
      source,
      {
        contextSize: Number(contextInput.value),
        batchSize: Number(batchInput.value),
        threads: 1
      },
      update => {
        progress.value = update.ratio ?? 0;
        status.textContent =
          `Loading ${formatBytes(update.loadedBytes)}` +
          (update.totalBytes > 0
            ? ` / ${formatBytes(update.totalBytes)}`
            : "");
      }
    );

    progress.value = 1;
    status.textContent =
      `Loaded ${formatBytes(info.bytesWritten)}. ` +
      `Context ${info.contextSize}, batch ${info.batchSize}.`;

    generateButton.disabled = false;
    generateRawButton.disabled = false;
  } catch (error) {
    status.textContent = describeError(error);
    engine?.terminate();
    engine = undefined;
  } finally {
    setBusy(false);
  }
}

generateButton.addEventListener("click", async () => {
  if (!engine) {
    return;
  }

  output.textContent = "";
  generateButton.disabled = true;
  cancelButton.disabled = false;
  status.textContent = "Generating…";

  try {
    for await (const text of engine.chat(
      [{ role: "user", content: promptInput.value }],
      {
        maxTokens: Number(maxTokensInput.value),
        temperature: Number(temperatureInput.value),
        topK: 40,
        topP: 0.95
      }
    )) {
      output.textContent += text;
    }

    status.textContent = "Generation complete.";
  } catch (error) {
    status.textContent = describeError(error);
  } finally {
    generateButton.disabled = false;
    cancelButton.disabled = true;
  }
});

cancelButton.addEventListener("click", () => {
  engine?.cancel();
  status.textContent = "Cancellation requested…";
});

generateRawButton.addEventListener("click", async () => {
  if (!engine) {
    return;
  }

  output.textContent = "";
  generateButton.disabled = true;
  generateRawButton.disabled = true;
  cancelButton.disabled = false;
  status.textContent = "Generating (raw)…";

  try {
    for await (const text of engine.completion(
      promptInput.value,
      {
        maxTokens: Number(maxTokensInput.value),
        temperature: Number(temperatureInput.value),
        topK: 40,
        topP: 0.95
      }
    )) {
      output.textContent += text;
    }

    status.textContent = "Generation complete.";
  } catch (error) {
    status.textContent = describeError(error);
  } finally {
    generateButton.disabled = false;
    generateRawButton.disabled = false;
    cancelButton.disabled = true;
  }
});

window.addEventListener("beforeunload", () => {
  engine?.terminate();
});

function setBusy(busy: boolean): void {
  loadButton.disabled = busy;
  loadLfmButton.disabled = busy;
  modelInput.disabled = busy;
  contextInput.disabled = busy;
  batchInput.disabled = busy;

  if (busy) {
    generateButton.disabled = true;
    generateRawButton.disabled = true;
  }
}

function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}.`);
  }
  return element as T;
}

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = units[0];

  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }

  return `${value.toFixed(value >= 100 ? 0 : 1)} ${unit}`;
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}


function playgroundAssetUrl(path: string): string {
  const baseUrl = new URL(import.meta.env.BASE_URL, window.location.origin);
  return new URL(path, baseUrl).href;
}
