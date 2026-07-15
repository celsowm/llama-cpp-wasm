import { LFM25_230M_Q4_K_M, LlamaCppWasm } from "../src/index.ts";
import type { ChatMessage, ModelInfo } from "../src/index.ts";
import { renderMarkdown } from "./markdown.ts";

const DEFAULT_MAX_TOKENS = 512;
const DEFAULT_TEMPERATURE = 0.7;

const setupScreen = getElement<HTMLElement>("setup");
const chatScreen = getElement<HTMLElement>("chat");
const modelInput = getElement<HTMLInputElement>("model");
const contextInput = getElement<HTMLInputElement>("context");
const batchInput = getElement<HTMLInputElement>("batch");
const loadLfmButton = getElement<HTMLButtonElement>("loadLfm");
const progress = getElement<HTMLProgressElement>("progress");
const status = getElement<HTMLParagraphElement>("status");

const modelLabel = getElement<HTMLElement>("modelLabel");
const chatStatus = getElement<HTMLElement>("chatStatus");
const messagesEl = getElement<HTMLElement>("messages");
const composer = getElement<HTMLFormElement>("composer");
const promptInput = getElement<HTMLTextAreaElement>("prompt");
const sendButton = getElement<HTMLButtonElement>("send");
const cancelButton = getElement<HTMLButtonElement>("cancel");
const newChatButton = getElement<HTMLButtonElement>("newChat");
const unloadButton = getElement<HTMLButtonElement>("unload");

let engine: LlamaCppWasm | undefined;
let history: ChatMessage[] = [];
let busy = false;

loadLfmButton.addEventListener("click", async () => {
  contextInput.value = String(LFM25_230M_Q4_K_M.recommendedContextSize);
  batchInput.value = String(LFM25_230M_Q4_K_M.recommendedBatchSize);
  await loadSource({ url: LFM25_230M_Q4_K_M.url }, LFM25_230M_Q4_K_M.label);
});

modelInput.addEventListener("change", async () => {
  const file = modelInput.files?.[0];
  if (!file) {
    return;
  }
  await loadSource({ file }, file.name);
});

async function loadSource(
  source: { file: File } | { url: string },
  label: string
): Promise<void> {
  setSetupBusy(true);
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

    const info: ModelInfo = await engine.load(
      source,
      {
        contextSize: Number(contextInput.value),
        batchSize: Number(batchInput.value),
        threads: 1
      },
      update => {
        progress.value = update.ratio ?? 0;
        status.textContent =
          "Loading " + formatBytes(update.loadedBytes) +
          (update.totalBytes > 0
            ? " / " + formatBytes(update.totalBytes)
            : "");
      }
    );

    progress.value = 1;
    status.textContent =
      "Loaded " + formatBytes(info.bytesWritten) + ". " +
      "Context " + info.contextSize + ", batch " + info.batchSize + ".";

    history = [];
    messagesEl.replaceChildren();
    modelLabel.textContent = label;
    showChat();
    focusPrompt();
  } catch (error) {
    status.textContent = describeError(error);
    engine?.terminate();
    engine = undefined;
  } finally {
    setSetupBusy(false);
  }
}

composer.addEventListener("submit", async event => {
  event.preventDefault();
  await sendMessage();
});

promptInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void sendMessage();
  }
});

promptInput.addEventListener("input", () => {
  autoGrow(promptInput);
});

cancelButton.addEventListener("click", () => {
  engine?.cancel();
  chatStatus.textContent = "Stopping…";
});

newChatButton.addEventListener("click", () => {
  if (busy) {
    return;
  }
  history = [];
  messagesEl.replaceChildren();
  setChatStatus("Ready");
  focusPrompt();
});

unloadButton.addEventListener("click", () => {
  if (busy) {
    engine?.cancel();
  }
  engine?.terminate();
  engine = undefined;
  history = [];
  messagesEl.replaceChildren();
  showSetup();
});

async function sendMessage(): Promise<void> {
  if (busy || !engine) {
    return;
  }

  const text = promptInput.value.trim();
  if (text === "") {
    return;
  }

  history.push({ role: "user", content: text });
  appendMessage("user", text);

  promptInput.value = "";
  autoGrow(promptInput);

  const rendered = appendMessage("assistant", "");

  busy = true;
  setBusy(true);
  setChatStatus("Generating…");

  let accumulated = "";

  try {
    for await (const chunk of engine.chat(history, {
      maxTokens: DEFAULT_MAX_TOKENS,
      temperature: DEFAULT_TEMPERATURE,
      topK: 40,
      topP: 0.95
    })) {
      accumulated += chunk;
      rendered.innerHTML = renderMarkdown(accumulated) + '<span class="caret"></span>';
      scrollToBottom();
    }

    rendered.innerHTML = renderMarkdown(accumulated);
    history.push({ role: "assistant", content: accumulated });
    setChatStatus("Ready");
  } catch (error) {
    if (accumulated === "") {
      rendered.closest(".message")?.remove();
      history.pop();
    } else {
      rendered.innerHTML = renderMarkdown(accumulated);
      history.push({ role: "assistant", content: accumulated });
    }
    setChatStatus(describeError(error));
  } finally {
    busy = false;
    setBusy(false);
    autoGrow(promptInput);
    focusPrompt();
  }
}

function appendMessage(role: ChatMessage["role"], content: string): HTMLElement {
  const row = document.createElement("div");
  row.className = "message " + role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const label = document.createElement("div");
  label.className = "role-label";
  label.textContent = role === "user" ? "You" : "Assistant";

  const body = document.createElement("div");
  body.className = "content";
  if (content !== "") {
    body.innerHTML = renderMarkdown(content);
  }

  bubble.appendChild(label);
  bubble.appendChild(body);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();

  return body;
}

function setBusy(value: boolean): void {
  busy = value;
  sendButton.disabled = value;
  promptInput.disabled = value;
  newChatButton.disabled = value;
  cancelButton.classList.toggle("hidden", !value);
}

function setSetupBusy(value: boolean): void {
  loadLfmButton.disabled = value;
  modelInput.disabled = value;
  contextInput.disabled = value;
  batchInput.disabled = value;
}

function setChatStatus(text: string): void {
  chatStatus.textContent = text;
}

function showChat(): void {
  setupScreen.classList.add("hidden");
  chatScreen.classList.remove("hidden");
}

function showSetup(): void {
  chatScreen.classList.add("hidden");
  setupScreen.classList.remove("hidden");
}

function focusPrompt(): void {
  promptInput.focus();
}

function scrollToBottom(): void {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function autoGrow(textarea: HTMLTextAreaElement): void {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
}

function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error("Missing element #" + id + ".");
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

  return value.toFixed(value >= 100 ? 0 : 1) + " " + unit;
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function playgroundAssetUrl(path: string): string {
  const baseUrl = new URL(import.meta.env.BASE_URL, window.location.origin);
  return new URL(path, baseUrl).href;
}

window.addEventListener("beforeunload", () => {
  engine?.terminate();
});
