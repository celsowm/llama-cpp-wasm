import {
  LFM25_230M_Q4_K_M,
  LFM25_1_2B_INSTRUCT_Q4_K_M,
  GEMMA4_E2B_IT_Q4_0,
  LlamaCppWasm
} from "../src/index.ts";
import type { ChatImage, ChatMessage, ModelInfo, ModelPreset } from "../src/index.ts";
import { renderMarkdown } from "./markdown.ts";

const DEFAULT_MAX_TOKENS = 512;
const DEFAULT_TEMPERATURE = 0.7;

const PRESETS: Record<string, ModelPreset> = {
  LFM25_230M_Q4_K_M,
  LFM25_1_2B_INSTRUCT_Q4_K_M,
  GEMMA4_E2B_IT_Q4_0
};

interface Conversation {
  id: string;
  title: string;
  history: ChatMessage[];
}

const setupScreen = getElement<HTMLElement>("setup");
const chatScreen = getElement<HTMLElement>("chat");
const modelSelect = getElement<HTMLSelectElement>("modelSelect");
const customModelField = getElement<HTMLElement>("customModelField");
const modelInput = getElement<HTMLInputElement>("model");
const contextInput = getElement<HTMLInputElement>("context");
const batchInput = getElement<HTMLInputElement>("batch");
const loadLfmButton = getElement<HTMLButtonElement>("loadLfm");
const progress = getElement<HTMLProgressElement>("progress");
const status = getElement<HTMLParagraphElement>("status");

const sidebar = getElement<HTMLElement>("sidebar");
const sidebarToggle = getElement<HTMLButtonElement>("sidebarToggle");
const sidebarReopen = getElement<HTMLButtonElement>("sidebarReopen");
const conversationsEl = getElement<HTMLElement>("conversations");
const newChatTop = getElement<HTMLButtonElement>("newChatTop");
const openSetup = getElement<HTMLButtonElement>("openSetup");

const modelLabel = getElement<HTMLElement>("modelLabel");
const chatStatus = getElement<HTMLElement>("chatStatus");
const messagesEl = getElement<HTMLElement>("messages");
const composer = getElement<HTMLFormElement>("composer");
const promptInput = getElement<HTMLTextAreaElement>("prompt");
const attachInput = getElement<HTMLInputElement>("attach");
const sendButton = getElement<HTMLButtonElement>("send");
const cancelButton = getElement<HTMLButtonElement>("cancel");
const newChatButton = getElement<HTMLButtonElement>("newChat");
const unloadButton = getElement<HTMLButtonElement>("unload");

let engine: LlamaCppWasm | undefined;
let busy = false;
let conversations: Conversation[] = [];
let active: Conversation | undefined;
// A decoded image queued for the next user turn (one image per turn).
let pendingImage: ChatImage | undefined;

modelSelect.addEventListener("change", () => {
  customModelField.classList.toggle("hidden", modelSelect.value !== "custom");
});

loadLfmButton.addEventListener("click", async () => {
  const choice = modelSelect.value;
  const label = modelSelect.options[modelSelect.selectedIndex].textContent ?? choice;

  let source: { file: File } | { url: string };
  if (choice === "custom") {
    const file = modelInput.files?.[0];
    if (!file) {
      status.textContent = "Please choose a GGUF file.";
      return;
    }
    source = { file };
  } else {
    const preset = PRESETS[choice];
    if (!preset) {
      status.textContent = "Unknown model selection.";
      return;
    }
    source = { url: preset.url };
    contextInput.value = String(preset.recommendedContextSize);
    batchInput.value = String(preset.recommendedBatchSize);
  }

  await loadSource(source, label, choice !== "custom" ? PRESETS[choice].mmproj : undefined);
});

async function loadSource(
  source: { file: File } | { url: string },
  label: string,
  mmproj?: { url: string }
): Promise<void> {
  setSetupBusy(true);
  status.textContent = "Initializing WASM…";
  progress.value = 0;

  try {
    engine?.terminate();

    engine = await LlamaCppWasm.createThreaded({
      workerFactory: () =>
        new Worker(new URL("../src/worker.ts", import.meta.url), {
          type: "module",
          name: "llama-cpp-wasm-playground"
        }),
      moduleUrlSt: playgroundAssetUrl("wasm/llama-cpp-wasm-st.js"),
      wasmUrlSt: playgroundAssetUrl("wasm/llama-cpp-wasm-st.wasm"),
      moduleUrlMt: playgroundAssetUrl("wasm/llama-cpp-wasm-mt.js"),
      wasmUrlMt: playgroundAssetUrl("wasm/llama-cpp-wasm-mt.wasm")
    });

    status.textContent = "Copying GGUF into the worker…";

    const info: ModelInfo = await engine.load(
      source,
      {
        contextSize: Number(contextInput.value),
        batchSize: Number(batchInput.value)
      },
      update => {
        progress.value = update.ratio ?? 0;
        status.textContent =
          "Loading " + formatBytes(update.loadedBytes) +
          (update.totalBytes > 0
            ? " / " + formatBytes(update.totalBytes)
            : "");
      },
      mmproj ? { url: mmproj.url } : undefined
    );

    progress.value = 1;
    status.textContent =
      "Loaded " + formatBytes(info.bytesWritten) + ". " +
      "Context " + info.contextSize + ", batch " + info.batchSize + ", " +
      "threads " + info.threads +
      (info.mmprojLoaded ? ", vision enabled" : ".") + ".";

    modelLabel.textContent = label;
    conversations = [];
    conversationsEl.replaceChildren();
    showChat();
    startNewChat();
    focusPrompt();
  } catch (error) {
    status.textContent = describeError(error);
    engine?.terminate();
    engine = undefined;
  } finally {
    setSetupBusy(false);
  }
}

function startNewChat(): void {
  if (busy) {
    return;
  }

  if (engine) {
    void engine.resetKV().catch(() => { /* logged on the worker */ });
  }
  messagesEl.replaceChildren();
  const conversation: Conversation = {
    id: "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    title: "New chat",
    history: []
  };
  conversations.unshift(conversation);
  active = conversation;

  renderConversationList();
  setChatStatus("Ready");
  focusPrompt();
}

function renderConversationList(): void {
  conversationsEl.replaceChildren();

  for (const conversation of conversations) {
    const row = document.createElement("div");
    row.className =
      "conversation" + (conversation === active ? " active" : "");
    row.title = conversation.title;

    const title = document.createElement("span");
    title.className = "title";
    title.textContent = conversation.title;

    const del = document.createElement("button");
    del.className = "delete";
    del.type = "button";
    del.title = "Delete conversation";
    del.textContent = "×";
    del.addEventListener("click", event => {
      event.stopPropagation();
      deleteConversation(conversation);
    });

    row.appendChild(title);
    row.appendChild(del);
    row.addEventListener("click", () => selectConversation(conversation));

    conversationsEl.appendChild(row);
  }
}

function selectConversation(conversation: Conversation): void {
  if (busy || conversation === active) {
    return;
  }

  active = conversation;
  // The persistent KV cache currently holds the previous conversation's
  // tokens. Since switching to an unrelated sidebar conversation rarely shares
  // a prefix, clear it instead of forcing `lcw_start` to truncate the entire
  // cache. Async fire-and-forget; can't race with generation because `busy` is
  // false (we just returned early above if it was).
  if (engine) {
    void engine.resetKV().catch(() => { /* logged on the worker */ });
  }
  renderConversationList();
  renderHistory(conversation.history);
  setChatStatus("Ready");
  focusPrompt();
}

function renderHistory(history: ChatMessage[]): void {
  messagesEl.replaceChildren();
  for (const message of history) {
    appendMessage(message.role, message.content, message.images);
  }
}

function deleteConversation(conversation: Conversation): void {
  if (busy) {
    return;
  }

  const index = conversations.indexOf(conversation);
  if (index === -1) {
    return;
  }
  conversations.splice(index, 1);

  if (conversation === active) {
    if (conversations.length > 0) {
      selectConversation(conversations[0]);
    } else {
      startNewChat();
    }
    return;
  }

  renderConversationList();
}

composer.addEventListener("submit", async event => {
  event.preventDefault();
  await sendMessage();
});

// Decode the chosen image file to raw RGB (length == width * height * 3) so it
// can be passed straight to the mtmd bitmap path on the worker. Each user turn
// holds at most one image, matching the single-marker constraint of
// lcw_eval_image.
attachInput.addEventListener("change", async () => {
  const file = attachInput.files?.[0];
  if (!file) {
    pendingImage = undefined;
    return;
  }
  try {
    pendingImage = await decodeImageToRgb(file);
  } catch {
    pendingImage = undefined;
    alert("Could not decode the selected image.");
  }
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
  startNewChat();
  focusPrompt();
});

newChatTop.addEventListener("click", () => {
  startNewChat();
  focusPrompt();
});

unloadButton.addEventListener("click", () => {
  if (busy) {
    engine?.cancel();
  }
  engine?.terminate();
  engine = undefined;
  conversations = [];
  conversationsEl.replaceChildren();
  active = undefined;
  showSetup();
});

openSetup.addEventListener("click", () => {
  if (busy) {
    engine?.cancel();
  }
  engine?.terminate();
  engine = undefined;
  showSetup();
});

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.add("collapsed");
  sidebarToggle.classList.add("hidden");
  sidebarReopen.classList.remove("hidden");
});

sidebarReopen.addEventListener("click", () => {
  sidebar.classList.remove("collapsed");
  sidebarToggle.classList.remove("hidden");
  sidebarReopen.classList.add("hidden");
});

async function sendMessage(): Promise<void> {
  if (busy || !engine || !active) {
    return;
  }

  const text = promptInput.value.trim();
  if (text === "" && pendingImage === undefined) {
    return;
  }

  if (active.title === "New chat" && text !== "") {
    active.title = text.slice(0, 40);
    renderConversationList();
  }

  const images: ChatImage[] = pendingImage ? [pendingImage] : [];
  pendingImage = undefined;
  attachInput.value = "";

  // Gemma 4 (and most chat templates) expect the user turn to carry text after
  // the media marker. When only an image is attached, substitute a default
  // caption so the rendered turn is never empty and the model gets a real
  // instruction.
  const hasText = text !== "";
  const promptText = hasText ? text : "Describe this image.";
  active.history.push({
    role: "user",
    content: promptText,
    images: images.length > 0 ? images : undefined
  });
  appendMessage("user", promptText, images);

  promptInput.value = "";
  autoGrow(promptInput);

  const rendered = appendMessage("assistant", "");

  let indicator: HTMLElement | undefined;
  let indicatorTimer: number | undefined;

  // The prompt-eval phase blocks before any token is produced. Show a
  // spinner only after a short grace period so a snappy first token does
  // not flash an unneeded dot. The indicator is cleared on first chunk.
  const startIndicator = (): void => {
    if (indicatorTimer !== undefined) return;
    indicatorTimer = window.setTimeout(() => {
      indicatorTimer = undefined;
      indicator = document.createElement("div");
      indicator.className = "thinking-indicator";
      indicator.innerHTML =
        '<span class="dots"><span></span><span></span><span></span></span>' +
        '<span>Thinking…</span>';
      rendered.appendChild(indicator);
      scrollToBottom();
    }, 200);
  };

  const stopIndicator = (): void => {
    if (indicatorTimer !== undefined) {
      clearTimeout(indicatorTimer);
      indicatorTimer = undefined;
    }
    if (indicator) {
      indicator.remove();
      indicator = undefined;
    }
  };

  startIndicator();

  busy = true;
  setBusy(true);
  setChatStatus("Generating…");

  let accumulated = "";
  let dirty = false;
  let rafHandle: number | undefined;
  let done = false;
  let startTime = performance.now();

  // Schedule at most one DOM reflow per animation frame. The model produces
  // several chunks quickly and we must not re-parse the entire accumulated
  // Markdown on every chunk - the O(n^2) reflow cost dominates long answers.
  const scheduleRender = (): void => {
    if (dirty || done) return;
    dirty = true;
    rafHandle = requestAnimationFrame(() => {
      dirty = false;
      rafHandle = undefined;
      rendered.innerHTML =
        renderMarkdown(accumulated) + '<span class="caret"></span>';
      scrollToBottom();
    });
  };

  try {
    for await (const chunk of engine.chat(active.history, {
      maxTokens: DEFAULT_MAX_TOKENS,
      temperature: DEFAULT_TEMPERATURE,
      topK: 40,
      topP: 0.95
    })) {
      if (accumulated === "") {
        stopIndicator();
      }
      accumulated += chunk;
      scheduleRender();
    }

    stopIndicator();

    // Drain any pending frame before swapping to the final rendering.
    if (rafHandle !== undefined) {
      cancelAnimationFrame(rafHandle);
      rafHandle = undefined;
    }
    done = true;
    rendered.innerHTML = renderMarkdown(accumulated);
    scrollToBottom();

    active.history.push({ role: "assistant", content: accumulated });
    setChatStatus(summarizePerformance(engine, startTime, accumulated.length));
  } catch (error) {
    stopIndicator();
    if (rafHandle !== undefined) {
      cancelAnimationFrame(rafHandle);
      rafHandle = undefined;
    }
    if (accumulated === "") {
      rendered.closest(".message")?.remove();
      active.history.pop();
    } else {
      done = true;
      rendered.innerHTML = renderMarkdown(accumulated);
      active.history.push({ role: "assistant", content: accumulated });
    }
    setChatStatus(describeError(error));
  } finally {
    busy = false;
    setBusy(false);
    autoGrow(promptInput);
    focusPrompt();
  }
}

function appendMessage(
  role: ChatMessage["role"],
  content: string,
  images?: ChatImage[]
): HTMLElement {
  const row = document.createElement("div");
  row.className = "message " + role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const label = document.createElement("div");
  label.className = "role-label";
  label.textContent = role === "user" ? "You" : "Assistant";

  const body = document.createElement("div");
  body.className = "content";
  if (images && images.length > 0) {
    for (const image of images) {
      body.appendChild(renderImageThumbnail(image));
    }
  }
  if (content !== "") {
    body.innerHTML = (body.innerHTML ? body.innerHTML + "<br/>" : "") + renderMarkdown(content);
  }

  bubble.appendChild(label);
  bubble.appendChild(body);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();

  return body;
}

// Render raw RGB pixels (length === width * height * 3) as a small thumbnail by
// stamping them into an ImageData and drawing onto a 2D canvas, then exporting
// the canvas as a data URL. Used by both the live composer and renderHistory.
function renderImageThumbnail(image: ChatImage): HTMLImageElement {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return document.createElement("img");
  }
  const rgba = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0, j = 0; i < image.data.length; i += 3, j += 4) {
    rgba[j] = image.data[i];
    rgba[j + 1] = image.data[i + 1];
    rgba[j + 2] = image.data[i + 2];
    rgba[j + 3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, image.width, image.height), 0, 0);
  const img = document.createElement("img");
  img.src = canvas.toDataURL("image/png");
  img.className = "chat-image";
  img.alt = "Attached image";
  // Constrain the thumbnail's display size via inline styles; CSS sizing keeps
  // the actual pixels until the data URL is decoded.
  img.style.maxWidth = "200px";
  img.style.maxHeight = "200px";
  return img;
}

function setBusy(value: boolean): void {
  busy = value;
  sendButton.disabled = value;
  promptInput.disabled = value;
  newChatButton.disabled = value;
  newChatTop.disabled = value;
  cancelButton.classList.toggle("hidden", !value);
}

function setSetupBusy(value: boolean): void {
  loadLfmButton.disabled = value;
  modelSelect.disabled = value;
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

// Decode an image File into raw RGB pixels. We draw it onto a 2D canvas and
// read back the 8-bit RGBA buffer, then drop the alpha channel. The plane is
// height x width x 3, matching mtmd_bitmap_init's RGBRGB... expectation.
async function decodeImageToRgb(file: File): Promise<ChatImage> {
  const bitmap = await createImageBitmap(file);
  try {
    const width = bitmap.width;
    const height = bitmap.height;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Canvas 2D context unavailable.");
    }
    ctx.drawImage(bitmap, 0, 0, width, height);
    const rgba = ctx.getImageData(0, 0, width, height).data;
    const data = new Uint8Array(width * height * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      data[j] = rgba[i];
      data[j + 1] = rgba[i + 1];
      data[j + 2] = rgba[i + 2];
    }
    return { data, width, height };
  } finally {
    bitmap.close();
  }
}

/**
 * Composes a single-line chat-status string that summarises both wall-clock
 * and the native-side benchmark counters. Prompt tokens/second is dominated
 * by llama_decode's prompt batches; generated tokens/second is the
 * single-token decode rate; time-to-first-token is the prompt phase plus the
 * wall-clock the worker spent in lcw_start before the first chunk arrived.
 */
function summarizePerformance(
  engine: LlamaCppWasm,
  startedAt: number,
  chars: number
): string {
  const wallMs = performance.now() - startedAt;
  const bench = engine.lastBench;
  const parts: string[] = [
    `${chars} chars in ${wallMs.toFixed(0)} ms`
  ];
  if (bench) {
    const promptTps = bench.promptMs > 0
      ? (bench.promptTokens / (bench.promptMs / 1000)).toFixed(1)
      : "0";
    const genTps = bench.generateMs > 0
      ? (bench.generateTokens / (bench.generateMs / 1000)).toFixed(1)
      : "0";
    parts.push(`prompt ${promptTps} tok/s`);
    parts.push(`gen ${genTps} tok/s`);
    parts.push(`TTFT ${bench.promptMs.toFixed(0)} ms`);
    if (bench.loadMs > 0) {
      parts.push(`load ${bench.loadMs.toFixed(0)} ms`);
    }
  }
  return `Ready - ${parts.join(", ")}`;
}

function playgroundAssetUrl(path: string): string {
  const baseUrl = new URL(import.meta.env.BASE_URL, window.location.origin);
  return new URL(path, baseUrl).href;
}

window.addEventListener("beforeunload", () => {
  engine?.terminate();
});
