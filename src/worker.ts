/// <reference lib="webworker" />

import type { WorkerRequest, WorkerResponse } from "./protocol.js";
import type { ChatImage, ModelSource } from "./types.js";

interface EmscriptenFileSystem {
  mkdir(path: string): void;
  open(path: string, flags: string): unknown;
  write(
    stream: unknown,
    buffer: Uint8Array,
    offset: number,
    length: number,
    position: number
  ): number;
  close(stream: unknown): void;
  unlink(path: string): void;
}

interface EmscriptenMemory {
  buffer: ArrayBuffer;
  grow(delta: number): number;
}

interface LlamaModule {
  FS: EmscriptenFileSystem;
  HEAPU8: Uint8Array;
  _malloc(size: number): number;
  _free(pointer: number): void;
  cwrap(
    name: string,
    returnType: string | null,
    argumentTypes: string[]
  ): (...args: unknown[]) => unknown;
  wasmMemory?: EmscriptenMemory;
}

interface Bindings {
  formatChat(
    messagesSerialized: string,
    outputPointer: number,
    outputCapacity: number
  ): number;
  formatChatMm(
    messagesSerialized: string,
    outputPointer: number,
    outputCapacity: number
  ): number;
  loadModel(
    path: string,
    contextSize: number,
    batchSize: number,
    threads: number
  ): number;
  loadMmproj(
    path: string,
    threads: number
  ): number;
  mmprojLoaded(): number;
  evalImage(
    prompt: string,
    rgbPointer: number,
    nx: number,
    ny: number,
    maxTokens: number,
    temperature: number,
    topK: number,
    topP: number,
    seed: number
  ): number;
  start(
    prompt: string,
    maxTokens: number,
    temperature: number,
    topK: number,
    topP: number,
    seed: number
  ): number;
  generateChunk(
    outputPointer: number,
    outputCapacity: number,
    maxTokens: number
  ): number;
  finished(): number;
  resetKV(): void;
  unload(): void;
  lastError(): number;
  benchPromptMs(): number;
  benchGenerateMs(): number;
  benchPromptTokens(): number;
  benchGenerateTokens(): number;
  benchPromptBatch(): number;
  benchLoadMs(): number;
}

interface BenchmarkReport {
  promptMs: number;
  generateMs: number;
  promptTokens: number;
  generateTokens: number;
  promptBatchSize: number;
  loadMs: number;
}

const scope = self as DedicatedWorkerGlobalScope;
const MODEL_DIRECTORY = "/models";
const MODEL_PATH = `${MODEL_DIRECTORY}/model.gguf`;
const MMPROJ_PATH = `${MODEL_DIRECTORY}/mmproj.gguf`;
const OUTPUT_CAPACITY = 8192;
const CHUNK_TOKENS = 8;

// Auto-unlink the MEMFS model file as soon as llama.cpp has finished loading the
// tensors. The GGUF bytes are no longer needed by llama.cpp when
// `use_mmap = false` (the C++ bridge pins that off), so keeping them only
// inflates peak heap by the model size. The demo's 1.2B preset is the worst
// offender (~731 MiB retained). Emscripten allows FS.unlink() on an open path,
// and llama.cpp closes its FILE* before returning from
// llama_model_load_from_file(), so this is safe immediately after loadModel
// succeeds. The flag toggles so the smoke test or benchmarks can disable the
// cleanup for verification if needed.
const UNLINK_MODEL_AFTER_LOAD = true;

let moduleInstance: LlamaModule | undefined;
let bindings: Bindings | undefined;
let activeGenerationId: number | undefined;
let cancelRequested = false;

scope.addEventListener("message", event => {
  void dispatch(event.data as WorkerRequest);
});

async function dispatch(message: WorkerRequest): Promise<void> {
  try {
    switch (message.type) {
      case "init":
        await initialize(message.requestId, message.assets.moduleUrl, message.assets.wasmUrl);
        break;

      case "load":
        await loadModel(message);
        break;

      case "generate":
        await generate(message);
        break;

      case "cancel":
        if (message.generationId === activeGenerationId) {
          cancelRequested = true;
        }
        break;

      case "resetKV": {
        if (activeGenerationId !== undefined) {
          throw new Error("Cannot reset the KV cache while generation is active.");
        }
        const b = requireBindings();
        b.resetKV();
        post({ type: "kvReset", requestId: message.requestId });
        break;
      }

      case "unload":
        if (activeGenerationId !== undefined) {
          throw new Error("Cannot unload the model during generation.");
        }
        requireBindings().unload();
        removeModelFile();
        post({ type: "unloaded", requestId: message.requestId });
        break;
    }
  } catch (error) {
    postError(message.requestId, error);
  }
}

async function initialize(
  requestId: number,
  moduleUrl: string,
  wasmUrl: string
): Promise<void> {
  if (moduleInstance) {
    post({ type: "ready", requestId });
    return;
  }

  const imported = (await import(
    /* @vite-ignore */
    moduleUrl
  )) as {
    default?: (options: Record<string, unknown>) => Promise<LlamaModule>;
  };

  if (typeof imported.default !== "function") {
    throw new Error("The Emscripten module does not export a default factory.");
  }

  moduleInstance = await imported.default({
    noInitialRun: true,
    locateFile(path: string): string {
      return path.endsWith(".wasm") ? wasmUrl : path;
    },
    print: (...values: unknown[]) => console.log("[llama.cpp]", ...values),
    printErr: (...values: unknown[]) => console.error("[llama.cpp]", ...values)
  });

  const cwrap = moduleInstance.cwrap.bind(moduleInstance);

  bindings = {
    formatChat: cwrap(
      "lcw_format_chat_multi",
      "number",
      ["string", "number", "number"]
    ) as Bindings["formatChat"],

    formatChatMm: cwrap(
      "lcw_format_chat_mm",
      "number",
      ["string", "number", "number"]
    ) as Bindings["formatChatMm"],

    loadModel: cwrap(
      "lcw_load_model",
      "number",
      ["string", "number", "number", "number"]
    ) as Bindings["loadModel"],

    loadMmproj: cwrap(
      "lcw_load_mmproj",
      "number",
      ["string", "number"]
    ) as Bindings["loadMmproj"],

    mmprojLoaded: cwrap(
      "lcw_mmproj_loaded",
      "number",
      []
    ) as Bindings["mmprojLoaded"],

    evalImage: cwrap(
      "lcw_eval_image",
      "number",
      ["string", "number", "number", "number", "number", "number", "number", "number", "number"]
    ) as Bindings["evalImage"],

    start: cwrap(
      "lcw_start",
      "number",
      ["string", "number", "number", "number", "number", "number"]
    ) as Bindings["start"],

    generateChunk: cwrap(
      "lcw_generate_chunk",
      "number",
      ["number", "number", "number"]
    ) as Bindings["generateChunk"],

    finished: cwrap(
      "lcw_finished",
      "number",
      []
    ) as Bindings["finished"],

    resetKV: cwrap(
      "lcw_reset_kv",
      null,
      []
    ) as Bindings["resetKV"],

    unload: cwrap(
      "lcw_unload",
      null,
      []
    ) as Bindings["unload"],

    lastError: cwrap(
      "lcw_last_error",
      "number",
      []
    ) as Bindings["lastError"],

    benchPromptMs: cwrap(
      "lcw_bench_prompt_ms",
      "number",
      []
    ) as Bindings["benchPromptMs"],

    benchGenerateMs: cwrap(
      "lcw_bench_generate_ms",
      "number",
      []
    ) as Bindings["benchGenerateMs"],

    benchPromptTokens: cwrap(
      "lcw_bench_prompt_tokens",
      "number",
      []
    ) as Bindings["benchPromptTokens"],

    benchGenerateTokens: cwrap(
      "lcw_bench_generate_tokens",
      "number",
      []
    ) as Bindings["benchGenerateTokens"],

    benchPromptBatch: cwrap(
      "lcw_bench_prompt_batch",
      "number",
      []
    ) as Bindings["benchPromptBatch"],

    benchLoadMs: cwrap(
      "lcw_bench_load_ms",
      "number",
      []
    ) as Bindings["benchLoadMs"]
  };

  try {
    moduleInstance.FS.mkdir(MODEL_DIRECTORY);
  } catch {
    // Directory already exists.
  }

  post({ type: "ready", requestId });
}

async function loadModel(
  message: Extract<WorkerRequest, { type: "load" }>
): Promise<void> {
  const module = requireModule();
  const native = requireBindings();

  if (activeGenerationId !== undefined) {
    throw new Error("Cannot replace the model during generation.");
  }

  native.unload();
  removeModelFile();

  const totalBytesHint =
    sourceSizeHint(message.source) + (message.mmproj ? sourceSizeHint(message.mmproj) : 0);
  if (totalBytesHint > 0) {
    preGrowHeap(module, totalBytesHint);
  }

  const bytesWritten = await writeSourceToMemfs(
    module.FS,
    message.source,
    (loadedBytes, totalBytes) => {
      post({
        type: "progress",
        requestId: message.requestId,
        loadedBytes,
        totalBytes
      });
    }
  );

  const result = native.loadModel(
    MODEL_PATH,
    message.options.contextSize,
    message.options.batchSize,
    message.options.threads
  );

  if (result !== 0) {
    throw new Error(readNativeError());
  }

  if (UNLINK_MODEL_AFTER_LOAD) {
    try {
      module.FS.unlink(MODEL_PATH);
    } catch {
      // Already gone; harmless.
    }
  }

  let mmprojLoaded = false;
  if (message.mmproj) {
    const mmprojBytes = await writeSourceToMemfs(
      module.FS,
      message.mmproj,
      (loadedBytes, totalBytes) => {
        post({
          type: "progress",
          requestId: message.requestId,
          loadedBytes,
          totalBytes
        });
      }
    );

    const mmprojResult = native.loadMmproj(MMPROJ_PATH, message.options.threads);
    if (mmprojResult !== 0) {
      throw new Error(readNativeError());
    }
    mmprojLoaded = true;

    if (UNLINK_MODEL_AFTER_LOAD) {
      try {
        module.FS.unlink(MMPROJ_PATH);
      } catch {
        // Already gone; harmless.
      }
    }
  }

  post({
    type: "loaded",
    requestId: message.requestId,
    info: {
      contextSize: message.options.contextSize,
      batchSize: message.options.batchSize,
      threads: message.options.threads,
      bytesWritten,
      loadMs: native.benchLoadMs(),
      threadsCap: message.options.threads,
      mmprojLoaded
    }
  });
}

async function generate(
  message: Extract<WorkerRequest, { type: "generate" }>
): Promise<void> {
  if (activeGenerationId !== undefined) {
    throw new Error("Only one generation can run at a time.");
  }

  const module = requireModule();
  const native = requireBindings();

  activeGenerationId = message.requestId;
  cancelRequested = false;

  // Multimodal path: exactly one image in the latest user turn. mtmd requires
  // a single media marker in the prompt, so we only support one image per turn
  // for now. The image pixels are copied into the WASM heap and passed to
  // lcw_eval_image, which tokenizes (text + image) and decodes the prefix.
  const images = message.chatMessages
    ? lastUserImages(message.chatMessages)
    : undefined;

  let started: number;
  if (images && images.length > 0) {
    if (native.mmprojLoaded() !== 1) {
      activeGenerationId = undefined;
      throw new Error("This model has no mmproj loaded; images are not supported.");
    }
    const image = images[0] as ChatImage;
    if (images.length > 1) {
      activeGenerationId = undefined;
      throw new Error("Only one image per turn is supported.");
    }

    const promptWithMarker = message.chatMessages
      ? formatChatPromptWithImage(message.chatMessages)
      : `${mtmdDefaultMarker()}\n${message.prompt}`;

    const rgbPointer = module._malloc(image.data.byteLength);
    module.HEAPU8.set(image.data, rgbPointer);

    try {
      started = native.evalImage(
        promptWithMarker,
        rgbPointer,
        image.width,
        image.height,
        message.options.maxTokens,
        message.options.temperature,
        message.options.topK,
        message.options.topP,
        message.options.seed >>> 0
      );
    } finally {
      module._free(rgbPointer);
    }
  } else {
    const generationPrompt = message.chatMessages
      ? formatChatPrompt(message.chatMessages)
      : message.prompt;

    started = native.start(
      generationPrompt,
      message.options.maxTokens,
      message.options.temperature,
      message.options.topK,
      message.options.topP,
      message.options.seed >>> 0
    );
  }

  if (started !== 0) {
    activeGenerationId = undefined;
    throw new Error(readNativeError());
  }

  const pointer = module._malloc(OUTPUT_CAPACITY);
  const decoder = new TextDecoder("utf-8", { fatal: false });
  const heapView = module.HEAPU8;

  try {
    while (!cancelRequested) {
      const result = native.generateChunk(pointer, OUTPUT_CAPACITY, CHUNK_TOKENS);

      if (result < 0) {
        throw new Error(readNativeError());
      }

      if (result > 0) {
        const text = decoder.decode(
          new Uint8Array(heapView.buffer, pointer, result),
          { stream: true }
        );
        if (text.length > 0) {
          post({
            type: "token",
            requestId: message.requestId,
            text
          });
        }
      }

      if (native.finished() !== 0) {
        break;
      }

      // Yield to the worker event loop between chunks so a cancel message can
      // be observed. CHUNK_TOKENS is small enough that this still feels
      // responsive (a 16-token chunk at ~5 tokens/s on the 1.2B model is
      // ~3 seconds per native call, which would make cancellation laggy).
      // The 4-token chunk size keeps latency bounded.
      await delay(0);
    }

    const tail = decoder.decode();
    if (tail.length > 0) {
      post({
        type: "token",
        requestId: message.requestId,
        text: tail
      });
    }

    const bench = readBenchmark(native);
    post({
      type: "done",
      requestId: message.requestId,
      bench
    });
  } finally {
    module._free(pointer);
    activeGenerationId = undefined;
    cancelRequested = false;
  }
}

function readBenchmark(native: Bindings): BenchmarkReport {
  return {
    promptMs: native.benchPromptMs(),
    generateMs: native.benchGenerateMs(),
    promptTokens: native.benchPromptTokens(),
    generateTokens: native.benchGenerateTokens(),
    promptBatchSize: native.benchPromptBatch(),
    loadMs: native.benchLoadMs()
  };
}

function formatChatPrompt(
  messages: NonNullable<Extract<WorkerRequest, { type: "generate" }>["chatMessages"]>
): string {
  const module = requireModule();
  const native = requireBindings();

  // Serialize the full conversation. Each record is "role<US>content" joined by
  // <RS>. The native side turns this into a ChatML prompt with one turn per
  // message and an assistant generation prefix at the end. This path is only
  // reached when no images are present (see formatChatPromptWithImage for the
  // multimodal case), so the content is plain text with no media marker.
  const serialized = messages
    .map(message => `${message.role}\x1f${message.content}`)
    .join("\x1e");

  const pointer = module._malloc(65536);
  try {
    const length = native.formatChat(serialized, pointer, 65536);
    if (length < 0) {
      throw new Error(readNativeError());
    }
    const decoded = new TextDecoder().decode(
      module.HEAPU8.slice(pointer, pointer + length)
    );
    return decoded;
  } finally {
    module._free(pointer);
  }
}

// The mtmd default media marker ("<__media__>") inserted into the prompt at the
// position of each image. Must stay in sync with mtmd_default_marker() in the
// vendored llama.cpp tree.
function mtmdDefaultMarker(): string {
  return "<__media__>";
}

// Returns the images attached to the latest user turn, or undefined when there
// are none. Only one image per turn is supported by lcw_eval_image, so the
// generate path throws if more than one is present.
function lastUserImages(
  messages: NonNullable<Extract<WorkerRequest, { type: "generate" }>["chatMessages"]>
): ChatImage[] | undefined {
  for (let i = messages.length - 1; i >= 0; --i) {
    const message = messages[i];
    if (message && message.role === "user" && message.images && message.images.length > 0) {
      return message.images;
    }
  }
  return undefined;
}

// Build the prompt for a single-image user turn. When an mmproj is loaded we
// use the model's own chat template (lcw_format_chat_mm) so the media marker is
// placed inside a correctly-formatted Gemma 4 turn. Without an mmproj the image
// path cannot run, but we still fall back to the ChatML formatter for safety.
function formatChatPromptWithImage(
  messages: NonNullable<Extract<WorkerRequest, { type: "generate" }>["chatMessages"]>
): string {
  const module = requireModule();
  const native = requireBindings();

  const serialized = messages
    .map(message => {
      if (message.images && message.images.length > 0) {
        const text = message.content.length > 0 ? `\n${message.content}` : "";
        return `${message.role}\x1f${mtmdDefaultMarker()}${text}`;
      }
      return `${message.role}\x1f${message.content}`;
    })
    .join("\x1e");

  const pointer = module._malloc(65536);
  try {
    const formatter = native.mmprojLoaded() === 1
      ? native.formatChatMm
      : native.formatChat;
    const length = formatter(serialized, pointer, 65536);
    if (length < 0) {
      throw new Error(readNativeError());
    }
    return new TextDecoder().decode(module.HEAPU8.slice(pointer, pointer + length));
  } finally {
    module._free(pointer);
  }
}

// Returns a best-effort lower bound on the GGUF byte size before the actual
// download/write begins, so the Emscripten heap can be grown once instead of
// triggering several incremental growth operations while copying the model.
function sourceSizeHint(source: ModelSource): number {
  if ("file" in source) {
    return source.file.size;
  }
  return 0;
}

// Pre-grow the WASM heap by enough to cover the upcoming model copy. Emscripten
// starts at INITIAL_MEMORY (256 MiB) and grows incrementally under memory
// pressure when ALLOW_MEMORY_GROWTH is set; for a known-size model that grown
// path triggers several large memmove()s. Pre-growing bypasses that: one grow,
// one heap bump, one copy.
function preGrowHeap(module: LlamaModule, modelBytes: number): void {
  const mem = module.wasmMemory;
  if (!mem) {
    return;
  }

  const page = 64 * 1024;
  // 64 MiB headroom is enough so the heap is already larger than the model
  // leaving room for runtime allocations; growing further while loading the
  // tensors becomes unlikely.
  const want = modelBytes + 64 * 1024 * 1024;
  const current = mem.buffer.byteLength;
  if (want <= current) {
    return;
  }
  const delta = Math.ceil((want - current) / page);
  try {
    mem.grow(delta);
  } catch {
    // Growing may fail near the 4 GiB ceiling (MAXIMUM_MEMORY in CMake). Keep
    // going; the incremental growth machine is still enabled as a fallback.
  }
}

async function writeSourceToMemfs(
  fs: EmscriptenFileSystem,
  source: ModelSource,
  progress: (loadedBytes: number, totalBytes: number) => void
): Promise<number> {
  let totalBytes = 0;
  let reader: ReadableStreamDefaultReader<Uint8Array>;

  if ("file" in source) {
    totalBytes = source.file.size;
    reader = source.file.stream().getReader();
  } else {
    const response = await fetch(source.url, {
      headers: source.headers
    });

    if (!response.ok) {
      throw new Error(
        `Model download failed with HTTP ${response.status}.`
      );
    }

    totalBytes = Number(response.headers.get("content-length") ?? 0);

    if (!response.body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      totalBytes = bytes.byteLength;
      reader = new Blob([bytes]).stream().getReader();
    } else {
      reader = response.body.getReader();
    }
  }

  const stream = fs.open(MODEL_PATH, "w");
  let position = 0;

  try {
    while (true) {
      const { value, done } = await reader.read();

      if (done) {
        break;
      }

      if (!value || value.byteLength === 0) {
        continue;
      }

      const written = fs.write(
        stream,
        value,
        0,
        value.byteLength,
        position
      );

      if (written !== value.byteLength) {
        throw new Error("A partial write occurred while storing the GGUF.");
      }

      position += written;
      progress(position, totalBytes);
    }
  } finally {
    fs.close(stream);
    reader.releaseLock();
  }

  progress(position, totalBytes || position);
  return position;
}

function removeModelFile(): void {
  if (!moduleInstance) {
    return;
  }

  // Drop both the main model and any mmproj file from MEMFS so a second load
  // of a non-multimodal model does not keep a stale ~3 GiB blob around (and
  // does not pick up the wrong projection on the next load).
  for (const path of [MODEL_PATH, MMPROJ_PATH]) {
    try {
      moduleInstance.FS.unlink(path);
    } catch {
      // No file at this path; harmless.
    }
  }
}

function readNativeError(): string {
  const module = requireModule();
  const native = requireBindings();
  const pointer = native.lastError();

  if (!pointer) {
    return "The native inference engine returned an unknown error.";
  }

  // Find the NUL terminator in a single pass, then decode a view straight out
  // of the WASM heap without boxing each byte into a JS Array<number>.
  let end = pointer;
  while (module.HEAPU8[end] !== 0) {
    end += 1;
  }

  const length = end - pointer;
  if (length <= 0) {
    return "The native inference engine returned an unknown error.";
  }

  return new TextDecoder().decode(
    new Uint8Array(module.HEAPU8.buffer, pointer, length)
  );
}

function requireModule(): LlamaModule {
  if (!moduleInstance) {
    throw new Error("The WASM module has not been initialized.");
  }

  return moduleInstance;
}

function requireBindings(): Bindings {
  if (!bindings) {
    throw new Error("The native bindings have not been initialized.");
  }

  return bindings;
}

function post(message: WorkerResponse): void {
  scope.postMessage(message);
}

function postError(requestId: number, error: unknown): void {
  const message =
    error instanceof Error ? error.message : String(error);

  post({
    type: "error",
    requestId,
    message
  });
}

function delay(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}
