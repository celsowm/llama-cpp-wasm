/// <reference lib="webworker" />

import type { WorkerRequest, WorkerResponse } from "./protocol.js";
import type { ModelSource } from "./types.js";

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
}

interface Bindings {
  formatChat(
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
  start(
    prompt: string,
    maxTokens: number,
    temperature: number,
    topK: number,
    topP: number,
    seed: number
  ): number;
  next(outputPointer: number, outputCapacity: number): number;
  unload(): void;
  lastError(): number;
}

const scope = self as DedicatedWorkerGlobalScope;
const MODEL_DIRECTORY = "/models";
const MODEL_PATH = `${MODEL_DIRECTORY}/model.gguf`;
const OUTPUT_CAPACITY = 4096;

let moduleInstance: LlamaModule | undefined;
let bindings: Bindings | undefined;
let activeGenerationId: number | undefined;
let cancelRequested = false;
let bytesWritten = 0;

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

      case "unload":
        if (activeGenerationId !== undefined) {
          throw new Error("Cannot unload the model during generation.");
        }
        requireBindings().unload();
        removeModelFile();
        bytesWritten = 0;
        post({
          type: "unloaded",
          requestId: message.requestId
        });
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

    loadModel: cwrap(
      "lcw_load_model",
      "number",
      ["string", "number", "number", "number"]
    ) as Bindings["loadModel"],

    start: cwrap(
      "lcw_start",
      "number",
      ["string", "number", "number", "number", "number", "number"]
    ) as Bindings["start"],

    next: cwrap(
      "lcw_next",
      "number",
      ["number", "number"]
    ) as Bindings["next"],

    unload: cwrap(
      "lcw_unload",
      null,
      []
    ) as Bindings["unload"],

    lastError: cwrap(
      "lcw_last_error",
      "number",
      []
    ) as Bindings["lastError"]
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

  bytesWritten = await writeSourceToMemfs(
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

  post({
    type: "loaded",
    requestId: message.requestId,
    info: {
      contextSize: message.options.contextSize,
      batchSize: message.options.batchSize,
      threads: message.options.threads,
      bytesWritten
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

  const generationPrompt = message.chatMessages
    ? formatChatPrompt(message.chatMessages)
    : message.prompt;

  const started = native.start(
    generationPrompt,
    message.options.maxTokens,
    message.options.temperature,
    message.options.topK,
    message.options.topP,
    message.options.seed >>> 0
  );

  if (started !== 0) {
    activeGenerationId = undefined;
    throw new Error(readNativeError());
  }

  const pointer = module._malloc(OUTPUT_CAPACITY);
  const decoder = new TextDecoder("utf-8", { fatal: false });

  try {
    let tokenCounter = 0;

    while (!cancelRequested) {
      const result = native.next(pointer, OUTPUT_CAPACITY);

      if (result < 0) {
        throw new Error(readNativeError());
      }

      if (result === 0) {
        break;
      }

      const length = result - 1;
      if (length > 0) {
        const bytes = module.HEAPU8.slice(pointer, pointer + length);
        const text = decoder.decode(bytes, { stream: true });

        if (text.length > 0) {
          post({
            type: "token",
            requestId: message.requestId,
            text
          });
        }
      }

      tokenCounter += 1;

      // Yield to the worker event loop so a cancel message can be observed.
      if (tokenCounter % 4 === 0) {
        await delay(0);
      }
    }

    const tail = decoder.decode();
    if (tail.length > 0) {
      post({
        type: "token",
        requestId: message.requestId,
        text: tail
      });
    }

    post({
      type: "done",
      requestId: message.requestId
    });
  } finally {
    module._free(pointer);
    activeGenerationId = undefined;
    cancelRequested = false;
  }
}


function formatChatPrompt(
  messages: NonNullable<Extract<WorkerRequest, { type: "generate" }>["chatMessages"]>
): string {
  const module = requireModule();
  const native = requireBindings();

  // Serialize the full conversation. Each record is "role<US>content" joined by
  // <RS>. The native side turns this into a ChatML prompt with one turn per
  // message and an assistant generation prefix at the end.
  const serialized = messages
    .map(message => `${message.role}\x1f${message.content}`)
    .join("\x1e");

  console.warn("[worker] formatChatPrompt: messages=" + messages.length + " serialized_len=" + serialized.length);

  const pointer = module._malloc(65536);
  try {
    const length = native.formatChat(serialized, pointer, 65536);

    console.warn("[worker] formatChatPrompt: native returned length=" + length);

    if (length < 0) {
      throw new Error(readNativeError());
    }

    const decoded = new TextDecoder().decode(
      module.HEAPU8.slice(pointer, pointer + length)
    );
    console.warn("[worker] formatChatPrompt: prompt=" + JSON.stringify(decoded.slice(0, 200)));
    return decoded;
  } finally {
    module._free(pointer);
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

  try {
    moduleInstance.FS.unlink(MODEL_PATH);
  } catch {
    // No model file exists.
  }
}

function readNativeError(): string {
  const module = requireModule();
  const native = requireBindings();
  const pointer = native.lastError();

  if (!pointer) {
    return "The native inference engine returned an unknown error.";
  }

  const bytes: number[] = [];
  for (let index = pointer; module.HEAPU8[index] !== 0; index += 1) {
    bytes.push(module.HEAPU8[index] ?? 0);
  }

  return new TextDecoder().decode(new Uint8Array(bytes));
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
