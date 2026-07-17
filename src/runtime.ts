import { AsyncQueue } from "./async-queue.js";
import type { BenchmarkReport, WorkerRequest, WorkerResponse } from "./protocol.js";
import type {
  ChatMessage,
  CompletionOptions,
  LoadOptions,
  LoadProgress,
  ModelInfo,
  ModelSource,
  RuntimeAssets,
  ThreadedRuntimeAssets
} from "./types.js";
import { recommendedThreads, supportsThreads } from "./types.js";

interface PendingRequest<T> {
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
  onProgress?: (progress: LoadProgress) => void;
}

const DEFAULT_LOAD_OPTIONS: Required<LoadOptions> = {
  contextSize: 2048,
  batchSize: 256,
  threads: 1
};

const DEFAULT_COMPLETION_OPTIONS: {
  maxTokens: number;
  temperature: number;
  topK: number;
  topP: number;
} = {
  maxTokens: 128,
  temperature: 0.7,
  topK: 40,
  topP: 0.95
};

/**
 * A 32-bit seed is drawn at random when the caller does not provide one,
 * so the default behaviour produces varied output instead of repeating the
 * same deterministic stream every time.
 */
function randomSeed(): number {
  return (Math.floor(Math.random() * 0xffffffff) >>> 0) || 1;
}

function sanitizeNumber(
  value: number | undefined,
  fallback: number,
  min: number,
  name: string
): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved)) {
    throw new Error(`Option "${name}" must be a finite number.`);
  }
  if (resolved < min) {
    throw new Error(`Option "maxTokens" must be greater than or equal to ${min}.`);
  }
  return resolved;
}

function sanitizeSeed(value: number): number {
  if (!Number.isFinite(value)) {
    throw new Error(`Option "seed" must be a finite number.`);
  }
  return value >>> 0;
}

export class LlamaCppWasm {
  private readonly worker: Worker;
  private readonly pending = new Map<number, PendingRequest<unknown>>();
  private readonly streams = new Map<
    number,
    AsyncQueue<{ text: string; bench?: unknown }>
  >();
  private nextRequestId = 1;
  private activeGenerationId?: number;
  private terminated = false;
  private pendingCancelTimer: ReturnType<typeof setTimeout> | undefined;

  /**
   * The most recent benchmark report pushed by the worker via the "done"
   * message. Undefined if the worker hasn't finished a generation yet (or
   * the build doesn't expose the topic counters).
   */
  lastBench: BenchmarkReport | undefined;

  private constructor(worker: Worker) {
    this.worker = worker;

    this.worker.addEventListener("message", event => {
      this.handleResponse(event.data as WorkerResponse);
    });

    this.worker.addEventListener("error", event => {
      const error = new Error(event.message || "The inference worker failed.");
      this.rejectEverything(error);
    });
  }

  static async create(assets: RuntimeAssets): Promise<LlamaCppWasm> {
    return LlamaCppWasm.createInternal(assets, 1);
  }

  /**
   * Creates the worker and picks the single-threaded or multithreaded WASM
   * artifact at runtime, based on whether the page is cross-origin isolated
   * and a SharedArrayBuffer is available. When `assets.forceSingleThread` is
   * true and a single-threaded build is supplied, the ST build is selected
   * regardless of isolation. The chosen number of threads is exposed on the
   * returned engine via {@link LlamaCppWasm.defaultThreads}.
   */
  static async createThreaded(
    assets: ThreadedRuntimeAssets
  ): Promise<LlamaCppWasm> {
    const wantMt =
      !assets.forceSingleThread &&
      assets.moduleUrlMt &&
      supportsThreads();

    const wantStAvailable = assets.moduleUrlSt !== undefined;

    const useMt = wantMt || !wantStAvailable;

    const chosenModuleUrl = useMt ? assets.moduleUrlMt : (assets.moduleUrlSt ?? assets.moduleUrlMt);
    const chosenWasmUrl = useMt ? assets.wasmUrlMt : (assets.wasmUrlSt ?? assets.wasmUrlMt);

    const effectiveThreads = (useMt ? recommendedThreads() : 1) || 1;
    return LlamaCppWasm.createInternal(
      {
        moduleUrl: chosenModuleUrl,
        wasmUrl: chosenWasmUrl,
        workerUrl: assets.workerUrl,
        workerFactory: assets.workerFactory
      },
      effectiveThreads
    );
  }

  private static async createInternal(
    assets: RuntimeAssets,
    defaultThreads: number
  ): Promise<LlamaCppWasm> {
    const worker =
      assets.workerFactory?.() ??
      new Worker(
        assets.workerUrl ?? new URL("./worker.js", import.meta.url),
        {
          type: "module",
          name: "llama-cpp-wasm"
        }
      );

    const runtime = new LlamaCppWasm(worker);
    runtime.defaultThreads = defaultThreads;
    const requestId = runtime.allocateRequestId();

    await runtime.request<void>({
      type: "init",
      requestId,
      assets: {
        moduleUrl: assets.moduleUrl,
        wasmUrl: assets.wasmUrl
      }
    });

    return runtime;
  }

  /**
   * The thread count that will be applied to `load()` when `threads` is not
   * explicit. 1 for the single-threaded build, otherwise
   * `min(4, navigator.hardwareConcurrency)`.
   */
  defaultThreads = 1;

  async load(
    source: ModelSource,
    options: LoadOptions = {},
    onProgress?: (progress: LoadProgress) => void,
    mmproj?: ModelSource
  ): Promise<ModelInfo> {
    this.assertAlive();

    const contextSize = sanitizeNumber(
      options.contextSize,
      DEFAULT_LOAD_OPTIONS.contextSize,
      1,
      "contextSize"
    );
    const batchSize = sanitizeNumber(
      options.batchSize,
      DEFAULT_LOAD_OPTIONS.batchSize,
      1,
      "batchSize"
    );
    const threads = sanitizeNumber(
      options.threads,
      this.defaultThreads,
      1,
      "threads"
    );

    const requestId = this.allocateRequestId();
    return this.request<ModelInfo>(
      {
        type: "load",
        requestId,
        source,
        options: {
          contextSize,
          batchSize,
          threads
        },
        mmproj
      },
      onProgress
    );
  }

  completion(
    prompt: string,
    options: CompletionOptions = {}
  ): AsyncGenerator<string, void, void> {
    return this.generate({ prompt }, options);
  }

  chat(
    messages: ChatMessage[],
    options: CompletionOptions = {}
  ): AsyncGenerator<string, void, void> {
    if (messages.length === 0) {
      throw new Error("At least one chat message is required.");
    }

    const last = messages[messages.length - 1];
    if (last?.role !== "user") {
      throw new Error("The last chat message must be from the user.");
    }

    return this.generate(
      {
        prompt: "",
        chatMessages: messages
      },
      options
    );
  }

  /**
   * Drops the persistent KV cache and conversation prefix without unloading
   * the model. Call this when switching to an unrelated sidebar conversation
   * whose prompt is not a continuation of the current one; otherwise
   * `lcw_start` will perform an expensive full truncation anyway, but a
   * reset avoids holding the stale cache in memory.
   */
  async resetKV(): Promise<void> {
    this.assertAlive();
    if (this.activeGenerationId !== undefined) {
      throw new Error("Cannot reset the KV cache while a generation is running.");
    }
    const requestId = this.allocateRequestId();
    await this.request<void>({ type: "resetKV", requestId });
  }

  private async *generate(
    input: { prompt: string; chatMessages?: ChatMessage[] },
    options: CompletionOptions
  ): AsyncGenerator<string, void, void> {
    this.assertAlive();

    if (this.activeGenerationId !== undefined) {
      throw new Error("Only one completion can run at a time.");
    }

    const requestId = this.allocateRequestId();
    const queue = new AsyncQueue<{ text: string; bench?: unknown }>();
    this.streams.set(requestId, queue);
    this.activeGenerationId = requestId;

    const merged = { ...DEFAULT_COMPLETION_OPTIONS, ...options };
    const resolvedOptions: Required<CompletionOptions> = {
      maxTokens: sanitizeNumber(
        merged.maxTokens,
        DEFAULT_COMPLETION_OPTIONS.maxTokens,
        1,
        "maxTokens"
      ),
      temperature: sanitizeNumber(
        merged.temperature,
        DEFAULT_COMPLETION_OPTIONS.temperature,
        0,
        "temperature"
      ),
      topK: sanitizeNumber(
        merged.topK,
        DEFAULT_COMPLETION_OPTIONS.topK,
        0,
        "topK"
      ),
      topP: sanitizeNumber(
        merged.topP,
        DEFAULT_COMPLETION_OPTIONS.topP,
        0,
        "topP"
      ),
      seed:
        merged.seed === undefined
          ? randomSeed()
          : sanitizeSeed(merged.seed)
    };

    if (resolvedOptions.topP > 1) {
      throw new Error(`Option "topP" must be less than or equal to 1.`);
    }

    const message: WorkerRequest = {
      type: "generate",
      requestId,
      prompt: input.prompt,
      chatMessages: input.chatMessages,
      options: resolvedOptions
    };

    this.worker.postMessage(message);

    try {
      while (true) {
        const item = await queue.next();
        if (item.done) {
          break;
        }

        if (item.value !== undefined) {
          yield item.value.text;
        }
      }
    } finally {
      const endedNormally = queue.isFinished();
      this.streams.delete(requestId);

      if (this.pendingCancelTimer !== undefined) {
        clearTimeout(this.pendingCancelTimer);
        this.pendingCancelTimer = undefined;
      }

      if (this.activeGenerationId === requestId) {
        this.activeGenerationId = undefined;
      }

      if (!endedNormally && !this.terminated) {
        this.worker.postMessage({
          type: "cancel",
          requestId: this.allocateRequestId(),
          generationId: requestId
        } satisfies WorkerRequest);
      }
    }
  }

  cancel(): void {
    this.assertAlive();

    if (this.activeGenerationId === undefined) {
      return;
    }

    this.worker.postMessage({
      type: "cancel",
      requestId: this.allocateRequestId(),
      generationId: this.activeGenerationId
    } satisfies WorkerRequest);

    // The worker should acknowledge the cancellation by ending the generation
    // (posting "done" or "error"). If it never does -- for example because the
    // WebAssembly worker has hung -- the runtime would otherwise stay wedged on
    // `activeGenerationId` forever. As a safety net, terminate the worker if
    // the generation has not finished within a generous timeout.
    this.pendingCancelTimer = setTimeout(() => {
      this.pendingCancelTimer = undefined;
      this.terminate();
    }, 30000);
  }

  async unload(): Promise<void> {
    this.assertAlive();

    if (this.activeGenerationId !== undefined) {
      throw new Error(
        "Cannot unload while a generation is running. Cancel it first."
      );
    }

    const requestId = this.allocateRequestId();
    await this.request<void>({
      type: "unload",
      requestId
    });
  }

  terminate(): void {
    if (this.terminated) {
      return;
    }

    this.terminated = true;

    if (this.pendingCancelTimer !== undefined) {
      clearTimeout(this.pendingCancelTimer);
      this.pendingCancelTimer = undefined;
    }

    this.worker.terminate();
    this.rejectEverything(new Error("The inference runtime was terminated."));
  }

  private handleResponse(message: WorkerResponse): void {
    if (message.type === "token") {
      this.streams.get(message.requestId)?.push({ text: message.text });
      return;
    }

    if (message.type === "done") {
      if (message.bench) {
        this.lastBench = message.bench as BenchmarkReport;
      }
      this.streams.get(message.requestId)?.end();
      return;
    }

    if (message.type === "error") {
      const error = new Error(message.message);
      const stream = this.streams.get(message.requestId);

      if (stream) {
        stream.fail(error);
        return;
      }

      const pending = this.pending.get(message.requestId);
      if (pending) {
        this.pending.delete(message.requestId);
        pending.reject(error);
      }
      return;
    }

    const pending = this.pending.get(message.requestId);
    if (!pending) {
      return;
    }

    if (message.type === "progress") {
      const ratio =
        message.totalBytes > 0
          ? message.loadedBytes / message.totalBytes
          : undefined;

      pending.onProgress?.({
        loadedBytes: message.loadedBytes,
        totalBytes: message.totalBytes,
        ratio
      });
      return;
    }

    this.pending.delete(message.requestId);

    if (message.type === "loaded") {
      pending.resolve(message.info);
      return;
    }

    if (message.type === "kvReset" || message.type === "unloaded") {
      pending.resolve(undefined);
      return;
    }

    pending.resolve(undefined);
  }

  private request<T>(
    message: WorkerRequest,
    onProgress?: (progress: LoadProgress) => void
  ): Promise<T> {
    this.assertAlive();

    return new Promise<T>((resolve, reject) => {
      this.pending.set(message.requestId, {
        resolve: value => resolve(value as T),
        reject,
        onProgress
      });

      this.worker.postMessage(message);
    });
  }

  private allocateRequestId(): number {
    return this.nextRequestId++;
  }

  private assertAlive(): void {
    if (this.terminated) {
      throw new Error("The inference runtime has been terminated.");
    }
  }

  private rejectEverything(error: Error): void {
    for (const request of this.pending.values()) {
      request.reject(error);
    }
    this.pending.clear();

    for (const stream of this.streams.values()) {
      stream.fail(error);
    }
    this.streams.clear();
    this.activeGenerationId = undefined;
  }
}
