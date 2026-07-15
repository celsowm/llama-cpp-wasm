import { AsyncQueue } from "./async-queue.js";
import type { WorkerRequest, WorkerResponse } from "./protocol.js";
import type {
  ChatMessage,
  CompletionOptions,
  LoadOptions,
  LoadProgress,
  ModelInfo,
  ModelSource,
  RuntimeAssets
} from "./types.js";

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
    throw new Error(`Option "${name}" must be greater than or equal to ${min}.`);
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
  private readonly streams = new Map<number, AsyncQueue<string>>();
  private nextRequestId = 1;
  private activeGenerationId?: number;
  private terminated = false;

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

  async load(
    source: ModelSource,
    options: LoadOptions = {},
    onProgress?: (progress: LoadProgress) => void
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
      DEFAULT_LOAD_OPTIONS.threads,
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
        }
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
      throw new Error("Version 0.0.2 requires the last chat message to be from the user.");
    }

    return this.generate(
      {
        prompt: "",
        chatMessages: messages
      },
      options
    );
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
    const queue = new AsyncQueue<string>();
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
          yield item.value;
        }
      }
    } finally {
      const endedNormally = queue.isFinished();
      this.streams.delete(requestId);

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
    this.worker.terminate();
    this.rejectEverything(new Error("The inference runtime was terminated."));
  }

  private handleResponse(message: WorkerResponse): void {
    if (message.type === "token") {
      this.streams.get(message.requestId)?.push(message.text);
      return;
    }

    if (message.type === "done") {
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
