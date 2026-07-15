export type ModelSource =
  | {
      file: File;
    }
  | {
      url: string;
      headers?: Record<string, string>;
    };

export interface RuntimeAssets {
  /**
   * URL of the generated Emscripten ES module.
   */
  moduleUrl: string;

  /**
   * URL of the generated .wasm binary.
   */
  wasmUrl: string;

  /**
   * Optional worker entry URL. The built package defaults to ./worker.js.
   */
  workerUrl?: string | URL;

  /**
   * Optional worker factory. Browser bundlers such as Vite can use this to
   * statically discover and bundle the module worker.
   */
  workerFactory?: () => Worker;
}

export interface LoadOptions {
  contextSize?: number;
  batchSize?: number;
  threads?: number;
}

export interface LoadProgress {
  loadedBytes: number;
  totalBytes: number;
  ratio?: number;
}

export interface ModelInfo {
  contextSize: number;
  batchSize: number;
  threads: number;
  bytesWritten: number;
}

export interface CompletionOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  seed?: number;
}


export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}
