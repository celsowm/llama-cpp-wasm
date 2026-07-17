export type ModelSource =
  | {
      file: File;
    }
  | {
      url: string;
      headers?: Record<string, string>;
    };

/**
 * A decoded still image, ready to be handed to the multimodal (mmproj) path.
 * The pixels are raw RGB (length === width * height * 3), matching the layout
 * expected by mtmd's `mtmd_bitmap_init` (RGBRGB... with no padding).
 */
export interface ChatImage {
  data: Uint8Array;
  width: number;
  height: number;
}

/**
 * Runs on the main thread, returns true only when the surrounding document is
 * cross-origin isolated AND a SharedArrayBuffer is actually reachable. Used to
 * decide whether the multithreaded pthread build can run.
 */
export function supportsThreads(): boolean {
  if (typeof crossOriginIsolated === "undefined" || !crossOriginIsolated) {
    return false;
  }
  return typeof SharedArrayBuffer !== "undefined";
}

/**
 * Default thread count when the multithreaded build can be used. Capped at 4
 * as a balance between prompt processing throughput and memory-bandwidth limits
 * in token generation. The cap can be made conservative without losing the
 * practical gain for the 230M model.
 */
export function recommendedThreads(): number {
  if (typeof navigator === "undefined" || !navigator.hardwareConcurrency) {
    return 1;
  }
  return Math.min(4, Math.max(1, navigator.hardwareConcurrency));
}

/**
 * A pair of single-threaded and multithreaded WASM artifacts plus their
 * worker. The single-threaded build is used when the host cannot run pthreads
 * (no SharedArrayBuffer or no cross-origin isolation), or when the caller opts
 * in to a one-thread run.
 */
export interface ThreadedRuntimeAssets {
  /**
   * Optional single-threaded WASM module URL. Used when the host can't run
   * pthreads, or when `forceSingleThread` is set. If absent, the multithreaded
   * build is used with `threads = 1`.
   */
  moduleUrlSt?: string;
  wasmUrlSt?: string;

  /**
   * Idiomatic default: the multithreaded pthread build that also runs as
   * single-threaded when `threads = 1` is passed. Required.
   */
  moduleUrlMt: string;
  wasmUrlMt: string;

  /**
   * Optional worker entry URL. The built package defaults to ./worker.js.
   */
  workerUrl?: string | URL;

  /**
   * Optional worker factory. Browser bundlers such as Vite can use this to
   * statically discover and bundle the module worker.
   */
  workerFactory?: () => Worker;

  /**
   * When true, ignore pthread detection and always pick the single-threaded
   * build if it's available. Used by the benchmark harness or on hosts with
   * very few cores.
   */
  forceSingleThread?: boolean;
}

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
  loadMs?: number;
  /**
   * The pthread pool size the worker was actually configured with (1 for the
   * single-threaded build).
   */
  threadsCap?: number;
  /**
   * True when an mmproj projection file was loaded alongside the main model,
   * enabling image (and audio, for capable models) input.
   */
  mmprojLoaded?: boolean;
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
  /**
   * Optional inline images for multimodal models. Gemma 4 expects the image
   * before the text in a user turn; the runtime inserts a media marker at the
   * position of the first image and routes the turn through the mmproj path.
   */
  images?: ChatImage[];
}
