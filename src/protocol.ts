import type {
  ChatMessage,
  CompletionOptions,
  LoadOptions,
  ModelInfo,
  ModelSource,
  RuntimeAssets
} from "./types.js";

export interface BenchmarkReport {
  promptMs: number;
  generateMs: number;
  promptTokens: number;
  generateTokens: number;
  promptBatchSize: number;
  loadMs: number;
}

export type WorkerRequest =
  | {
      type: "init";
      requestId: number;
      assets: RuntimeAssets;
    }
  | {
      type: "load";
      requestId: number;
      source: ModelSource;
      options: Required<LoadOptions>;
    }
  | {
      type: "generate";
      requestId: number;
      prompt: string;
      chatMessages?: ChatMessage[];
      options: Required<CompletionOptions>;
    }
  | {
      type: "cancel";
      requestId: number;
      generationId: number;
    }
  | {
      type: "resetKV";
      requestId: number;
    }
  | {
      type: "unload";
      requestId: number;
    };

export type WorkerResponse =
  | {
      type: "ready";
      requestId: number;
    }
  | {
      type: "progress";
      requestId: number;
      loadedBytes: number;
      totalBytes: number;
    }
  | {
      type: "loaded";
      requestId: number;
      info: ModelInfo;
    }
  | {
      type: "token";
      requestId: number;
      text: string;
    }
  | {
      type: "done";
      requestId: number;
      bench?: BenchmarkReport;
    }
  | {
      type: "kvReset";
      requestId: number;
    }
  | {
      type: "unloaded";
      requestId: number;
    }
  | {
      type: "error";
      requestId: number;
      message: string;
    };
