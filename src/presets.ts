export interface ModelPreset {
  id: string;
  label: string;
  repository: string;
  filename: string;
  url: string;
  recommendedContextSize: number;
  recommendedBatchSize: number;

  /**
   * Optional GGUF quantization label (e.g. "Q4_K_M") shown by the benchmark
   * harness when tabulating results.
   */
  quantization?: string;

  /**
   * Optional multimodal projection file (mmproj GGUF) required by vision
   * models. When present the runtime loads it alongside the main model so the
   * preset can accept images.
   */
  mmproj?: {
    filename: string;
    url: string;
  };
}

function lfm230mUrl(filename: string): string {
  return `https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/${filename}`;
}

function lfm2_1_2b_instructUrl(filename: string): string {
  return `https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/${filename}`;
}

export const LFM25_230M_Q4_K_M: ModelPreset = {
  id: "liquidai-lfm2.5-230m-q4-k-m",
  label: "LiquidAI LFM2.5 230M Q4_K_M",
  repository: "LiquidAI/LFM2.5-230M-GGUF",
  filename: "LFM2.5-230M-Q4_K_M.gguf",
  url: lfm230mUrl("LFM2.5-230M-Q4_K_M.gguf"),
  recommendedContextSize: 2048,
  recommendedBatchSize: 256,
  quantization: "Q4_K_M"
};

export const LFM25_1_2B_INSTRUCT_Q4_K_M: ModelPreset = {
  id: "liquidai-lfm2.5-1.2b-instruct-q4-k-m",
  label: "LiquidAI LFM2.5 1.2B Instruct Q4_K_M",
  repository: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
  filename: "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
  url: lfm2_1_2b_instructUrl("LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
  recommendedContextSize: 2048,
  recommendedBatchSize: 256,
  quantization: "Q4_K_M"
};

function gemma4E2BUrl(filename: string): string {
  return `https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/${filename}`;
}

// Google Gemma 4 E2B (effective 2B) multimodal instruction model, QAT Q4_0.
// It is text+image (and audio on E2B) and requires the mmproj projection file
// for vision input via the mtmd path.
export const GEMMA4_E2B_IT_Q4_0: ModelPreset = {
  id: "google-gemma-4-e2b-it-q4-0",
  label: "Google Gemma 4 E2B IT Q4_0 (multimodal)",
  repository: "google/gemma-4-E2B-it-qat-q4_0-gguf",
  filename: "gemma-4-E2B_q4_0-it.gguf",
  url: gemma4E2BUrl("gemma-4-E2B_q4_0-it.gguf"),
  recommendedContextSize: 16384,
  recommendedBatchSize: 256,
  quantization: "Q4_0",
  mmproj: {
    filename: "gemma-4-E2B-it-mmproj.gguf",
    url: gemma4E2BUrl("gemma-4-E2B-it-mmproj.gguf")
  }
};

/**
 * The presets below are intended for the quantization benchmark matrix.
 * They all target the small 230M model so the benchmark fits inside a
 * browser memory budget and finishes within seconds. The 1.2B variants use
 * the same quantization options but are added with the same naming for
 * completeness.
 *
 * Quantizations to measure on WASM SIMD (per the perf-optimization notes):
 *
 *   Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q8_0
 *
 * The benchmark harness drives each preset through the full pipeline and
 * records: prompt tokens/s, generated tokens/s, time to first token,
 * model loading time, and peak WASM memory.
 */
function lfm230mPreset(
  id: string,
  filename: string,
  quantization: string
): ModelPreset {
  return {
    id,
    label: `LiquidAI LFM2.5 230M ${quantization}`,
    repository: "LiquidAI/LFM2.5-230M-GGUF",
    filename,
    url: lfm230mUrl(filename),
    recommendedContextSize: 2048,
    recommendedBatchSize: 256,
    quantization
  };
}

export const LFM25_230M_Q4_0: ModelPreset = lfm230mPreset(
  "liquidai-lfm2.5-230m-q4_0",
  "LFM2.5-230M-Q4_0.gguf",
  "Q4_0"
);

export const LFM25_230M_Q4_K_S: ModelPreset = lfm230mPreset(
  "liquidai-lfm2.5-230m-q4_k_s",
  "LFM2.5-230M-Q4_K_S.gguf",
  "Q4_K_S"
);

export const LFM25_230M_Q5_0: ModelPreset = lfm230mPreset(
  "liquidai-lfm2.5-230m-q5_0",
  "LFM2.5-230M-Q5_0.gguf",
  "Q5_0"
);

export const LFM25_230M_Q8_0: ModelPreset = lfm230mPreset(
  "liquidai-lfm2.5-230m-q8_0",
  "LFM2.5-230M-Q8_0.gguf",
  "Q8_0"
);

/**
 * The full set of quantizations to benchmark on the 230M model. Q4_K_M is
 * included so it can be compared head-to-head with the rest, which is the
 * core question of optimization 6 in the perf notes ("Quantizations are not
 * necessarily faster as they get smaller on WASM SIMD").
 */
export const LFM25_230M_QUANTIZATION_PRESETS: readonly ModelPreset[] = [
  LFM25_230M_Q4_0,
  LFM25_230M_Q4_K_S,
  LFM25_230M_Q4_K_M,
  LFM25_230M_Q5_0,
  LFM25_230M_Q8_0
];
