export interface ModelPreset {
  id: string;
  label: string;
  repository: string;
  filename: string;
  url: string;
  recommendedContextSize: number;
  recommendedBatchSize: number;
}

export const LFM25_230M_Q4_K_M: ModelPreset = {
  id: "liquidai-lfm2.5-230m-q4-k-m",
  label: "LiquidAI LFM2.5 230M Q4_K_M",
  repository: "LiquidAI/LFM2.5-230M-GGUF",
  filename: "LFM2.5-230M-Q4_K_M.gguf",
  url: "https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-Q4_K_M.gguf",
  recommendedContextSize: 2048,
  recommendedBatchSize: 256
};

export const LFM25_1_2B_INSTRUCT_Q4_K_M: ModelPreset = {
  id: "liquidai-lfm2.5-1.2b-instruct-q4-k-m",
  label: "LiquidAI LFM2.5 1.2B Instruct Q4_K_M",
  repository: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
  filename: "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
  url: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
  recommendedContextSize: 2048,
  recommendedBatchSize: 256
};
