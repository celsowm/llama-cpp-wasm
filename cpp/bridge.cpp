#include "llama.h"

#include <emscripten/emscripten.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

// llama.cpp prints verbose INFO lines while loading a model. Route its logging
// through a callback that keeps only warnings and errors so the playground
// console stays readable.
void lcw_log_callback(enum ggml_log_level level, const char * text, void *) {
    if (text == nullptr || text[0] == '\0') {
        return;
    }
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fprintf(stderr, "%s", text);
    }
}

llama_model * g_model = nullptr;
llama_context * g_context = nullptr;
llama_sampler * g_sampler = nullptr;
const llama_vocab * g_vocab = nullptr;

std::string g_last_error;
int32_t g_context_size = 2048;
int32_t g_batch_size = 256;
int32_t g_threads = 1;
int32_t g_max_tokens = 128;
int32_t g_generated_tokens = 0;
bool g_backend_initialized = false;

void set_error(const std::string & message) {
    g_last_error = message;
}

void clear_generation() {
    if (g_sampler != nullptr) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }

    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    g_generated_tokens = 0;
}

void clear_model() {
    clear_generation();

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    g_vocab = nullptr;
}

bool ensure_backend() {
    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }

    return true;
}

std::vector<llama_token> tokenize(const std::string & text) {
    const int32_t required = -llama_tokenize(
        g_vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        true,
        true
    );

    if (required <= 0) {
        return {};
    }

    std::vector<llama_token> tokens(static_cast<size_t>(required));
    const int32_t written = llama_tokenize(
        g_vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        true
    );

    if (written < 0) {
        return {};
    }

    tokens.resize(static_cast<size_t>(written));
    return tokens;
}

bool token_to_bytes(llama_token token, std::string & output) {
    std::vector<char> buffer(256);

    int32_t written = llama_token_to_piece(
        g_vocab,
        token,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        0,
        true
    );

    if (written < 0) {
        buffer.resize(static_cast<size_t>(-written));
        written = llama_token_to_piece(
            g_vocab,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            true
        );
    }

    if (written < 0) {
        set_error("Failed to decode a generated token.");
        return false;
    }

    output.assign(buffer.data(), static_cast<size_t>(written));
    return true;
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int32_t lcw_load_model(
    const char * model_path,
    int32_t context_size,
    int32_t batch_size,
    int32_t threads
) {
    g_last_error.clear();

    if (model_path == nullptr || model_path[0] == '\0') {
        set_error("A model path is required.");
        return -1;
    }

    clear_model();
    ensure_backend();

    // Silence llama.cpp's verbose load logging (keep warnings/errors only).
    llama_log_set(lcw_log_callback, nullptr);

    g_context_size = std::max<int32_t>(128, context_size);
    g_batch_size = std::max<int32_t>(1, batch_size);
    g_threads = std::max<int32_t>(1, threads);

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    params.use_mlock = false;
    params.check_tensors = false;

    g_model = llama_model_load_from_file(model_path, params);
    if (g_model == nullptr) {
        set_error("llama.cpp could not load the GGUF model.");
        return -2;
    }

    if (llama_model_has_encoder(g_model)) {
        set_error("Version 0.0.1 supports decoder-only text models.");
        clear_model();
        return -3;
    }

    g_vocab = llama_model_get_vocab(g_model);
    if (g_vocab == nullptr) {
        set_error("The model does not expose a usable vocabulary.");
        clear_model();
        return -4;
    }

    return 0;
}


EMSCRIPTEN_KEEPALIVE
int32_t lcw_format_chat_multi(
    const char * messages_serialized,
    uint8_t * output,
    int32_t output_capacity
) {
    g_last_error.clear();

    try {
    if (g_model == nullptr) {
        set_error("Load a model before formatting chat messages.");
        return -1;
    }

    if (messages_serialized == nullptr || messages_serialized[0] == '\0') {
        set_error("At least one chat message is required.");
        return -2;
    }

    if (output == nullptr || output_capacity <= 0) {
        set_error("The chat output buffer is invalid.");
        return -3;
    }

    // The worker serializes the full conversation as records separated by the
    // Record Separator (0x1E); each record is "role<US>content" where <US> is the
    // Unit Separator (0x1F). Content may contain newlines.
    //
    // We build the ChatML prompt directly (`<|im_start|>`/`<|im_end|>` turns)
    // instead of using llama.cpp's heuristic template detector. That detector
    // mis-classifies many modern instruction models and can throw on templates it
    // does not recognize (e.g. `std::map::at` from `llm_chat_template_from_str`).
    // The model's BOS is added automatically during tokenization because
    // `add_bos_token` is true, so we must not emit it here.
    std::string prompt;
    int32_t record_count = 0;

    const char * record = messages_serialized;
    while (true) {
        const char * record_end = std::strchr(record, '\x1e');
        const size_t record_len = record_end != nullptr
            ? static_cast<size_t>(record_end - record)
            : std::strlen(record);

        if (record_len > 0) {
            const std::string record_str(record, record_len);
            const size_t sep = record_str.find('\x1f');
            const std::string role = sep == std::string::npos
                ? std::string("user")
                : record_str.substr(0, sep);
            const std::string content = sep == std::string::npos
                ? record_str
                : record_str.substr(sep + 1);

            std::fprintf(stderr, "[lcw] format_chat_multi: turn #%d role='%s' content_len=%zu\n",
                record_count, role.c_str(), content.size());

            if (role == "system") {
                prompt += "<|im_start|>system\n";
                prompt += content;
                prompt += "<|im_end|>\n";
            } else if (role == "assistant") {
                prompt += "<|im_start|>assistant\n";
                prompt += content;
                prompt += "<|im_end|>\n";
            } else {
                prompt += "<|im_start|>user\n";
                prompt += content;
                prompt += "<|im_end|>\n";
            }
            record_count += 1;
        }

        if (record_end == nullptr) {
            break;
        }
        record = record_end + 1;
    }

    // Generation prefix: the model continues generating from here. The LFM2
    // template emits "<|im_start|>assistant\n" (with a trailing newline), so the
    // model's first generated token is the start of the reply, not BOS.
    prompt += "<|im_start|>assistant\n";

    std::fprintf(stderr, "[lcw] format_chat_multi: built prompt %zu bytes, %d turn(s)\n",
        prompt.size(), record_count);
    {
        const size_t preview = std::min<size_t>(prompt.size(), 256);
        std::string snippet(preview, '\0');
        for (size_t i = 0; i < preview; i++) {
            const char c = prompt[i];
            snippet[i] = (c == '\n') ? '\\' : c;
        }
        std::fprintf(stderr, "[lcw] prompt[0..%zu]=\"%s\"\n", preview, snippet.c_str());
    }

    if (static_cast<int32_t>(prompt.size()) > output_capacity) {
        set_error("The formatted chat prompt exceeds the output buffer.");
        return -6;
    }

    if (!prompt.empty()) {
        std::memcpy(output, prompt.data(), prompt.size());
    }

    return static_cast<int32_t>(prompt.size());
    } catch (const std::exception & e) {
        set_error(std::string("lcw_format_chat_multi: ") + e.what());
        return -100;
    }
}

EMSCRIPTEN_KEEPALIVE
int32_t lcw_start(
    const char * prompt,
    int32_t max_tokens,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t seed
) {
    g_last_error.clear();

    try {
    if (g_model == nullptr || g_vocab == nullptr) {
        set_error("Load a model before starting generation.");
        return -1;
    }

    if (prompt == nullptr) {
        set_error("A prompt is required.");
        return -2;
    }

    clear_generation();

    const std::string prompt_text(prompt);
    std::vector<llama_token> prompt_tokens = tokenize(prompt_text);

    std::fprintf(stderr, "[lcw] start: prompt_text_len=%zu prompt_tokens=%zu max_tokens=%d temp=%.3f seed=%u\n",
        prompt_text.size(), prompt_tokens.size(), std::max<int32_t>(0, max_tokens), temperature, seed);

    if (prompt_tokens.empty()) {
        set_error("The prompt could not be tokenized.");
        return -3;
    }

    g_max_tokens = std::max<int32_t>(0, max_tokens);
    g_generated_tokens = 0;

    if (static_cast<int64_t>(prompt_tokens.size()) + g_max_tokens > g_context_size) {
        set_error("Prompt tokens plus maxTokens exceed the configured context size.");
        return -4;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(g_context_size);
    ctx_params.n_batch = static_cast<uint32_t>(
        std::max<int32_t>(
            1,
            std::min<int32_t>(
                g_batch_size,
                static_cast<int32_t>(prompt_tokens.size())
            )
        )
    );
    ctx_params.n_threads = g_threads;
    ctx_params.n_threads_batch = g_threads;
    ctx_params.no_perf = true;

    g_context = llama_init_from_model(g_model, ctx_params);
    if (g_context == nullptr) {
        set_error("llama.cpp could not create an inference context.");
        return -5;
    }

    llama_sampler_chain_params sampler_params =
        llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    g_sampler = llama_sampler_chain_init(sampler_params);

    if (g_sampler == nullptr) {
        set_error("llama.cpp could not create a sampler.");
        clear_generation();
        return -6;
    }

    if (temperature <= 0.0f) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(
            g_sampler,
            llama_sampler_init_top_k(std::max<int32_t>(0, top_k))
        );
        llama_sampler_chain_add(
            g_sampler,
            llama_sampler_init_top_p(
                std::clamp(top_p, 0.0f, 1.0f),
                1
            )
        );
        llama_sampler_chain_add(
            g_sampler,
            llama_sampler_init_temp(temperature)
        );
        llama_sampler_chain_add(
            g_sampler,
            llama_sampler_init_dist(seed)
        );
    }

    size_t offset = 0;
    while (offset < prompt_tokens.size()) {
        const size_t count = std::min<size_t>(
            static_cast<size_t>(g_batch_size),
            prompt_tokens.size() - offset
        );

        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data() + offset,
            static_cast<int32_t>(count)
        );

        if (llama_decode(g_context, batch) != 0) {
            set_error("llama_decode failed while processing the prompt.");
            clear_generation();
            return -7;
        }

        offset += count;
    }

    return 0;
    } catch (const std::exception & e) {
        set_error(std::string("lcw_start: ") + e.what());
        return -100;
    }
}

EMSCRIPTEN_KEEPALIVE
int32_t lcw_next(uint8_t * output, int32_t output_capacity) {
    g_last_error.clear();

    try {
    if (g_context == nullptr || g_sampler == nullptr || g_vocab == nullptr) {
        set_error("No generation is active.");
        return -1;
    }

    if (output == nullptr || output_capacity <= 0) {
        set_error("The output buffer is invalid.");
        return -2;
    }

    if (g_generated_tokens >= g_max_tokens) {
        std::fprintf(stderr, "[lcw] next: stop (max_tokens=%d reached)\n", g_max_tokens);
        return 0;
    }

    llama_token token = llama_sampler_sample(
        g_sampler,
        g_context,
        -1
    );

    if (llama_vocab_is_eog(g_vocab, token)) {
        std::fprintf(stderr, "[lcw] next: EOS after %d tokens\n", g_generated_tokens);
        return 0;
    }

    std::string piece;
    if (!token_to_bytes(token, piece)) {
        return -3;
    }

    if (g_generated_tokens < 5) {
        std::string clean;
        for (char c : piece) {
            clean += (c == '\n') ? '\\' : c;
        }
        std::fprintf(stderr, "[lcw] next: token #%d id=%d piece=\"%s\" (%zu bytes)\n",
            g_generated_tokens, (int) token, clean.c_str(), piece.size());
    }

    if (piece.size() > static_cast<size_t>(output_capacity)) {
        set_error("The generated token piece exceeds the output buffer.");
        return -4;
    }

    if (!piece.empty()) {
        std::memcpy(output, piece.data(), piece.size());
    }

    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(g_context, batch) != 0) {
        set_error("llama_decode failed while evaluating a generated token.");
        return -5;
    }

    ++g_generated_tokens;

    // Add one so that an empty token piece remains distinguishable from EOS.
    return static_cast<int32_t>(piece.size()) + 1;
    } catch (const std::exception & e) {
        set_error(std::string("lcw_next: ") + e.what());
        return -100;
    }
}

EMSCRIPTEN_KEEPALIVE
void lcw_unload() {
    g_last_error.clear();
    clear_model();
}

EMSCRIPTEN_KEEPALIVE
const char * lcw_last_error() {
    return g_last_error.c_str();
}

} // extern "C"
