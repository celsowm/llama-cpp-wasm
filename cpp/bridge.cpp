#include "llama.h"

#include <emscripten/emscripten.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

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
int32_t lcw_format_chat(
    const char * system_message,
    const char * user_message,
    uint8_t * output,
    int32_t output_capacity
) {
    g_last_error.clear();

    try {
    if (g_model == nullptr) {
        set_error("Load a model before formatting chat messages.");
        return -1;
    }

    if (user_message == nullptr || user_message[0] == '\0') {
        set_error("A non-empty user message is required.");
        return -2;
    }

    if (output == nullptr || output_capacity <= 0) {
        set_error("The chat output buffer is invalid.");
        return -3;
    }

    char template_buffer[1 << 20];
    const int32_t template_length = llama_model_meta_val_str(
        g_model,
        "tokenizer.chat_template",
        template_buffer,
        sizeof(template_buffer)
    );

    if (template_length < 0) {
        set_error("The GGUF model does not contain a supported chat template.");
        return -4;
    }

    if (template_length > static_cast<int32_t>(sizeof(template_buffer))) {
        set_error("The model chat template is too large to format.");
        return -7;
    }

    const char * chat_template = template_buffer;

    std::vector<llama_chat_message> messages;
    if (system_message != nullptr && system_message[0] != '\0') {
        messages.push_back({"system", system_message});
    }
    messages.push_back({"user", user_message});

    const int32_t required = llama_chat_apply_template(
        chat_template,
        messages.data(),
        messages.size(),
        true,
        reinterpret_cast<char *>(output),
        static_cast<size_t>(output_capacity)
    );

    if (required < 0) {
        set_error("llama.cpp failed to apply the model chat template.");
        return -5;
    }

    if (required > output_capacity) {
        set_error("The formatted chat prompt exceeds the output buffer.");
        return -6;
    }

    return required;
    } catch (const std::exception & e) {
        set_error(std::string("lcw_format_chat: ") + e.what());
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
        return 0;
    }

    llama_token token = llama_sampler_sample(
        g_sampler,
        g_context,
        -1
    );

    if (llama_vocab_is_eog(g_vocab, token)) {
        return 0;
    }

    std::string piece;
    if (!token_to_bytes(token, piece)) {
        return -3;
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
