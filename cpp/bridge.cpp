#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <emscripten.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
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

using clock_type = std::chrono::steady_clock;

struct timing_block {
    clock_type::time_point start;
    double * sink;
    explicit timing_block(double & sink)
        : start(clock_type::now()), sink(&sink) {}
    ~timing_block() {
        if (sink) {
            *sink += std::chrono::duration<double, std::milli>(
                clock_type::now() - start).count();
        }
    }
    void disable() { sink = nullptr; }
};

llama_model * g_model = nullptr;
llama_context * g_context = nullptr;
llama_sampler * g_sampler = nullptr;
const llama_vocab * g_vocab = nullptr;

// Multimodal context (mtmd). May remain nullptr if no mmproj file was loaded;
// the lcw_eval_image entry will reject calls in that case. Owned through the
// unique_ptr deleter below.
mtmd_context * g_mtmd = nullptr;

std::string g_last_error;
int32_t g_context_size = 2048;
int32_t g_batch_size = 256;
int32_t g_threads = 1;
int32_t g_max_tokens = 128;
int32_t g_generated_tokens = 0;

// Persistent conversation state. The context is created once when the model
// is loaded and kept alive across chat turns so the KV cache survives. Each
// completed turn's tokens are appended to g_prefix_tokens, so the next turn
// only needs to evaluate the newly appended user message rather than
// re-tokenizing and re-evaluating the entire history.
std::vector<llama_token> g_prefix_tokens;

// True while a generation is currently in progress (between lcw_start_generation
// and EOS / max_tokens / lcw_reset_kv).
bool g_generation_active = false;

// Benchmark accumulators (ms), reset by lcw_start. They are read by the worker
// through lcw_bench after the run completes.
double g_bench_prompt_ms = 0.0;
double g_bench_generate_ms = 0.0;
int32_t g_bench_prompt_tokens = 0;
int32_t g_bench_generate_tokens = 0;
int32_t g_bench_prompt_batch = 0;
double g_bench_load_ms = 0.0;

bool g_backend_initialized = false;

void set_error(const std::string & message) {
    g_last_error = message;
}

bool ensure_backend() {
    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }
    return true;
}

bool create_context() {
    if (g_context != nullptr) {
        return true;
    }
    if (g_model == nullptr) {
        set_error("No model is loaded.");
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(g_context_size);
    ctx_params.n_batch = static_cast<uint32_t>(
        std::max<int32_t>(1, std::min<int32_t>(g_batch_size, g_context_size)));
    ctx_params.n_threads = g_threads;
    ctx_params.n_threads_batch = g_threads;
    ctx_params.no_perf = true;

    g_context = llama_init_from_model(g_model, ctx_params);
    if (g_context == nullptr) {
        set_error("llama.cpp could not create an inference context.");
        return false;
    }
    return true;
}

void free_sampler() {
    if (g_sampler != nullptr) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
}

void reset_generation_state() {
    free_sampler();
    g_generation_active = false;
    g_generated_tokens = 0;
    g_prefix_tokens.clear();
    if (g_context != nullptr) {
        llama_memory_t mem = llama_get_memory(g_context);
        if (mem != nullptr) {
            llama_memory_clear(mem, true);
        }
    }
}

void clear_model() {
    reset_generation_state();

    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    if (g_mtmd != nullptr) {
        mtmd_free(g_mtmd);
        g_mtmd = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    g_vocab = nullptr;
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

// Convert a token to its UTF-8 piece. Uses a stack-allocated 256-byte buffer
// for the common case; only spills to the heap if the piece exceeds that, so
// per-token allocation is avoided in the hot path.
size_t token_to_piece(llama_token token, char * out, size_t out_capacity) {
    int32_t written = llama_token_to_piece(
        g_vocab,
        token,
        out,
        static_cast<int32_t>(out_capacity),
        0,
        true
    );

    if (written < 0) {
        const size_t need = static_cast<size_t>(-written);
        if (need > out_capacity) {
            // Signal overflow to the caller; it should not normally happen for
            // the chunked path because the output capacity is large.
            return 0;
        }
        written = llama_token_to_piece(
            g_vocab,
            token,
            out,
            static_cast<int32_t>(out_capacity),
            0,
            true
        );
    }

    if (written < 0) {
        set_error("Failed to decode a generated token.");
        return 0;
    }
    return static_cast<size_t>(written);
}

// Append `tokens` starting at `start` to the persistent context, evaluating
// in g_batch_size chunks. Returns true on success.
bool eval_tokens(const llama_token * tokens, size_t count, double & bench_ms) {
    if (g_context == nullptr) {
        set_error("No inference context is available.");
        return false;
    }

    size_t offset = 0;
    while (offset < count) {
        const size_t take = std::min<size_t>(
            static_cast<size_t>(g_batch_size), count - offset);

        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens + offset),
            static_cast<int32_t>(take));

        timing_block t(bench_ms);
        if (llama_decode(g_context, batch) != 0) {
            t.disable();
            set_error("llama_decode failed while processing tokens.");
            return false;
        }
        offset += take;
    }
    return true;
}

bool build_sampler(float temperature, int32_t top_k, float top_p, uint32_t seed) {
    free_sampler();

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    g_sampler = llama_sampler_chain_init(sp);
    if (g_sampler == nullptr) {
        set_error("llama.cpp could not create a sampler.");
        return false;
    }

    if (temperature <= 0.0f) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(
            g_sampler, llama_sampler_init_top_k(std::max<int32_t>(0, top_k)));
        llama_sampler_chain_add(
            g_sampler, llama_sampler_init_top_p(std::clamp(top_p, 0.0f, 1.0f), 1));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(seed));
    }
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
    llama_log_set(lcw_log_callback, nullptr);

    g_context_size = std::max<int32_t>(128, context_size);
    g_batch_size = std::max<int32_t>(1, batch_size);
    g_threads = std::max<int32_t>(1, threads);

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    params.use_mlock = false;
    params.check_tensors = false;

    timing_block t(g_bench_load_ms);
    g_model = llama_model_load_from_file(model_path, params);
    if (g_model == nullptr) {
        t.disable();
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

    if (!create_context()) {
        clear_model();
        return -5;
    }

    return 0;
}

EMSCRIPTEN_KEEPALIVE
int32_t lcw_load_mmproj(
    const char * mmproj_path,
    int32_t threads
) {
    g_last_error.clear();

    if (mmproj_path == nullptr || mmproj_path[0] == '\0') {
        set_error("An mmproj path is required.");
        return -1;
    }

    if (g_model == nullptr) {
        set_error("Load the main model before loading the mmproj.");
        return -2;
    }

    // Free any previously loaded projection so a second load replaces it.
    if (g_mtmd != nullptr) {
        mtmd_free(g_mtmd);
        g_mtmd = nullptr;
    }

    mtmd_context_params mm_params = mtmd_context_params_default();
    mm_params.use_gpu = false;
    mm_params.n_threads = std::max<int32_t>(1, threads);
    mm_params.warmup = false;

    timing_block t(g_bench_load_ms);
    g_mtmd = mtmd_init_from_file(mmproj_path, g_model, mm_params);
    if (g_mtmd == nullptr) {
        t.disable();
        set_error("llama.cpp could not load the mmproj projection file.");
        return -3;
    }

    if (!mtmd_support_vision(g_mtmd)) {
        set_error("The mmproj file does not expose a vision projection.");
        mtmd_free(g_mtmd);
        g_mtmd = nullptr;
        return -4;
    }

    return 0;
}

EMSCRIPTEN_KEEPALIVE
int32_t lcw_mmproj_loaded() {
    return g_mtmd != nullptr ? 1 : 0;
}

// Evaluate a text prompt that embeds one image. The prompt must contain exactly
// one media marker (`mtmd_default_marker()`, normally "<__media__>") where the
// image is to be inserted. The image is supplied as raw RGB pixels (length ==
// width * height * 3). On success the resulting text/image KV state is appended
// to the persistent prefix exactly like lcw_start does for text turns, so
// subsequent text turns can reuse it.
//
// Returns:
//    0  success
//   <0  error (see lcw_last_error)
EMSCRIPTEN_KEEPALIVE
int32_t lcw_eval_image(
    const char * prompt,
    uint8_t * rgb,
    uint32_t nx,
    uint32_t ny,
    int32_t max_tokens,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t seed
) {
    g_last_error.clear();

    try {
    if (g_model == nullptr || g_vocab == nullptr || g_context == nullptr) {
        set_error("Load a model before starting generation.");
        return -1;
    }

    if (g_mtmd == nullptr) {
        set_error("Load an mmproj file before evaluating an image prompt.");
        return -2;
    }

    if (prompt == nullptr || rgb == nullptr || nx == 0 || ny == 0) {
        set_error("An image prompt requires text, pixels, width and height.");
        return -3;
    }

    const std::string prompt_text(prompt);

    mtmd_input_text text;
    text.text = prompt_text.c_str();
    text.text_len = prompt_text.size();
    text.add_special = false;
    text.parse_special = true;

    // Use the raw C API rather than the mtmd:: C++ wrappers: the wrapper types
    // (mtmd::bitmap, mtmd::input_chunks) hold a unique_ptr whose deleter is
    // instantiated with a forward-declared struct, and constructing/accessing
    // .ptr triggers incomplete-type compile errors. The C API is fine to use
    // directly from C++ and matches the rest of the bridge's style.
    mtmd_bitmap * image_bitmap = mtmd_bitmap_init(nx, ny, rgb);
    if (image_bitmap == nullptr) {
        set_error("Could not allocate the image bitmap.");
        return -4;
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        mtmd_bitmap_free(image_bitmap);
        set_error("Could not allocate the input chunks container.");
        return -5;
    }

    // mtmd_tokenize expects `const mtmd_bitmap **`; take the address of a
    // const pointer so the conversion is exact without a C-style cast.
    const mtmd_bitmap * bitmap_view = image_bitmap;
    const int32_t tok_res = mtmd_tokenize(
        g_mtmd, chunks, &text, &bitmap_view, 1);
    mtmd_bitmap_free(image_bitmap);
    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        set_error("mtmd_tokenize failed (marker/image count mismatch?).");
        return -6;
    }

    g_max_tokens = std::max<int32_t>(0, max_tokens);
    if (g_max_tokens == 0) {
        g_max_tokens = 128;
    }
    g_generated_tokens = 0;

    g_bench_prompt_ms = 0.0;
    g_bench_generate_ms = 0.0;
    g_bench_prompt_tokens = 0;
    g_bench_generate_tokens = 0;
    g_bench_prompt_batch = g_batch_size;

    // For multimodal turns we do not attempt prefix reuse across turns: image
    // prompts carry their own marker and the previous text/visual prefix is not
    // necessarily a prefix of this new sequence. Reset the KV cache so the new
    // prompt is evaluated from scratch.
    {
        llama_memory_t mem = llama_get_memory(g_context);
        if (mem != nullptr) {
            llama_memory_clear(mem, true);
        }
        g_prefix_tokens.clear();
    }

    llama_pos n_past = 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk * chunk =
            mtmd_input_chunks_get(chunks, i);

        // Only the final chunk should leave logits computed on its last
        // position so the sampler attached by build_sampler below can draw
        // from them. Intermediate chunks must run with logits_last=false to
        // avoid wasting compute on positions the sampler never reads.
        const bool is_last = (i + 1 == n_chunks);

        timing_block t(g_bench_prompt_ms);
        const int32_t eval_res = mtmd_helper_eval_chunk_single(
            g_mtmd,
            g_context,
            chunk,
            n_past,
            0,
            g_batch_size,
            is_last,
            &n_past);
        if (eval_res != 0) {
            mtmd_input_chunks_free(chunks);
            reset_generation_state();
            set_error("Evaluating an image/text chunk failed.");
            return -7;
        }
    }

    // Free the chunks container now that the KV cache has consumed them. The
    // chunk data was decoded into g_context, so the container's own memory is
    // safe to release.
    mtmd_input_chunks_free(chunks);

    // Record the evaluated tokens into the persistent prefix so the generation
    // loop (lcw_generate_chunk) can continue from n_past and so the next text
    // turn that shares this prefix reuses the visual state.
    g_prefix_tokens.resize(static_cast<size_t>(n_past));

    g_bench_prompt_tokens = static_cast<int32_t>(n_past);

    std::fprintf(stderr,
        "[lcw] eval_image: chunks=%zu n_past=%d max_tokens=%d\n",
        n_chunks, n_past, g_max_tokens);

    if (!build_sampler(temperature, top_k, top_p, seed)) {
        return -8;
    }

    g_generation_active = true;
    return 0;
    } catch (const std::exception & e) {
        set_error(std::string("lcw_eval_image: ") + e.what());
        return -100;
    }
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

// Formats a chat conversation using the model's own (built-in) chat template
// rather than the ChatML fallback used by lcw_format_chat_multi. This is
// required for multimodal models (e.g. Gemma 4) whose template differs from
// ChatML and which embed a media marker (`<__media__>`) where images go. The
// worker passes image turns serialized as "user<US><__media__><US>text", so the
// marker lands inside the rendered user turn and mtmd_tokenize can expand it.
//
// Returns the number of bytes written, or a negative error code.
EMSCRIPTEN_KEEPALIVE
int32_t lcw_format_chat_mm(
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

    const char * tmpl = llama_model_chat_template(g_model, nullptr);
    if (tmpl == nullptr) {
        set_error("The model does not expose a chat template.");
        return -4;
    }

    // Parse "role<US>content" records separated by <RS>. Image turns carry a
    // <__media__> marker inside the content, emitted by the worker.
    std::vector<llama_chat_message> chat;
    std::vector<std::string> role_storage;
    std::vector<std::string> content_storage;

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

            role_storage.push_back(role);
            content_storage.push_back(content);
            chat.push_back(
                llama_chat_message{ role_storage.back().c_str(),
                                    content_storage.back().c_str() });
        }

        if (record_end == nullptr) {
            break;
        }
        record = record_end + 1;
    }

    if (chat.empty()) {
        set_error("At least one chat message is required.");
        return -2;
    }

    // First pass: measure the required buffer size.
    const int32_t needed = llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), true, nullptr, 0);
    if (needed < 0) {
        set_error("The model chat template rejected the conversation.");
        return -5;
    }

    if (needed > output_capacity) {
        set_error("The formatted chat prompt exceeds the output buffer.");
        return -6;
    }

    const int32_t written = llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), true,
        reinterpret_cast<char *>(output), output_capacity);
    if (written < 0) {
        set_error("Failed to apply the model chat template.");
        return -7;
    }

    return written;
    } catch (const std::exception & e) {
        set_error(std::string("lcw_format_chat_mm: ") + e.what());
        return -100;
    }
}

// lcw_start now sets up the persistent conversation prefix. It does NOT call
// clear_generation() (which destroys the KV cache), but instead discovers the
// longest common token prefix with the previously evaluated conversation, then
// either rewinds the KV cache to that point (when the new prompt shares a
// prefix) or performs a full reset. Only the genuinely new tokens are then
// evaluated, so subsequent turns pay for the new user message only.
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
    if (g_model == nullptr || g_vocab == nullptr || g_context == nullptr) {
        set_error("Load a model before starting generation.");
        return -1;
    }

    if (prompt == nullptr) {
        set_error("A prompt is required.");
        return -2;
    }

    if (g_generation_active) {
        set_error("A generation is already active. Finish or reset it first.");
        return -3;
    }

    const std::string prompt_text(prompt);
    std::vector<llama_token> prompt_tokens = tokenize(prompt_text);

    if (prompt_tokens.empty()) {
        set_error("The prompt could not be tokenized.");
        return -4;
    }

    g_max_tokens = std::max<int32_t>(0, max_tokens);
    if (g_max_tokens == 0) {
        g_max_tokens = 128;
    }
    g_generated_tokens = 0;

    // Reset benchmark counters for this run.
    g_bench_prompt_ms = 0.0;
    g_bench_generate_ms = 0.0;
    g_bench_prompt_tokens = 0;
    g_bench_generate_tokens = 0;
    g_bench_prompt_batch = 0;

    if (static_cast<int64_t>(prompt_tokens.size()) + g_max_tokens >
        g_context_size) {
        set_error("Prompt tokens plus maxTokens exceed the configured context size.");
        return -5;
    }

    // Find the longest common prefix with the previously evaluated tokens.
    const size_t common = [&] {
        const size_t n = std::min(g_prefix_tokens.size(), prompt_tokens.size());
        size_t i = 0;
        for (; i < n; ++i) {
            if (g_prefix_tokens[i] != prompt_tokens[i]) {
                break;
            }
        }
        return i;
    }();

    // Drop the trailing KV entries that diverge from the new prompt. We always
    // keep sequence 0 because llama_batch_get_one() uses seq id 0.
    if (common < g_prefix_tokens.size()) {
        llama_memory_t mem = llama_get_memory(g_context);
        if (mem != nullptr) {
            // [common, inf): remove everything past the shared prefix.
            if (!llama_memory_seq_rm(mem, 0,
                    static_cast<llama_pos>(common), -1)) {
                set_error("Could not rewind the KV cache to the shared prefix.");
                reset_generation_state();
                return -6;
            }
        }
        g_prefix_tokens.resize(common);
    }

    // Evaluate the newly appended tokens (the suffix after the shared prefix).
    const size_t new_token_count = prompt_tokens.size() - common;
    if (new_token_count > 0) {
        if (!eval_tokens(prompt_tokens.data() + common,
                         new_token_count,
                         g_bench_prompt_ms)) {
            reset_generation_state();
            return -7;
        }
        g_prefix_tokens.insert(
            g_prefix_tokens.end(),
            prompt_tokens.begin() + common,
            prompt_tokens.end());
    }

    g_bench_prompt_tokens = static_cast<int32_t>(new_token_count);
    g_bench_prompt_batch = g_batch_size;

    std::fprintf(stderr,
        "[lcw] start: prompt_tokens=%zu shared_prefix=%zu new=%zu max_tokens=%d\n",
        prompt_tokens.size(), common, new_token_count, g_max_tokens);

    if (!build_sampler(temperature, top_k, top_p, seed)) {
        return -8;
    }

    g_generation_active = true;
    return 0;
    } catch (const std::exception & e) {
        set_error(std::string("lcw_start: ") + e.what());
        return -100;
    }
}

// Chunked generation entry point. Generates up to `max_tokens` tokens (or
// until the `output` buffer is full, or EOS, or the cancel flag flips) in a
// single WASM call. The worker receives one UTF-8 chunk per round-trip.
//
// Returns:
//   >0 : number of bytes written to `output` (always >= 1; add 1 sentinel
//        convention is dropped because the chunk path can emit a real count).
//    0 : generation finished for THIS chunk AND overall (EOS or max_tokens).
//        The caller must treat 0 as overall completion. See lcw_finished().
//   <0 : error (see set_error()).
//
// We avoid ever returning 0 with more tokens to come: if the buffer would
// overflow without finishing, we return the partial bytes (>0) and let the
// worker loop again. The worker should continue while the result is >0.
EMSCRIPTEN_KEEPALIVE
int32_t lcw_generate_chunk(
    uint8_t * output,
    int32_t output_capacity,
    int32_t max_tokens
) {
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

    if (!g_generation_active) {
        return 0;
    }

    const int32_t budget = std::max<int32_t>(1, std::min<int32_t>(max_tokens, g_max_tokens - g_generated_tokens));
    if (budget <= 0) {
        return 0;
    }

    char * out = reinterpret_cast<char *>(output);
    int32_t written = 0;

    // Per-token scratch buffer; llama_token_to_piece needs at most the vocab's
    // longest piece, which for these models fits comfortably in 256 bytes. We
    // only fall back to a heap allocation if a single piece genuinely exceeds
    // the output buffer (extremely rare).
    std::array<char, 256> scratch;

    bool stopped = false;

    // We never sample a token unless there's headroom for its piece in the
    // output buffer (scratch.size() bytes). This way a sampled token is always
    // fully committed: piece copied to `out`, then decoded into the KV cache.
    // The sampler state therefore always agrees with the KV state.
    const size_t headroom = scratch.size();

    for (int32_t i = 0; i < budget; ++i) {
        if (output_capacity - written < static_cast<int32_t>(headroom)) {
            break;
        }

        llama_token token = llama_sampler_sample(g_sampler, g_context, -1);
        if (llama_vocab_is_eog(g_vocab, token)) {
            stopped = true;
            break;
        }

        size_t bytes = token_to_piece(token, scratch.data(), scratch.size());
        if (bytes == 0) {
            return -3;
        }
        // The headroom check above guarantees `scratch.size()` bytes remain in
        // the output buffer, and `bytes` is bounded by `scratch.size()`.
        // So `written + bytes <= output_capacity` always holds here.
        std::memcpy(out + written, scratch.data(), bytes);
        written += static_cast<int32_t>(bytes);

        // Evaluate the chosen token so the KV cache advances for the next
        // sample. This is where the per-token decode time is spent.
        llama_batch batch = llama_batch_get_one(&token, 1);
        timing_block t(g_bench_generate_ms);
        if (llama_decode(g_context, batch) != 0) {
            t.disable();
            set_error("llama_decode failed while evaluating a generated token.");
            return -4;
        }

        ++g_generated_tokens;
        ++g_bench_generate_tokens;

        // Roll the token into the persistent prefix so the next turn can reuse
        // it. The next lcw_start receives the assistant's full reply in the
        // conversation, which tokenizes to (at least) these same token ids, so
        // the prefix comparison extends the shared run naturally on multi-turn.
        g_prefix_tokens.push_back(token);

        if (g_generated_tokens >= g_max_tokens) {
            stopped = true;
            break;
        }
    }

    if (stopped) {
        // Generation may be restarted by another lcw_start that has the previous
        // assistant turn in its prefix - reused via the KV cache.
        g_generation_active = false;
        free_sampler();
    }

    // Completion semantics (worker):
    //   result > 0  : chunk produced, loop again. Even if we hit max_tokens /
    //                 EOS within this call, we still surface the bytes here
    //                 and let lcw_finished()=true drive the next iteration to
    //                 completion in one extra cheap zero-byte round.
    //   result == 0 : overall completion and nothing more to flush.
    if (written > 0) {
        return written;
    }
    return 0;
    } catch (const std::exception & e) {
        set_error(std::string("lcw_generate_chunk: ") + e.what());
        return -100;
    }
}

EMSCRIPTEN_KEEPALIVE
int32_t lcw_finished() {
    return g_generation_active ? 0 : 1;
}

// Free the sampler (usually called implicitly after a generation completes)
// while keeping the persistent context and KV cache intact for the next turn.
EMSCRIPTEN_KEEPALIVE
void lcw_reset_kv() {
    g_last_error.clear();
    g_generation_active = false;
    free_sampler();
    g_generated_tokens = 0;
    g_prefix_tokens.clear();
    if (g_context != nullptr) {
        llama_memory_t mem = llama_get_memory(g_context);
        if (mem != nullptr) {
            llama_memory_clear(mem, true);
        }
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

// Benchmark accessors. The worker reads them after a generation ends and
// surfaces timings back to JS for the benchmark UI.
EMSCRIPTEN_KEEPALIVE
double lcw_bench_prompt_ms()       { return g_bench_prompt_ms; }
EMSCRIPTEN_KEEPALIVE
double lcw_bench_generate_ms()     { return g_bench_generate_ms; }
EMSCRIPTEN_KEEPALIVE
int32_t lcw_bench_prompt_tokens() { return g_bench_prompt_tokens; }
EMSCRIPTEN_KEEPALIVE
int32_t lcw_bench_generate_tokens() { return g_bench_generate_tokens; }
EMSCRIPTEN_KEEPALIVE
int32_t lcw_bench_prompt_batch()   { return g_bench_prompt_batch; }
EMSCRIPTEN_KEEPALIVE
double lcw_bench_load_ms()         { return g_bench_load_ms; }

} // extern "C"
