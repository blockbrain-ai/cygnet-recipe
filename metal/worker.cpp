// Text-only resident readout adapted from GemmaJev's native/gemmajev.cpp.
// https://github.com/dashidhy/GemmaJev
//
// MIT License
// Copyright (c) 2026 Hongyuan Du
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

namespace {
constexpr size_t MAX_REQUEST_BYTES = 2 * 1024 * 1024;
// Canonical Google template, restricted to text system + user, no tools,
// add_generation_prompt=true, enable_thinking=false. Both contents use |trim.
// https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/chat_template.jinja
constexpr const char * TEMPLATE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7";
constexpr const char * TEMPLATE_SHA256 = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4";
constexpr const char * ANSWER_PREFIX = "<|turn>model\n<|channel>thought\n<channel|>";

class InputError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double peak_rss_mb() {
    struct rusage usage {};
    return getrusage(RUSAGE_SELF, &usage) == 0
        ? static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0) : 0;
}

template<class T, void (*Free)(T *)>
using Handle = std::unique_ptr<T, decltype(Free)>;

// Python/Jinja's Unicode str.strip(), independent of the machine's locale.
bool whitespace(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20)
        || cp == 0x85 || cp == 0xa0 || cp == 0x1680
        || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029
        || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

std::string trim_utf8(const std::string & text) {
    if (text.size() > MAX_REQUEST_BYTES) throw InputError("text exceeds the request size limit");
    size_t first = text.size(), last = 0;
    for (size_t i = 0; i < text.size();) {
        const size_t start = i;
        uint32_t cp = static_cast<unsigned char>(text[i++]);
        unsigned continuation = 0;
        uint32_t minimum = 0;
        if (cp >= 0xc2 && cp <= 0xdf) { cp &= 0x1f; continuation = 1; minimum = 0x80; }
        else if (cp >= 0xe0 && cp <= 0xef) { cp &= 0x0f; continuation = 2; minimum = 0x800; }
        else if (cp >= 0xf0 && cp <= 0xf4) { cp &= 7; continuation = 3; minimum = 0x10000; }
        else if (cp >= 0x80) throw InputError("text must be valid UTF-8");
        for (unsigned j = 0; j < continuation; ++j) {
            if (i == text.size()) throw InputError("text must be valid UTF-8");
            const auto byte = static_cast<unsigned char>(text[i++]);
            if ((byte & 0xc0) != 0x80) throw InputError("text must be valid UTF-8");
            cp = (cp << 6) | (byte & 0x3f);
        }
        if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
            throw InputError("text must be valid UTF-8");
        }
        if (cp == 0) throw InputError("text must not contain NUL bytes");
        if (!whitespace(cp)) { first = std::min(first, start); last = i; }
    }
    return first == text.size() ? "" : text.substr(first, last - first);
}

std::string render_chat(const std::string & system, const std::string & prompt) {
    return "<bos><|turn>system\n" + trim_utf8(system) + "<turn|>\n<|turn>user\n"
        + trim_utf8(prompt) + "<turn|>\n" + ANSWER_PREFIX;
}

struct Request {
    std::string system, prompt;
    std::vector<std::string> letters;
    size_t top_logprobs = 20;
    bool reset_cache = false, tokenize_only = false;
};

Request validate(const json & value) {
    if (!value.is_object()) throw InputError("request must be a JSON object");
    if (!value.contains("system") || !value["system"].is_string()
            || !value.contains("prompt") || !value["prompt"].is_string()) {
        throw InputError("system and prompt must be strings");
    }
    Request result;
    result.system = value["system"].get<std::string>();
    result.prompt = value["prompt"].get<std::string>();
    trim_utf8(result.system);
    trim_utf8(result.prompt);
    if (!value.contains("letters") || !value["letters"].is_array()
            || value["letters"].empty() || value["letters"].size() > 26) {
        throw InputError("letters must contain 1 to 26 unique uppercase letters");
    }
    std::set<std::string> seen;
    for (const auto & item : value["letters"]) {
        if (!item.is_string()) throw InputError("letters must be strings");
        const auto label = item.get<std::string>();
        if (label.size() != 1 || label[0] < 'A' || label[0] > 'Z' || !seen.insert(label).second) {
            throw InputError("letters must contain 1 to 26 unique uppercase letters");
        }
        result.letters.push_back(label);
    }
    if (value.contains("top_logprobs")) {
        const auto & count = value["top_logprobs"];
        if (!count.is_number_integer() || count < 1 || count > 20) {
            throw InputError("top_logprobs must be an integer from 1 to 20");
        }
        result.top_logprobs = count.get<size_t>();
    }
    if (value.contains("reset_cache")) {
        if (!value["reset_cache"].is_boolean()) throw InputError("reset_cache must be boolean");
        result.reset_cache = value["reset_cache"].get<bool>();
    }
    if (value.contains("tokenize_only")) {
        if (!value["tokenize_only"].is_boolean()) throw InputError("tokenize_only must be boolean");
        result.tokenize_only = value["tokenize_only"].get<bool>();
    }
    return result;
}

struct Options {
    std::string model;
    int context = 4096, threads = 4, gpu_layers = 99;
    bool self_test = false;
};

Options options(int argc, char ** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "cygnet-metal-worker --model FILE [--ctx-size 4096] [--threads 4] [--gpu-layers 99]\n"
                         "cygnet-metal-worker --self-test (no model or GPU required)\n";
            std::exit(0);
        }
        if (key == "--self-test") { result.self_test = true; continue; }
        if (++i == argc) throw InputError("missing value for " + key);
        const std::string value = argv[i];
        if (key == "--model") { result.model = value; continue; }
        if (key != "--ctx-size" && key != "--threads" && key != "--gpu-layers") {
            throw InputError("unknown argument: " + key);
        }
        size_t consumed = 0;
        const int number = std::stoi(value, &consumed);
        if (consumed != value.size()) throw InputError("invalid integer for " + key);
        if (key == "--ctx-size") result.context = number;
        else if (key == "--threads") result.threads = number;
        else result.gpu_layers = number;
    }
    if (result.context < 512 || result.context > 16384 || result.context % 256 != 0
            || result.threads < 1 || result.threads > 64 || result.gpu_layers < 0 || result.gpu_layers > 999) {
        throw InputError("ctx-size must be a multiple of 256 in 512..16384, threads in 1..64, gpu-layers in 0..999");
    }
    if (!result.self_test && result.model.empty()) throw InputError("--model is required");
    return result;
}

void log_stderr(enum ggml_log_level level, const char * text, void *) {
    if (level != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr);
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    // The rendered canonical template already includes <bos>. Parsing special
    // tokens here matches HF apply_chat_template(..., tokenize=True).
    int32_t count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, false, true);
    if (count >= 0 || count == std::numeric_limits<int32_t>::min()) throw std::runtime_error("failed to size tokenization");
    std::vector<llama_token> result(static_cast<size_t>(-count));
    count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), result.data(),
                           static_cast<int32_t>(result.size()), false, true);
    if (count <= 0) throw std::runtime_error("tokenization failed");
    result.resize(static_cast<size_t>(count));
    return result;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> output(128);
    int count = llama_token_to_piece(vocab, token, output.data(), static_cast<int>(output.size()), 0, true);
    if (count < 0) {
        output.resize(static_cast<size_t>(-count));
        count = llama_token_to_piece(vocab, token, output.data(), static_cast<int>(output.size()), 0, true);
    }
    if (count < 0) throw std::runtime_error("token decoding failed");
    return std::string(output.data(), static_cast<size_t>(count));
}

struct Candidate { llama_token id; std::string letter; double logit; };

json candidate_logprobs(std::vector<Candidate> candidates, size_t limit) {
    if (candidates.empty()) throw std::runtime_error("no allowed tokens");
    for (const auto & candidate : candidates) {
        if (!std::isfinite(candidate.logit)) throw std::runtime_error("non-finite allowed-token logit");
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
        return a.logit == b.logit ? a.id < b.id : a.logit > b.logit;
    });
    const double maximum = candidates.front().logit;
    double sum = 0;
    for (const auto & candidate : candidates) sum += std::exp(candidate.logit - maximum);
    const double log_sum = std::log(sum);
    json records = json::array();
    // Normalize over ALL allowed token IDs before truncation, exactly as the
    // constrained-token readout. Keep duplicate surface letters as records.
    for (size_t i = 0; i < std::min(limit, candidates.size()); ++i) {
        const auto & candidate = candidates[i];
        records.push_back({{"token", candidate.letter}, {"token_id", candidate.id},
                           {"logprob", candidate.logit - maximum - log_sum}});
    }
    return records;
}

json provenance() {
    return {{"runtime_revision", CYGNET_LLAMA_REVISION}, {"worker_source_sha256", CYGNET_WORKER_SHA256},
            {"runtime_patches", json::array()}, {"chat_template_revision", TEMPLATE_REVISION},
            {"chat_template_sha256", TEMPLATE_SHA256}, {"thinking", false},
            {"readout", "prefill_only_exact_letter_tokens"}, {"generated_tokens", 0}};
}

class Worker {
public:
    explicit Worker(const Options & opts) : opts_(opts) {
        const auto started = Clock::now();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = opts.gpu_layers;
        model_.reset(llama_model_load_from_file(opts.model.c_str(), mp));
        if (!model_) throw std::runtime_error("failed to load language model");
        char architecture[128] {};
        if (llama_model_meta_val_str(model_.get(), "general.architecture", architecture, sizeof(architecture)) < 0
                || std::string(architecture) != "gemma4") {
            throw std::runtime_error("this worker requires a Gemma 4 model");
        }
        const auto parameters = llama_model_n_params(model_.get());
        if (parameters < 11000000000ULL || parameters > 14000000000ULL) {
            throw std::runtime_error("this chat profile supports Gemma 4 12B IT only");
        }
        vocab_ = llama_model_get_vocab(model_.get());
        for (const auto * delimiter : {"<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>"}) {
            const auto tokens = tokenize(vocab_, delimiter);
            if (tokens.size() != 1 || piece(vocab_, tokens.front()) != delimiter) {
                throw std::runtime_error("model does not contain the canonical Gemma 4 special tokens");
            }
        }
        for (llama_token id = 0; id < llama_vocab_n_tokens(vocab_); ++id) {
            const auto decoded = piece(vocab_, id);
            if (decoded.size() == 1 && decoded[0] >= 'A' && decoded[0] <= 'Z') {
                letter_tokens_[decoded[0] - 'A'].push_back(id);
            }
        }
        for (const auto & ids : letter_tokens_) {
            if (ids.empty()) throw std::runtime_error("model vocabulary cannot express all option letters");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(opts.context);
        cp.n_batch = std::min<uint32_t>(cp.n_ctx, 512);
        cp.n_ubatch = 128;
        cp.n_threads = opts.threads;
        cp.n_threads_batch = opts.threads;
        cp.swa_full = true;
        context_.reset(llama_init_from_model(model_.get(), cp));
        if (!context_) throw std::runtime_error("failed to create language context");
        load_ms_ = milliseconds(started);
    }

    json ready() const {
        auto result = provenance();
        result.update({{"ready", true}, {"model", opts_.model}, {"model_size", "12b"}, {"load_ms", load_ms_},
                       {"context_size", llama_n_ctx(context_.get())}, {"batch_size", llama_n_batch(context_.get())},
                       {"microbatch_size", llama_n_ubatch(context_.get())}, {"threads", opts_.threads},
                       {"gpu_layers_requested", opts_.gpu_layers}, {"vocab_size", llama_vocab_n_tokens(vocab_)},
                       {"process_peak_rss_mb", peak_rss_mb()}, {"kv_cache", true}});
        json aliases = json::object();
        for (size_t i = 0; i < letter_tokens_.size(); ++i) aliases[std::string(1, 'A' + i)] = letter_tokens_[i];
        result["letter_token_ids"] = aliases;
        return result;
    }

    void invalidate() {
        llama_synchronize(context_.get());
        llama_memory_clear(llama_get_memory(context_.get()), true);
        cached_tokens_.clear();
        previous_failed_ = true;
    }

    json score(const json & request) {
        const auto started = Clock::now();
        const auto input = validate(request);
        auto tokens = tokenize(vocab_, render_chat(input.system, input.prompt));
        if (input.tokenize_only) {
            json aliases = json::object();
            for (const auto & letter : input.letters) aliases[letter] = letter_tokens_[letter[0] - 'A'];
            return {{"id", request.value("id", json(nullptr))}, {"ok", true}, {"tokenize_only", true},
                    {"prompt_tokens", tokens.size()}, {"prompt_token_ids", tokens},
                    {"letter_token_ids", aliases}, {"completion_tokens", 0},
                    {"context_size", llama_n_ctx(context_.get())},
                    {"within_context", tokens.size() <= llama_n_ctx(context_.get())},
                    {"kv_cache", {{"mutated", false}}}, {"timings_ms", {{"total", milliseconds(started)}}}};
        }
        if (tokens.size() > llama_n_ctx(context_.get())) {
            throw InputError("prompt exceeds the maximum context length " + std::to_string(llama_n_ctx(context_.get())));
        }
        const double preprocess_ms = milliseconds(started);
        const auto prefill_started = Clock::now();
        const auto cache = prepare_cache(tokens, input.reset_cache);
        decode(tokens, cache.reused);
        const float * logits = llama_get_logits_ith(context_.get(), -1); // synchronizes Metal
        if (!logits) throw std::runtime_error("no answer-position logits produced");
        const double prefill_ms = milliseconds(prefill_started);
        const auto score_started = Clock::now();
        std::vector<Candidate> candidates;
        for (const auto & letter : input.letters) {
            for (const auto id : letter_tokens_[letter[0] - 'A']) candidates.push_back({id, letter, logits[id]});
        }
        const auto top = candidate_logprobs(std::move(candidates), input.top_logprobs);
        json result = {{"id", request.value("id", json(nullptr))}, {"ok", true},
                       {"prompt_tokens", tokens.size()}, {"completion_tokens", 0},
                       {"total_tokens", tokens.size()}, {"top_logprobs", top},
                       {"kv_cache", {{"enabled", true}, {"reused_tokens", cache.reused},
                                     {"evaluated_tokens", tokens.size() - cache.reused},
                                     {"reset_reason", cache.reason.empty() ? json(nullptr) : json(cache.reason)}}},
                       {"timings_ms", {{"preprocess", preprocess_ms}, {"prefill", prefill_ms},
                                       {"score", milliseconds(score_started)}, {"total", milliseconds(started)}}},
                       {"process_peak_rss_mb", peak_rss_mb()}};
        // The selected token is never decoded or appended to cache.
        cached_tokens_ = std::move(tokens);
        return result;
    }

private:
    struct Cache { size_t reused = 0; std::string reason; };

    Cache prepare_cache(const std::vector<llama_token> & tokens, bool requested_reset) {
        const auto memory = llama_get_memory(context_.get());
        const bool failed = previous_failed_;
        previous_failed_ = false;
        auto clear = [&](const char * reason) {
            llama_memory_clear(memory, true);
            cached_tokens_.clear();
            return Cache{0, reason};
        };
        if (requested_reset) return clear("requested");
        if (cached_tokens_.empty()) return clear(failed ? "previous_request_failed" : "empty");
        if (llama_memory_seq_pos_min(memory, 0) != 0
                || llama_memory_seq_pos_max(memory, 0) != static_cast<llama_pos>(cached_tokens_.size() - 1)) {
            return clear("incomplete_prefix");
        }
        size_t common = 0;
        // Even identical requests re-evaluate their final position for logits.
        const size_t limit = std::min(cached_tokens_.size(), tokens.size() - 1);
        while (common < limit && cached_tokens_[common] == tokens[common]) ++common;
        if (common == 0) return clear("no_common_prefix");
        if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(common), -1)) return clear("suffix_remove_failed");
        if (llama_memory_seq_pos_min(memory, 0) != 0
                || llama_memory_seq_pos_max(memory, 0) != static_cast<llama_pos>(common - 1)) {
            return clear("incomplete_prefix_after_trim");
        }
        return {common, ""};
    }

    void decode(const std::vector<llama_token> & tokens, size_t offset) {
        while (offset < tokens.size()) {
            const size_t count = std::min<size_t>(llama_n_batch(context_.get()), tokens.size() - offset);
            auto batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + offset), static_cast<int32_t>(count));
            std::vector<llama_pos> positions(count);
            std::vector<int32_t> sequence_counts(count, 1);
            llama_seq_id sequence_id = 0;
            std::vector<llama_seq_id *> sequences(count, &sequence_id);
            for (size_t i = 0; i < count; ++i) positions[i] = static_cast<llama_pos>(offset + i);
            std::vector<int8_t> output_flags(count, 0);
            output_flags.back() = offset + count == tokens.size();
            batch.pos = positions.data();
            batch.n_seq_id = sequence_counts.data();
            batch.seq_id = sequences.data();
            batch.logits = output_flags.data();
            if (llama_decode(context_.get(), batch) != 0) throw std::runtime_error("text prefill failed");
            offset += count;
        }
    }

    Options opts_;
    Handle<llama_model, llama_model_free> model_{nullptr, llama_model_free};
    Handle<llama_context, llama_free> context_{nullptr, llama_free};
    const llama_vocab * vocab_ = nullptr;
    std::array<std::vector<llama_token>, 26> letter_tokens_;
    std::vector<llama_token> cached_tokens_;
    bool previous_failed_ = false;
    double load_ms_ = 0;
};

// Drain oversized lines without retaining them; the following request is usable.
bool read_line(std::istream & stream, std::string & line, bool & oversized) {
    line.clear();
    oversized = false;
    bool read_any = false;
    for (char ch; stream.get(ch);) {
        read_any = true;
        if (ch == '\n') break;
        if (line.size() < MAX_REQUEST_BYTES) line.push_back(ch);
        else oversized = true;
    }
    return read_any;
}

void emit(const json & value) {
    std::cout << value.dump(-1, ' ', false, json::error_handler_t::replace) << '\n' << std::flush;
}

void self_test() {
    const std::string system = " \tYou choose a letter.\u3000";
    const std::string prompt = "\u00a0Café 中文\nOptions:\nA. yes\nB. no\r\n";
    const auto rendered = render_chat(system, prompt);
    if (rendered != "<bos><|turn>system\nYou choose a letter.<turn|>\n<|turn>user\nCafé 中文\nOptions:\nA. yes\nB. no<turn|>\n<|turn>model\n<|channel>thought\n<channel|>") {
        throw std::runtime_error("canonical chat serialization self-test failed");
    }
    for (const auto & invalid : {std::string("a\0b", 3), std::string("\xc0\xaf", 2), std::string("\xed\xa0\x80", 3)}) {
        bool rejected = false;
        try { trim_utf8(invalid); } catch (const InputError &) { rejected = true; }
        if (!rejected) throw std::runtime_error("UTF-8 validation self-test failed");
    }
    const auto records = candidate_logprobs({{1, "A", 2}, {2, "A", 2}, {3, "B", 2}}, 2);
    if (records.size() != 2 || records[0]["token"] != "A" || records[1]["token"] != "A"
            || std::abs(records[0]["logprob"].get<double>() + std::log(3.0)) > 1e-12) {
        throw std::runtime_error("duplicate-token probability self-test failed");
    }
    auto result = provenance();
    result.update({{"ok", true}, {"self_test", true}, {"context_size", 4096},
                   {"chat_fixture", {{"system", system}, {"prompt", prompt}, {"rendered", rendered}}}});
    emit(result);
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const auto opts = options(argc, argv);
        if (opts.self_test) { self_test(); return 0; }
        llama_log_set(log_stderr, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        {
            Worker worker(opts);
            emit(worker.ready());
            std::string line;
            bool oversized = false;
            while (read_line(std::cin, line, oversized)) {
                json id = nullptr;
                try {
                    if (oversized) throw InputError("request exceeds the 2 MiB size limit");
                    json request;
                    try { request = json::parse(line); }
                    catch (const json::exception &) { throw InputError("invalid JSON or UTF-8 request"); }
                    if (request.is_object() && request.contains("id")) id = request["id"];
                    emit(worker.score(request));
                } catch (const InputError & error) {
                    worker.invalidate();
                    emit({{"id", id}, {"ok", false}, {"error", error.what()}, {"error_type", "input"}});
                } catch (const std::exception & error) {
                    worker.invalidate();
                    emit({{"id", id}, {"ok", false}, {"error", error.what()}, {"error_type", "inference"}});
                }
            }
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        emit({{"ready", false}, {"ok", false}, {"error", error.what()}, {"error_type", "inference"}});
        return 1;
    }
}
