/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: common_datatype.h
 * Date: 24-12-20
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_COMMON_DATATYPE_H
#define MORTRED_MODEL_SERVER_COMMON_DATATYPE_H

#include <string>
#include <vector>

#include "llama_cpp/llama.h"

namespace jinq {
namespace models {
namespace llm {

struct ModelStatus {
    uint32_t n_ctx_size = 0;
    int32_t kv_cache_cell_nums = 0;
    int32_t embed_dims = 0;

    bool has_vision_tower = false;
    int32_t clip_embedding_dims = 0;
    int32_t clip_hidden_size = 0;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

class Dialog {
  public:
    /***
     *
     */
    Dialog() = default;

    /***
     *
     */
    ~Dialog() = default;

    /***
     *
     * @param transformer
     */
    Dialog(const Dialog &transformer) = default;

    /***
     *
     * @param transformer
     * @return
     */
    Dialog &operator=(const Dialog &transformer) = default;

    /***
     *
     * @param msg
     */
    explicit Dialog(const ChatMessage &msg) { messages.push_back(msg); }

    /***
     *
     * @param role
     * @param content
     */
    Dialog(const std::string &role, const std::string &content) {
        ChatMessage msg = {role, content};
        messages.push_back(msg);
    }

    /***
     *
     * @param role
     * @param content
     */
    Dialog(const char* role, const char* content) {
        ChatMessage msg = {role, content};
        messages.push_back(msg);
    }

    /***
     *
     * @param other
     * @return
     */
    Dialog operator+(const Dialog &other) {
        Dialog tmp(*this);
        std::copy(other.messages.begin(), other.messages.end(), std::back_inserter(tmp.messages));
        return tmp;
    }

    /***
     *
     * @param other
     * @return
     */
    Dialog &operator+=(const Dialog &other) {
        std::copy(other.messages.begin(), other.messages.end(), std::back_inserter(messages));
        return *this;
    }

    /***
     *
     * @param index
     * @return
     */
    inline ChatMessage &operator[](size_t index) { return messages[index]; }

    /***
     *
     * @param msg
     */
    inline void push_back(const ChatMessage &msg) { messages.push_back(msg); }

    /***
     *
     */
    inline void clean_cache() { messages.clear(); }

    /***
     *
     * @return
     */
    bool empty() const { return messages.empty(); };

    /***
     *
     * @return
     */
    inline size_t size() const { return messages.size(); }

  public:
    std::vector<ChatMessage> messages;
};

namespace llama {

enum common_sampler_type {
    COMMON_SAMPLER_TYPE_NONE        = 0,
    COMMON_SAMPLER_TYPE_DRY         = 1,
    COMMON_SAMPLER_TYPE_TOP_K       = 2,
    COMMON_SAMPLER_TYPE_TOP_P       = 3,
    COMMON_SAMPLER_TYPE_MIN_P       = 4,
    //COMMON_SAMPLER_TYPE_TFS_Z       = 5,
    COMMON_SAMPLER_TYPE_TYPICAL_P   = 6,
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_XTC         = 8,
    COMMON_SAMPLER_TYPE_INFILL      = 9,
};

struct common_params_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

    int32_t n_prev             = 64;    // number of previous tokens to remember
    int32_t n_probs            = 0;     // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep           = 0;     // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k              = 40;    // <= 0 to use vocab size
    float   top_p              = 0.95f; // 1.0 = disabled
    float   min_p              = 0.05f; // 0.0 = disabled
    float   xtc_probability    = 0.00f; // 0.0 = disabled
    float   xtc_threshold      = 0.10f; // > 0.5 disables XTC
    float   typ_p              = 1.00f; // typical_p, 1.0 = disabled
    float   temp               = 0.80f; // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range     = 0.00f; // 0.0 = disabled
    float   dynatemp_exponent  = 1.00f; // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat     = 1.00f; // 1.0 = disabled
    float   penalty_freq       = 0.00f; // 0.0 = disabled
    float   penalty_present    = 0.00f; // 0.0 = disabled
    float   dry_multiplier     = 0.0f;  // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    float   dry_base           = 1.75f; // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    int32_t dry_allowed_length = 2;     // tokens extending repetitions beyond this receive penalty
    int32_t dry_penalty_last_n = -1;    // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
    int32_t mirostat           = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau       = 5.00f; // target entropy
    float   mirostat_eta       = 0.10f; // learning rate
    bool    penalize_nl        = false; // consider newlines as a repeatable token
    bool    ignore_eos         = false;
    bool    no_perf            = false; // disable performance metrics
    bool    timing_per_token   = false;

    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};     // default sequence breakers for DRY


    std::vector<enum common_sampler_type> samplers = {
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };

    std::string grammar; // optional BNF-like grammar to constrain sampling

    std::vector<llama_logit_bias> logit_bias; // logit biases to apply
};

}

}
}
}

#endif // MORTRED_MODEL_SERVER_COMMON_DATATYPE_H
