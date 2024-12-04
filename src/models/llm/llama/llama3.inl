/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: Llama3.inl
 * Date: 24-11-22
 ************************************************/

#include "llama3.h"

#include "glog/logging.h"
#include "llama_cpp/llama.h"

#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {
namespace llm {

using jinq::common::CvUtils;
using jinq::common::Timestamp;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::io_define::llm::text::tokens;
using jinq::models::io_define::llm::text::token_id;

namespace llama {

namespace llama_impl {

using internal_input = std::string;
using internal_output = std::string;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, char*>::value, internal_input>::type transform_input(const INPUT& in) {
    internal_input result = std::string(in);
    return result;
}

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::string>::value, internal_input>::type transform_input(const INPUT& in) {
    return in;
}

/***
* transform different type of internal output into external output
* @tparam EXTERNAL_OUTPUT
* @tparam dummy
* @param in
* @return
 */
template <typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::string>::value, std::string>::type
transform_output(const llama_impl::internal_output& internal_out) {
    return internal_out;
}

}

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class Llama3<INPUT, OUTPUT>::Impl {
public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        llama_sampler_free(_m_sampler);
        llama_free(_m_ctx);
        llama_free_model(_m_model);
    }

    /***
    *
    * @param transformer
     */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const decltype(toml::parse("")) &config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @param prompt
     * @param prompt_tokens
     * @return
     */
    StatusCode tokenize_prompt(const std::string& prompt, std::vector<llama_token>& prompt_tokens);

    /***
     *
     * @return
     */
    ModelStatus get_model_stat() const;

    /***
     *
     */
    void clear_kv_cache_cell() const;

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

private:
    // model file path
    std::string _m_model_file_path;

    // llama model params
    llama_model_params _m_model_params = llama_model_default_params();
    // llama model
    llama_model* _m_model = nullptr;
    // llama context params
    llama_context_params _m_ctx_params = llama_context_default_params();
    // llama model context
    llama_context* _m_ctx = nullptr;
    // llama sampler params
    llama_sampler_chain_params _m_sampler_params = llama_sampler_chain_default_params();
    // llama sampler
    llama_sampler* _m_sampler = nullptr;

    // init flag
    bool _m_successfully_initialized = false;

private:
    /***
     *
     * @param prompt_tokens
     * @param generate_out
     * @return
     */
    StatusCode llama_generate(std::vector<llama_token>& prompt_tokens, std::string& generate_out);
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Llama3<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    if (!config.contains("LLAMA3")) {
        LOG(ERROR) << "Config file does not contain LLAMA3 section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    toml::value model_cfg = config.at("LLAMA3");

    // model_file_path
    if (!model_cfg.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = model_cfg.at("model_file_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "Llama3 model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init model
    auto n_gpu_layers = static_cast<int32_t >(model_cfg.at("n_gpu_layers").as_integer());
    auto main_gpu_device_id = static_cast<int32_t >(model_cfg.at("main_gpu_device").as_integer());
    _m_model_params.n_gpu_layers = n_gpu_layers;
    _m_model_params.main_gpu = main_gpu_device_id;
    _m_model = llama_load_model_from_file(_m_model_file_path.c_str(), _m_model_params);
    if (_m_model == nullptr) {
        LOG(ERROR) << "load llama3 model from: " << _m_model_file_path << " failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init sampler
    _m_sampler = llama_sampler_chain_init(_m_sampler_params);
    if (_m_sampler == nullptr) {
        LOG(ERROR) << "failed to create the llama sampler";
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto temp = static_cast<float>(model_cfg.at("sampler_temp").as_floating());
    auto init_min_p = 0.05f;
    auto min_keep = 1;
    llama_sampler_chain_add(_m_sampler, llama_sampler_init_min_p(init_min_p, min_keep));
    llama_sampler_chain_add(_m_sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(_m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // init ctx params
    if (!config.contains("CONTEXT")) {
        LOG(ERROR) << "Config file does not contain CONTEXT section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    toml::value ctx_cfg = config.at("CONTEXT");
    auto ctx_size = static_cast<int32_t >(ctx_cfg.at("context_size").as_integer());
    _m_ctx_params.n_ctx = ctx_size;
    _m_ctx_params.n_batch = ctx_size;
    _m_ctx = llama_new_context_with_model(_m_model, _m_ctx_params);
    if (_m_ctx == nullptr) {
        LOG(ERROR) << "failed to create the llama_context";
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_successfully_initialized = true;
    return StatusCode::OK;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Llama3<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    if constexpr(std::is_same<INPUT, std::string>::value) {
        // transform input
        llama_impl::internal_input prompt = llama_impl::transform_input(in);

        // tokenize input prompt
        std::vector<llama_token> prompt_tokens;
        auto status = tokenize_prompt(prompt, prompt_tokens);
        if (status != StatusCode::OK) {
            return status;
        }

        // run llama3 generate
        std::string generate_out;
        status = llama_generate(prompt_tokens, generate_out);

        // transform output
        out = llama_impl::transform_output<OUTPUT>(generate_out);

        return status;
    } else if constexpr(std::is_same<INPUT, std::vector<llama_token>&>::value) {
        // run llama3 generate
        std::string generate_out;
        auto status = llama_generate(in, generate_out);

        // transform output
        out = llama_impl::transform_output<OUTPUT>(generate_out);

        return status;
    } else {
        LOG(ERROR) << "wrong input data type";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param prompt
 * @param prompt_tokens
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Llama3<INPUT, OUTPUT>::Impl::tokenize_prompt(const std::string& prompt, std::vector<llama_token>& prompt_tokens) {
    if (prompt.empty()) {
        LOG(WARNING) << "input prompt is empty";
        return StatusCode::TOKENIZE_FAILED;
    }

    auto n_prompt_tokens = llama_tokenize(_m_model, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    n_prompt_tokens *= -1;
    prompt_tokens.resize(n_prompt_tokens);
    auto prompt_size = static_cast<int32_t >(prompt.size());
    auto token_data = prompt_tokens.data();
    auto token_size = static_cast<int32_t>(prompt_tokens.size());
    auto add_special = llama_get_kv_cache_used_cells(_m_ctx) == 0;
    auto token_nums = llama_tokenize(_m_model, prompt.c_str(), prompt_size, token_data, token_size, add_special, true);
    if (token_nums < 0) {
        LOG(ERROR) << "failed to tokenize the prompt: " << prompt;
        return StatusCode::TOKENIZE_FAILED;
    }

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
ModelStatus Llama3<INPUT, OUTPUT>::Impl::get_model_stat() const {
    ModelStatus stat{};
    stat.n_ctx_size = llama_n_ctx(_m_ctx);
    stat.kv_cache_cell_nums = llama_get_kv_cache_used_cells(_m_ctx);
    return stat;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
void Llama3<INPUT, OUTPUT>::Impl::clear_kv_cache_cell() const {
    llama_kv_cache_clear(_m_ctx);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param prompt_tokens
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Llama3<INPUT, OUTPUT>::Impl::llama_generate(std::vector<llama_token>& prompt_tokens, std::string& generate_out) {
    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    llama_token new_token_id;
    while (true) {
        // check if we have enough space in the context to evaluate this batch
        int n_ctx = llama_n_ctx(_m_ctx);
        int n_ctx_used = llama_get_kv_cache_used_cells(_m_ctx);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            LOG(ERROR) << "context size exceeded";
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }

        auto status = llama_decode(_m_ctx, batch);
        if (status == 1) {
            LOG(WARNING) << "llama generate failed. could not find a KV slot for the batch "
                          "(try reducing the size of the batch or increase the context)";
        } else if (status < 0) {
            LOG(ERROR) << "llama decode failed code: " << status;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }

        // sample the next token
        new_token_id = llama_sampler_sample(_m_sampler, _m_ctx, -1);

        // is it an end of generation?
        if (llama_token_is_eog(_m_model, new_token_id)) {
            break;
        }

        // convert the token to a string, print it and add it to the response
        char buf[256];
        int n = llama_token_to_piece(_m_model, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            LOG(ERROR) << "failed to convert token to piece";
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        std::string piece(buf, n);
        generate_out += piece;

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
Llama3<INPUT, OUTPUT>::Llama3() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
Llama3<INPUT, OUTPUT>::~Llama3() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Llama3<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
bool Llama3<INPUT, OUTPUT>::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param input
 * @param output
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Llama3<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param input
 * @param output
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Llama3<INPUT, OUTPUT>::tokenize_prompt(const std::string& prompt, std::vector<llama_token>& prompt_tokens) {
    return _m_pimpl->tokenize_prompt(prompt, prompt_tokens);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
ModelStatus Llama3<INPUT, OUTPUT>::get_model_stat() const {
    return _m_pimpl->get_model_stat();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
void Llama3<INPUT, OUTPUT>::clear_kv_cache_cell() const {
    return _m_pimpl->clear_kv_cache_cell();
}

}
}
}
}