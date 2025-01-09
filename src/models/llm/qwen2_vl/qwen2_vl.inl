/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: Qwen2VL.cpp
 * Date: 25-1-6
 ************************************************/

#include "qwen2_vl.h"

#include <regex>

#include "glog/logging.h"
#include "fmt/format.h"
#include "llama_cpp/llama.h"
#include "rapidjson/document.h"
#include "workflow/WFFacilities.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/HttpUtil.h"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/llm/llm_datatype.hpp"
#include "models/llm/qwen2_vl/clip.h"

namespace jinq {
namespace models {
namespace llm {

using jinq::common::Base64;
using jinq::common::CvUtils;
using jinq::common::Timestamp;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::io_define::llm::vlm::mat_input;
using jinq::models::io_define::llm::vlm::file_input;
using jinq::models::io_define::llm::vlm::bytes_input;
using jinq::models::io_define::llm::vlm::base64_input;
using jinq::models::io_define::llm::vlm::std_vlm_output;

namespace qwen2_vl {

namespace qwen2_vl_impl {

using internal_input = bytes_input;
using internal_output = std_vlm_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, mat_input>::value, internal_input>::type transform_input(const INPUT& in) {
    internal_input result;
    const cv::Mat& image = in.image;
    if (image.empty() || !image.data) {
        LOG(INFO) << "invalid opencv mat data or empty opencv mat";
        return result;
    }

    std::vector<unsigned char> buffer;
    cv::imencode(".jpg", image, buffer);
    result.image_bytes = new unsigned char[buffer.size()];
    std::memcpy(result.image_bytes, buffer.data(), buffer.size());

    result.bytes_length = buffer.size();
    result.text = in.text;

    return result;
}

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, file_input>::value, internal_input>::type transform_input(const INPUT& in) {
    internal_input result;
    auto image_path = in.image_path;
    if (!image_path.empty() && !FilePathUtil::is_file_exist(image_path)) {
        LOG(ERROR) << fmt::format("image file: {} not exist", image_path);
        return result;
    }
    if (image_path.empty()) {
        return result;
    }

    std::ifstream file(image_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG(ERROR) << fmt::format("Failed to open file: {}", image_path);
        return result;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    result.image_bytes = new unsigned char[file_size];
    if (!file.read(reinterpret_cast<char*>(result.image_bytes), file_size)) {
        LOG(ERROR) << fmt::format("Failed to read file: {}", image_path);
        delete[] result.image_bytes;
        result.image_bytes = nullptr;
        return result;
    }

    result.bytes_length = static_cast<size_t>(file_size);
    result.text = in.text;

    file.close();
    return result;
}

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, bytes_input>::value, internal_input>::type transform_input(
    const INPUT& in) {
    return in;
}

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, base64_input>::value, internal_input>::type transform_input(
    const INPUT& in) {
    internal_input result;
    std::string& img_b64_str = in.b64_image;
    if (img_b64_str.empty()) {
        LOG(ERROR) << "empty base64 image data";
        return result;
    }

    auto image_str = Base64::base64_decode(img_b64_str);
    std::vector<unsigned char> buffer(image_str.begin(), image_str.end());
    result.image_bytes = new unsigned char[buffer.size()];
    std::memcpy(result.image_bytes, buffer.data(), buffer.size());

    result.bytes_length = buffer.size();
    result.text = in.text;

    return result;
}

/***
* transform different type of internal output into external output
* @tparam EXTERNAL_OUTPUT
* @tparam dummy
* @param in
* @return
 */
template <typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std_vlm_output >::value, std_vlm_output >::type
transform_output(const qwen2_vl_impl::internal_output& internal_out) {
    return internal_out;
}

}

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class Qwen2VL<INPUT, OUTPUT>::Impl {
public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        if (nullptr != _m_smpl_chain) {
            llama_sampler_free(_m_smpl_chain);
            _m_smpl_chain = nullptr;
        }
        if (nullptr != _m_clip_ctx) {
            clip_free(_m_clip_ctx);
            _m_clip_ctx = nullptr;
        }
        if (nullptr != _m_llm_ctx) {
            llama_free(_m_llm_ctx);
            _m_llm_ctx = nullptr;
        }
        if (nullptr != _m_llm_model) {
            llama_free_model(_m_llm_model);
            _m_llm_model = nullptr;
        }
        llama_backend_free();
    };

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
     * @param config
     * @return
     */
    StatusCode init(const toml::value& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @param dialog
     * @param generate_output
     * @param truncate
     * @return
     */
    StatusCode chat_completion(Dialog& dialog, std::string& generate_output);

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
    llama_perf_context_data get_context_perf() const {
        return llama_perf_context(_m_llm_ctx);
    }

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

private:
    // vision tower params
    clip_ctx* _m_clip_ctx = nullptr;

    // language tower params
    llama_model_params _m_llm_model_params = llama_model_default_params();
    llama_model* _m_llm_model = nullptr;
    llama_context_params _m_llm_ctx_params = llama_context_default_params();
    llama_context* _m_llm_ctx = nullptr;
    llama::common_params_sampling _m_smpl_params{};
    llama_sampler* _m_smpl_chain = nullptr;
    llama_sampler* _m_smpl_grmr = nullptr;

    // init flag
    bool _m_successfully_initialized = false;

private:
    /***
     *
     * @param need_grama
     */
    StatusCode init_sampler();

    /***
     *
     * @param input_image_bytes
     * @param bytes_length
     * @param out_img_embeds
     * @return
     */
    StatusCode encode_image(const unsigned char* input_image_bytes, int bytes_length, std::vector<float>& out_img_embeds);

    /***
     *
     * @param text
     * @param add_special
     * @param parse_special
     * @param out_tokens
     * @return
     */
    StatusCode common_tokenize(const std::string& text, bool add_special, bool parse_special,
                               std::vector<int32_t>& out_tokens);

    /***
     *
     * @param token
     * @param special
     * @return
     */
    std::string common_token_to_piece(const llama_token& token, bool special = true);

    /***
     *
     * @param tokens
     * @param n_batch
     * @param n_past
     * @param st_pos_id
     * @return
     */
    StatusCode inference_tokens(std::vector<int32_t >& tokens, int n_batch, int* n_past, int* st_pos_id);

    /***
     *
     * @param text
     * @param n_batch
     * @param n_past
     * @param st_pos_id
     * @param add_bos
     * @return
     */
    StatusCode prefill_text_prompt(const std::string& text, int n_batch, int* n_past, int* st_pos_id, bool add_bos);

    /***
     *
     * @param image_embd
     * @param n_img_tokens
     * @param n_batch
     * @param n_past
     * @param st_pos_id
     * @param img_w
     * @param img_h
     * @return
     */
    StatusCode prefill_vision_prompt(
        std::vector<float>& image_embd, int n_img_tokens, int n_batch, int* n_past, int* st_pos_id, int img_w, int img_h);

    /***
     *
     * @param idx
     * @param out_sampled_token
     * @param grammar_first
     * @return
     */
    bool llama_sample(int idx, llama_token& out_sampled_token, bool grammar_first = false);

    /***
     *
     * @param n_past
     * @param st_pos_id
     * @param out_piece
     * @return
     */
    StatusCode autoregressive_generate(int* n_past, int* st_pos_id, std::string& out_piece);

    /***
     *
     * @param image_url
     * @param bytes_data
     * @return
     */
    StatusCode parse_image_url_data(const std::string& image_url, bytes_input& bytes_data);
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::init(const toml::value& config) {
    auto qwen_cfg = config.at("QWEN2-VL");

    // init language tower
    std::string language_model_path;
    if (!qwen_cfg.contains("llm_model_path")) {
        LOG(ERROR) << "Config doesn\'t have llm_model_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        language_model_path = qwen_cfg.at("llm_model_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(language_model_path)) {
        LOG(ERROR) << fmt::format("llm model file: {} not exist", language_model_path);
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    // init llama backend
    llama_backend_init();
    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;
    llama_numa_init(numa);

    // load llama model
    auto n_gpu_layers = static_cast<int32_t >(qwen_cfg.at("n_gpu_layers").as_integer());
    auto main_gpu_device_id = static_cast<int32_t >(qwen_cfg.at("main_gpu_device").as_integer());
    _m_llm_model_params.devices  = nullptr; // all available devices
    _m_llm_model_params.n_gpu_layers = n_gpu_layers; // number of layers to store in VRAM
    _m_llm_model_params.main_gpu = main_gpu_device_id;
    _m_llm_model_params.split_mode = LLAMA_SPLIT_MODE_LAYER; // how to split model cross gpus
    _m_llm_model_params.vocab_only = false;
    _m_llm_model_params.use_mmap = true; // use mmap for faster loads
    _m_llm_model_params.use_mlock = false; // use mlock to keep model in memory
    _m_llm_model_params.check_tensors = false;
    if (qwen_cfg.contains("vocab_only")) {
        _m_llm_model_params.vocab_only = qwen_cfg.at("vocab_only").as_boolean();
    }
    _m_llm_model = llama_load_model_from_file(language_model_path.c_str(), _m_llm_model_params);
    if (_m_llm_model == nullptr) {
        LOG(ERROR) << "load llm model from: " << language_model_path << " failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init vision tower
    std::string vision_model_path = qwen_cfg["mmproj_model_path"].as_string();
    if (!FilePathUtil::is_file_exist(vision_model_path)) {
        LOG(ERROR) << fmt::format("clip vision model file: {} not exist", vision_model_path);
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    std::string device = qwen_cfg["vision_model_device"].as_string();
    if (device == "cuda") {
        int device_id = static_cast<int>(qwen_cfg["vision_model_device_id"].as_integer());
        _m_clip_ctx = clip_model_load_cuda(vision_model_path.c_str(), device_id, 1);
    } else {
        _m_clip_ctx = clip_model_load(vision_model_path.c_str(), 1);
    }
    if (nullptr == _m_clip_ctx) {
        LOG(ERROR) << "init clip model ctx failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init llama model ctx
    if (!config.contains("CONTEXT")) {
        LOG(ERROR) << "Config file does not contain CONTEXT section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    toml::value ctx_cfg = config.at("CONTEXT");
    auto ctx_size = llama_n_ctx_train(_m_llm_model);
    if (ctx_cfg.contains("context_size")) {
        ctx_size = static_cast<int32_t >(ctx_cfg.at("context_size").as_integer());
    }
    _m_llm_ctx_params.n_ctx = ctx_size <= llama_n_ctx_train(_m_llm_model) ? ctx_size : llama_n_ctx_train(
                                  _m_llm_model); // context size
    _m_llm_ctx_params.n_batch = _m_llm_ctx_params.n_ctx /
                                4; // logical batch size for prompt processing (must be >=32 to use BLAS)
    _m_llm_ctx_params.n_ubatch = 512; // physical batch size for prompt processing (must be >=32 to use BLAS)
    _m_llm_ctx_params.logits_all = false; // return logits for all tokens in the batch
    _m_llm_ctx_params.embeddings = false;  // get only sentence embedding
    _m_llm_ctx_params.flash_attn = false; // flash attention
    _m_llm_ctx_params.no_perf = true; // no performance metrics
    _m_llm_ctx_params.offload_kqv = true; // disable KV offloading
    if (_m_llm_model_params.vocab_only) {
        _m_llm_ctx_params = llama_context_default_params();
    }
    _m_llm_ctx = llama_new_context_with_model(_m_llm_model, _m_llm_ctx_params);
    if (nullptr == _m_llm_ctx) {
        LOG(ERROR) << "failed to create the llama_context";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init sampler
    auto smpl_cfg = config.at("SAMPLER");
    _m_smpl_params.min_keep = static_cast<int32_t >(smpl_cfg.at("min_keep").as_integer());
    _m_smpl_params.top_k = static_cast<int32_t>(smpl_cfg.at("top_k").as_integer());
    _m_smpl_params.top_p = static_cast<float>(smpl_cfg.at("top_p").as_floating());
    _m_smpl_params.min_p = static_cast<float>(smpl_cfg.at("min_p").as_floating());
    _m_smpl_params.temp = static_cast<float>(smpl_cfg.at("temp").as_floating());
    _m_smpl_params.no_perf = smpl_cfg.at("no_perf").as_boolean();
    init_sampler();

    std::string result = "logits ";
    for (int i = 0; i < llama_sampler_chain_n(_m_smpl_chain); i++) {
        const auto* smpl = llama_sampler_chain_get(_m_smpl_chain, i);
        result += std::string("-> ") + llama_sampler_name(smpl) + " ";
    }
    LOG(INFO) << result;

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
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    auto internal_in = qwen2_vl_impl::transform_input(in);
    // encode input image
    std::vector<float> image_embds;
    if (nullptr != internal_in.image_bytes) {
        auto status = encode_image(internal_in.image_bytes, internal_in.bytes_length, image_embds);
        if (status != StatusCode::OK) {
            LOG(ERROR) << fmt::format("encode input image failed");
            return status;
        }
    }
    auto n_embed_dims = clip_n_mmproj_embd(_m_clip_ctx);
    auto n_img_patches = image_embds.size() / n_embed_dims;
    auto image_size = clip_get_load_image_size(_m_clip_ctx);

    // prepare prompt
    std::string& prompt = internal_in.text;
    std::string system_prompt;
    std::string user_prompt;

    std::string start_tag = "<|im_start|>";
    std::string end_tag = "<|im_end|>";
    if (prompt.find(start_tag) != std::string::npos) {
        size_t start = prompt.find(start_tag);
        size_t end = prompt.find(end_tag);
        while (start <= end) {
            if (start != std::string::npos && end != std::string::npos && end > start) {
                size_t contentStart = start + start_tag.length();
                std::string extracted = prompt.substr(contentStart, end - contentStart);
                if (extracted.find("system") == 0) {
                    system_prompt = extracted;
                }
                if (extracted.find("user") == 0) {
                    user_prompt = extracted;
                }
            }
            if (end + end_tag.length() > prompt.length()) {
                break;
            }
            std::string left_str = prompt.substr(end + end_tag.length());
            start = left_str.find(start_tag);
            start += (end + end_tag.length());
            end = left_str.find(end_tag);
            end += (end + end_tag.length());
        }
        system_prompt = fmt::format("{}{}{}\n", start_tag, system_prompt, end_tag);
        user_prompt = fmt::format("{}{}{}\n<|im_start|>assistant\n", start_tag, user_prompt, end_tag);
    } else {
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
        user_prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + prompt +
                      "<|im_end|>\n<|im_start|>assistant\n";
    }

    // prefill system prompt
    int n_past = 0;
    int cur_pos_id = 0;
    const int max_tgt_len = llama_n_ctx(_m_llm_ctx);
    auto status = prefill_text_prompt(system_prompt, _m_llm_ctx_params.n_batch, &n_past, &cur_pos_id, true);
    if (status != StatusCode::OK) {
        LOG(ERROR) << fmt::format("prefill system prompt: {} failed", system_prompt);
        return status;
    }

    // inference vision embedding
    if (!image_embds.empty()) {
        status = prefill_vision_prompt(
                     image_embds, n_img_patches, _m_llm_ctx_params.n_batch, &n_past, &cur_pos_id, image_size->width, image_size->height);
        if (status != StatusCode::OK) {
            LOG(ERROR) << fmt::format("decode vision embedding failed, status: ", std::to_string(status));
            return status;
        }
    }

    // prefill user prompt
    status = prefill_text_prompt(user_prompt, _m_llm_ctx_params.n_batch, &n_past, &cur_pos_id, false);
    if (status != StatusCode::OK) {
        LOG(ERROR) << fmt::format("prefill user prompt: {} failed", system_prompt);
        return status;
    }

    // generate response text
    std::string response;
    for (int i = 0; i < max_tgt_len; ++i) {
        std::string out_str;
        status = autoregressive_generate(&n_past, &cur_pos_id, out_str);
        if (status != StatusCode::OK) {
            LOG(ERROR) << fmt::format("autoregressive generate text failed, status: {}", std::to_string(status));
            return status;
        }
        if (out_str == "</s>" || out_str == "###") {
            break;
        }
        if (strstr(response.c_str(), "<|im_end|>")) {
            break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        }
        response += out_str;
        if (strstr(response.c_str(), "<|im_start|>")) {
            break; // Yi-34B llava-1.6
        }
        if (strstr(response.c_str(), "USER:")) {
            break; // mistral llava-1.6
        }
    }

    qwen2_vl_impl::internal_output internal_out = response;
    out = qwen2_vl_impl::transform_output<OUTPUT>(internal_out);

    return status;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param dialog
 * @param generate_output
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::chat_completion(jinq::models::llm::Dialog& dialog, std::string& generate_output) {
    // assemble text input
    std::string text_prompt;
    std::string image_url;
    for (auto& msg : dialog.messages) {
        std::string sys_str;
        std::string user_str;
        std::string assis_str;
        if (msg.role == "system") {
            sys_str = fmt::format("<|im_start|>system\n{}<|im_end|>\n", msg.content);
        } else if (msg.role == "user") {
            rapidjson::Document doc;
            doc.Parse(msg.content.c_str());
            if (doc.HasParseError()) {
                LOG(WARNING) << fmt::format("invalid json string: {} has parse error", msg.content);
                continue;
            }
            std::string text;
            if (!doc.HasMember("content")) {
                LOG(WARNING) << fmt::format("invalid json string: {}, missing \'content\' field", msg.content);
                continue;
            }
            for (auto& c : doc["content"].GetArray()) {
                std::string type = c["type"].GetString();
                if (type == "text") {
                    text = c["text"].GetString();
                } else if (type == "image") {
                    image_url = c["image"].GetString();
                } else {
                    LOG(WARNING) << fmt::format("not supported type: {}", type);
                }
            }
            user_str = fmt::format("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n", text);
        } else {
            assis_str = fmt::format("<|im_start|>assistant\n{}<|im_end|>\n", msg.content);
        }
        text_prompt += sys_str;
        text_prompt += user_str;
        text_prompt += assis_str;
    }

    // prepare image data
    bytes_input input;
    input.text = text_prompt;
    if (!image_url.empty()) {
        auto status = parse_image_url_data(image_url, input);
        if (status != StatusCode::OK) {
            LOG(ERROR) << fmt::format("parse image url: {} failed, status: {}", image_url, std::to_string(status));
            return status;
        }
    }

    // run model inference
    qwen2_vl_impl::internal_output output;
    auto status = run(input, output);
    if (status != StatusCode::OK) {
        LOG(ERROR) << fmt::format("run qwen2-vl model inference session failed, status: {}", std::to_string(status));
        return status;
    }
    generate_output = qwen2_vl_impl::transform_output<OUTPUT>(output);

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
ModelStatus Qwen2VL<INPUT, OUTPUT>::Impl::get_model_stat() const {
    ModelStatus stat{};
    stat.n_ctx_size = llama_n_ctx(_m_llm_ctx);
    stat.kv_cache_cell_nums = llama_get_kv_cache_used_cells(_m_llm_ctx);
    stat.embed_dims = llama_n_embd(_m_llm_model);
    stat.has_vision_tower = true;
    stat.clip_embedding_dims = clip_n_mmproj_embd(_m_clip_ctx);
    stat.clip_hidden_size = clip_hidden_size(_m_clip_ctx);
    return stat;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
void Qwen2VL<INPUT, OUTPUT>::Impl::clear_kv_cache_cell() const {
    llama_kv_cache_clear(_m_llm_ctx);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param need_grama
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::init_sampler() {
    auto lsmpl_params = llama_sampler_chain_default_params();
    lsmpl_params.no_perf = _m_smpl_params.no_perf;
    _m_smpl_chain = llama_sampler_chain_init(lsmpl_params);

    // add sampler to chain
    llama_sampler_chain_add(
        _m_smpl_chain,
        llama_sampler_init_logit_bias(
            llama_n_vocab(_m_llm_model),
            _m_smpl_params.logit_bias.size(),
            _m_smpl_params.logit_bias.data()
        )
    );
    llama_sampler_chain_add(
        _m_smpl_chain,
        llama_sampler_init_penalties(
            llama_n_vocab(_m_llm_model),
            llama_token_eos(_m_llm_model),
            llama_token_nl(_m_llm_model),
            _m_smpl_params.penalty_last_n,
            _m_smpl_params.penalty_repeat,
            _m_smpl_params.penalty_freq,
            _m_smpl_params.penalty_present,
            _m_smpl_params.penalize_nl,
            _m_smpl_params.ignore_eos
        )
    );
    auto& params = _m_smpl_params;
    if (params.mirostat == 0) {
        for (const auto& cnstr : params.samplers) {
            switch (cnstr) {
            case llama::COMMON_SAMPLER_TYPE_DRY: {
                std::vector<const char*> c_breakers;
                c_breakers.reserve(params.dry_sequence_breakers.size());
                for (const auto& str : params.dry_sequence_breakers) {
                    c_breakers.push_back(str.c_str());
                }
                llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_dry(_m_llm_model, params.dry_multiplier, params.dry_base,
                                        params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
            }
            break;
            case llama::COMMON_SAMPLER_TYPE_TOP_K:
                llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_top_k(params.top_k));
                break;
            case llama::COMMON_SAMPLER_TYPE_TOP_P:
                llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_top_p(params.top_p, params.min_keep));
                break;
            case llama::COMMON_SAMPLER_TYPE_MIN_P:
                llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_min_p(params.min_p, params.min_keep));
                break;
            case llama::COMMON_SAMPLER_TYPE_XTC:
                llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold,
                                        params.min_keep, params.seed));
                break;
            case llama::COMMON_SAMPLER_TYPE_TYPICAL_P:
                llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_typical(params.typ_p, params.min_keep));
                break;
            case llama::COMMON_SAMPLER_TYPE_TEMPERATURE:
                llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_temp_ext(params.temp, params.dynatemp_range,
                                        params.dynatemp_exponent));
                break;
            case llama::COMMON_SAMPLER_TYPE_INFILL:
                llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_infill(_m_llm_model));
                break;
            default:
                LOG(WARNING) << fmt::format("unknown sampler type: {}", static_cast<int>(cnstr));
                break;
            }
        }
        llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_dist(params.seed));
    } else if (params.mirostat == 1) {
        llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_mirostat(llama_n_vocab(_m_llm_model), params.seed,
                                params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(_m_smpl_chain, llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau,
                                params.mirostat_eta));
    } else {
        LOG(ERROR) << "unknown mirostat version";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_smpl_grmr = llama_sampler_init_grammar(_m_llm_model, params.grammar.c_str(), "root");

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param input_image_bytes
 * @param bytes_length
 * @param out_img_embeds
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::encode_image(
    const unsigned char* input_image_bytes, int bytes_length, std::vector<float>& out_img_embeds) {
    // load input image
    auto* image_u8 = clip_image_u8_init();
    if (!clip_image_load_from_bytes(input_image_bytes, bytes_length, image_u8)) {
        LOG(ERROR) << fmt::format("load image from bytes failed");
        return StatusCode::VLM_QWEN_ENCODE_IMAGE_FAILED;
    }
//    LOG(INFO) << fmt::format("input image size: [width / height] --> [{} / {}]", image_u8->nx, image_u8->ny);

    // malloc embedding vector size
    clip_image_f32 image_f32;
    image_f32.nx = image_u8->nx;
    image_f32.ny = image_u8->ny;
    auto n_patches = clip_n_patches_by_img(_m_clip_ctx, &image_f32);
//    LOG(INFO) << fmt::format("clip input image into patched: {}", n_patches);
    auto n_embed_dims = clip_n_mmproj_embd(_m_clip_ctx);
//    LOG(INFO) << fmt::format("clip patch embedding dims: {}", n_embed_dims);
    out_img_embeds.resize(n_patches * n_embed_dims, 0.0f);

    // preprocess input image
    clip_image_f32_batch img_res_v{};
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    if (!clip_image_preprocess(_m_clip_ctx, image_u8, &img_res_v)) {
        LOG(ERROR) << fmt::format("{}: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        clip_image_u8_free(image_u8);
        return StatusCode::VLM_QWEN_ENCODE_IMAGE_FAILED;
    }

    // encode image patch
    if (nullptr == _m_clip_ctx->load_image_size) {
        auto* load_image_size = clip_image_size_init();
        load_image_size->width = img_res_v.data[0].nx;
        load_image_size->height = img_res_v.data[0].ny;
        clip_add_load_image_size(_m_clip_ctx, load_image_size);
    } else {
        _m_clip_ctx->load_image_size->width = img_res_v.data[0].nx;
        _m_clip_ctx->load_image_size->height = img_res_v.data[0].ny;
    }
    auto successfully_encoded = clip_image_encode(_m_clip_ctx, 4, &img_res_v.data[0], out_img_embeds.data());
    if (!successfully_encoded) {
        LOG(ERROR) << fmt::format("clip encode image failed");
        return StatusCode::VLM_QWEN_ENCODE_IMAGE_FAILED;
    }
    _m_clip_ctx->load_image_size->width = image_u8->nx;
    _m_clip_ctx->load_image_size->height = image_u8->ny;

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param text
 * @param add_special
 * @param parse_special
 * @param out_tokens
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::common_tokenize(
    const std::string& text, bool add_special, bool parse_special, std::vector<int32_t>& out_tokens) {
    // check input text
    if (text.empty()) {
        LOG(ERROR) << "input prompt is empty";
        return StatusCode::TOKENIZE_FAILED;
    }

    // tokenize
    int32_t n_tokens = static_cast<int32_t >(text.length()) + 2 * add_special;
    out_tokens.resize(n_tokens);
    auto text_len = static_cast<int32_t >(text.length());
    auto out_tokens_len = static_cast<int32_t >(out_tokens.size());
    n_tokens = llama_tokenize(_m_llm_model, text.data(), text_len, out_tokens.data(), out_tokens_len, add_special,
                              parse_special);
    if (n_tokens < 0) {
        out_tokens.resize(-n_tokens);
        int32_t check = llama_tokenize(
                            _m_llm_model, text.data(), text_len, out_tokens.data(), out_tokens_len, add_special, parse_special);
        if (check != -n_tokens) {
            LOG(ERROR) << fmt::format("tokenize text: {} failed, check nums: {}, -n_tokens nums: {}. They are not equal",
                                      text, check, -n_tokens);
            return StatusCode::TOKENIZE_FAILED;
        }
    } else {
        out_tokens.resize(n_tokens);
    }

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param token
 * @param special
 * @return
 */
template <typename INPUT, typename OUTPUT>
std::string Qwen2VL<INPUT, OUTPUT>::Impl::common_token_to_piece(const llama_token& token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(_m_llm_model, token, &piece[0], static_cast<int32_t >(piece.size()), 0,
                        special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(_m_llm_model, token, &piece[0], static_cast<int32_t>(piece.size()), 0, special);
        if (check != -n_chars) {
            LOG(ERROR) << fmt::format(
                "convert token: {} into piece failed, check nums: {}, -n_chars nums: {}, not equal", token, check, -n_chars);
            return "";
        }
    } else {
        piece.resize(n_chars);
    }

    return piece;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param tokens
 * @param n_batch
 * @param n_past
 * @param st_pos_id
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::inference_tokens(std::vector<int32_t>& tokens, int n_batch, int* n_past,
        int* st_pos_id) {
    auto token_size = static_cast<int32_t>(tokens.size());
    std::vector<llama_pos> pos;
    for (int i = 0; i < token_size; i += n_batch) {
        int n_eval = static_cast<int>(tokens.size()) - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        auto batch = llama_batch_get_one(&tokens[i], n_eval);
        // TODO: add mrope pos ids somewhere else
        pos.resize(batch.n_tokens * 4);
        std::fill(pos.begin(), pos.end(), 0);
        for (int j = 0; j < batch.n_tokens * 3; j ++) {
            pos[j] = *st_pos_id + (j % batch.n_tokens);
        }
        batch.pos = pos.data();

        int successfully_decode = llama_decode(_m_llm_ctx, batch);
        if (successfully_decode == 1) {
            LOG(WARNING) << "llama generate failed. could not find a KV slot for the batch "
                         "(try reducing the size of the batch or increase the context)";
        } else if (successfully_decode < 0) {
            LOG(ERROR) << "llama decode failed code: " << successfully_decode;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        if (nullptr != n_past) {
            *n_past += n_eval;
        }
        *st_pos_id += n_eval;
    }

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param text
 * @param n_batch
 * @param n_past
 * @param st_pos_id
 * @param add_bos
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::prefill_text_prompt(
    const std::string& text, int n_batch, int* n_past, int* st_pos_id, bool add_bos) {
    if (text.empty()) {
        return StatusCode::OK;
    }
    std::vector<llama_token> embd_inp;
    auto status = common_tokenize(text, add_bos, true, embd_inp);
    if (status != StatusCode::OK) {
        return status;
    }
    status = inference_tokens(embd_inp, n_batch, n_past, st_pos_id);
    return status;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param image_embd
 * @param n_img_tokens
 * @param n_batch
 * @param n_past
 * @param st_pos_id
 * @param img_w
 * @param img_h
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::prefill_vision_prompt(
    std::vector<float>& image_embd, const int n_img_tokens, int n_batch, int* n_past, int* st_pos_id, int img_w,
    int img_h) {
    if (nullptr == n_past || nullptr == st_pos_id) {
        LOG(ERROR) << fmt::format("failed to decode image embeddings. n_past or st_pos_id is nullptr");
        return StatusCode::LLM_LLAMA_DECODE_FAILED;
    }

    int n_embd  = llama_n_embd(_m_llm_model);
    const int patch_size = 14 * 2;
    const int ph = img_h / patch_size + (img_h % patch_size > 0);
    const int pw = img_w / patch_size + (img_w % patch_size > 0);

    std::vector<llama_pos> mrope_pos;
    mrope_pos.resize(n_img_tokens * 4);
    for (int y = 0; y < ph; y++) {
        for (int x = 0; x < pw; x++) {
            int i = y * pw + x;
            mrope_pos[i] = *st_pos_id;
            mrope_pos[i + n_img_tokens] = *st_pos_id + y;
            mrope_pos[i + n_img_tokens * 2] = *st_pos_id + x;
            mrope_pos[i + n_img_tokens * 3] = 0;
        }
    }
    *st_pos_id += std::max(pw, ph);

    int processed = 0;
    std::vector<llama_pos> batch_mrope_pos;
    batch_mrope_pos.resize(n_img_tokens * 4);

    for (int i = 0; i < n_img_tokens; i += n_batch) {
        int n_eval = std::min(n_img_tokens - i, n_batch);

        std::fill(batch_mrope_pos.begin(), batch_mrope_pos.end(), 0);
        memcpy(batch_mrope_pos.data(), &mrope_pos[processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 1], &mrope_pos[n_img_tokens * 1 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 2], &mrope_pos[n_img_tokens * 2 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 3], &mrope_pos[n_img_tokens * 3 + processed], n_eval * sizeof(llama_pos));

        llama_batch batch = {
            int32_t(n_eval),                // n_tokens
            nullptr,                        // token
            image_embd.data() + i * n_embd,  // embed
            batch_mrope_pos.data(),         // pos
            nullptr,  // n_seq_id
            nullptr,  // seq_id
            nullptr,  // logits
        };

        int successfully_decode = llama_decode(_m_llm_ctx, batch);
        if (successfully_decode == 1) {
            LOG(WARNING) << "llama generate failed. could not find a KV slot for the batch "
                         "(try reducing the size of the batch or increase the context)";
        } else if (successfully_decode < 0) {
            LOG(ERROR) << fmt::format("failed to prefill image embeddings");
            LOG(ERROR) << "llama decode failed code: " << successfully_decode;
            return StatusCode::VLM_QWEN_DECODE_IMAGE_EMBEDDING_FAILED;
        }

        *n_past += n_eval;
        processed += n_eval;
    }

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param idx
 * @param out_sampled_token
 * @param grammar_first
 * @return
 */
template <typename INPUT, typename OUTPUT>
bool Qwen2VL<INPUT, OUTPUT>::Impl::llama_sample(int idx, llama_token& out_sampled_token, bool grammar_first) {
    // get logits
    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p;
    auto* logits = llama_get_logits_ith(_m_llm_ctx, idx);
    int n_vocab = llama_n_vocab(llama_get_model(_m_llm_ctx));
    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }
    cur_p = { cur.data(), cur.size(), -1, false };

    // chain sample
    if (grammar_first && _m_smpl_grmr != nullptr) {
        llama_sampler_apply(_m_smpl_grmr, &cur_p);
    }
    llama_sampler_apply(_m_smpl_chain, &cur_p);
    if (cur_p.selected == -1) {
        LOG(ERROR) << "no selected token during sampling - check your sampling configuration";
        return false;
    }
    const llama_token id = cur_p.data[cur_p.selected].id;
    if (grammar_first && _m_smpl_grmr != nullptr) {
        out_sampled_token = id;
        return true;
    }

    // check if sampled token fits the grammar
    llama_token_data single_token_data = { id, 1.0f, 0.0f };
    llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };
    llama_sampler_apply(_m_smpl_grmr, &single_token_data_array);
    bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
    if (is_valid) {
        out_sampled_token = id;
        return true;
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    logits = llama_get_logits_ith(_m_llm_ctx, idx);
    n_vocab = llama_n_vocab(llama_get_model(_m_llm_ctx));
    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }
    cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(_m_smpl_grmr,  &cur_p);
    llama_sampler_apply(_m_smpl_chain, &cur_p);
    if (cur_p.selected == -1) {
        LOG(ERROR) << "no selected token during sampling - check your sampling configuration";
        return false;
    }
    out_sampled_token = cur_p.data[cur_p.selected].id;

    // sampler accept
    llama_sampler_accept(_m_smpl_grmr, out_sampled_token);
    llama_sampler_accept(_m_smpl_chain, out_sampled_token);

    return true;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param n_past
 * @param st_pos_id
 * @param out_piece
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::autoregressive_generate(int* n_past, int* st_pos_id, std::string& out_piece) {
    // sample output token
    llama_token sampled_token;
    llama_sample(-1, sampled_token, false);
//    auto successfully_sampled = llama_sample(-1, sampled_token, false);
//    if (!successfully_sampled) {
//        LOG(ERROR) << "llama sample new token failed";
//        return StatusCode::LLM_LLAMA_SAMPLE_NEW_TOKEN_FAILED;
//    }

    // convert token into piece
    if (llama_token_is_eog(_m_llm_model, sampled_token)) {
        out_piece = "</s>";
    } else {
        out_piece = common_token_to_piece(sampled_token, true);
    }

    // inference new tokens
    std::vector<llama_token> tokens;
    tokens.push_back(sampled_token);
    auto status = inference_tokens(tokens, 1, n_past, st_pos_id);

    return status;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param image_url
 * @param byte_data
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::Impl::parse_image_url_data(const std::string& image_url, bytes_input& bytes_data) {
    auto is_local_file = [](const std::string & url) -> bool {
        return FilePathUtil::is_file_exist(url);
    };
    auto is_url = [](const std::string & url) -> bool {
        std::vector<std::string> url_prefixes = {"http://", "https://", "ftp://"};
        return std::any_of(
                   url_prefixes.begin(), url_prefixes.end(),
                   [&](const std::string & protocol) -> bool {return url.find(protocol) != std::string::npos;});
    };
    auto is_b64 = [](const std::string & url) -> bool {
        if (url.rfind("data:", 0) == 0 && url.find(";base64,") != std::string::npos) {
            return true;
        }
        std::regex base64_regex("^[A-Za-z0-9+/]*(={0,2})$");
        return (url.length() % 4 == 0) && std::regex_match(url, base64_regex);
    };

    if (is_local_file(image_url)) {
        std::ifstream file(image_url, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG(ERROR) << fmt::format("Failed to open file: {}", image_url);
            return StatusCode::VLM_QWEN_PARSE_IMAGE_URL_FAILED;
        }

        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        bytes_data.image_bytes = new unsigned char[file_size];
        if (!file.read(reinterpret_cast<char*>(bytes_data.image_bytes), file_size)) {
            LOG(ERROR) << fmt::format("Failed to read file: {}", image_url);
            delete[] bytes_data.image_bytes;
            bytes_data.image_bytes = nullptr;
            return StatusCode::VLM_QWEN_PARSE_IMAGE_URL_FAILED;
        }
        bytes_data.bytes_length = file_size;
        return StatusCode::OK;
    } else if (is_url(image_url)) {
        WFFacilities::WaitGroup wait_group(1);
        StatusCode status = StatusCode::OK;
        WFHttpTask* task = WFTaskFactory::create_http_task(image_url, 5, 5, [&](WFHttpTask * h_task) -> void {
            int state = task->get_state();
            int error = task->get_error();
            protocol::HttpResponse* resp = task->get_resp();
            auto status_code = resp->get_status_code();
            if (state != WFT_STATE_SUCCESS) {
                LOG(ERROR) << fmt::format("Download file from {} failed, state: {}, error: {}, error msg: {}",
                                          image_url, state, error, WFGlobal::get_error_string(state, error));
                status = StatusCode::VLM_QWEN_PARSE_IMAGE_URL_FAILED;
                return;
            }
            auto resp_str = protocol::HttpUtil::decode_chunked_body(resp);
            if (std::strcmp(status_code, "200") != 0) {
                LOG(ERROR) << fmt::format("Download file from {} failed http status: {}, resp content: {}",
                                          image_url, status_code, resp_str);
                status = StatusCode::VLM_QWEN_PARSE_IMAGE_URL_FAILED;
                return;
            }
            std::vector<unsigned char> buffer(resp_str.begin(), resp_str.end());
            bytes_data.image_bytes = new unsigned char[buffer.size()];
            std::memcpy(bytes_data.image_bytes, buffer.data(), buffer.size());
            bytes_data.bytes_length = buffer.size();
            wait_group.done();
        });
        task->get_req()->set_method("GET");
        task->start();
        wait_group.wait();
        return status;
    } else if (is_b64(image_url)) {
        auto image_str = Base64::base64_decode(image_url);
        std::vector<unsigned char> buffer(image_str.begin(), image_str.end());
        bytes_data.image_bytes = new unsigned char[buffer.size()];
        std::memcpy(bytes_data.image_bytes, buffer.data(), buffer.size());
        bytes_data.bytes_length = buffer.size();
        return StatusCode::OK;
    } else {
        LOG(ERROR) << fmt::format("not supported image url: {}", image_url);
        return StatusCode::VLM_QWEN_PARSE_IMAGE_URL_FAILED;
    }
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
Qwen2VL<INPUT, OUTPUT>::Qwen2VL() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
Qwen2VL<INPUT, OUTPUT>::~Qwen2VL() = default; // NOLINT(*-redundant-declaration)

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::init(const toml::value& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
bool Qwen2VL<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode Qwen2VL<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param dialog
 * @param generate_output
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Qwen2VL<INPUT, OUTPUT>::chat_completion(Dialog& dialog, std::string& generate_output) {
    return _m_pimpl->chat_completion(dialog, generate_output);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
ModelStatus Qwen2VL<INPUT, OUTPUT>::get_model_stat() const {
    return _m_pimpl->get_model_stat();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
void Qwen2VL<INPUT, OUTPUT>::clear_kv_cache_cell() const {
    return _m_pimpl->clear_kv_cache_cell();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
llama_perf_context_data Qwen2VL<INPUT, OUTPUT>::get_context_perf() const {
    return _m_pimpl->get_context_perf();
}

}
}
}
}