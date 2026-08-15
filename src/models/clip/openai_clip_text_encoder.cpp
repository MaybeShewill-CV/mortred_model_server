/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: openai_clip_text_encoder.cpp
 * Date: 23-6-26
 ************************************************/

#include "openai_clip_text_encoder.h"

#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/clip/simple_tokenizer.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::cv_utils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::clip::SimpleTokenizer;

namespace clip {

class OpenAiClipTextEncoder::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init(const toml::table& cfg);

    /***
     *
     * @param input_text
     * @param text_embeddings
     * @return
     */
    StatusCode encode(const std::string& input_text, std::vector<float>& text_embeddings);

    /***
     *
     * @return
     */
    std::vector<int> get_encoder_input_shape() const {
        return _m_input_shape;
    }

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

  private:
    // tokenizer
    SimpleTokenizer _m_tokenizer;
    int _m_context_length = 77;
    bool _m_truncate_token = true;

    // MNN runtime (owns interpreter/session/tensors)
    jinq::models::MnnNet _m_net;

    // model input/output shape info
    std::vector<int> _m_input_shape;
    std::vector<int> _m_output_shape;

    // init flag
    bool _m_successfully_init_model = false;

  private:
    /***
     *
     * @param input_text
     * @param token_ids
     * @param attn_mask
     */
    void tokenize(const std::string& input_text, std::vector<int32_t>& token_ids, std::vector<int32_t>& attn_mask);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode OpenAiClipTextEncoder::Impl::init(const toml::table& cfg) {
    const toml::table* cfg_content_ptr = cfg["OPENAI_CLIP_TEXT_ENCODER"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section OPENAI_CLIP_TEXT_ENCODER missing or not a table";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    auto init_status = _m_net.init(cfg_content, {"input"}, {"output"});
    if (init_status != StatusCode::OK) {
        _m_successfully_init_model = false;
        return init_status;
    }
    _m_input_shape = _m_net.input("input")->shape();
    _m_output_shape = _m_net.output("output")->shape();

    // init tokenizer
    auto status = _m_tokenizer.init(cfg);
    if (!_m_tokenizer.is_successfully_initialized()) {
        LOG(ERROR) << "init simple tokenizer failed, status code: " << status;
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (!cfg.contains("TOKENIZER")) {
        LOG(ERROR) << "Config section TOKENIZER missing";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table* tokenizer_cfg_ptr = cfg["TOKENIZER"].as_table();
    if (tokenizer_cfg_ptr == nullptr) {
        LOG(ERROR) << "Config section TOKENIZER missing or not a table";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& tokenizer_cfg = *tokenizer_cfg_ptr;
    _m_context_length = static_cast<int>(tokenizer_cfg["context_length"].value_or<int64_t>(77));
    _m_truncate_token = tokenizer_cfg["truncate_context"].value_or<bool>(false);

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load openai clip text encoder";
    return StatusCode::OK;
}

/***
 *
 * @param input_text
 * @param text_embeddings
 * @return
 */
StatusCode OpenAiClipTextEncoder::Impl::encode(
    const std::string& input_text,
    std::vector<float>& text_embeddings) {
    // tokenize input text
    std::vector<int32_t> token_ids;
    std::vector<int32_t> attn_masks;
    tokenize(input_text, token_ids, attn_masks);
    if (token_ids.size() < 3) {
        LOG(ERROR) << "tokenization failed, source text: " << input_text;
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // run encoder
    auto input_tensor_user = MNN::Tensor(_m_net.input("input"), MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!cv_utils::copy_image_to_tensor(input_tensor_data, token_ids, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("input")->copyFromHostTensor(&input_tensor_user);

    _m_net.run_session();

    MNN::Tensor output_tensor_user(_m_net.output("output"), MNN::Tensor::DimensionType::CAFFE);
    _m_net.output("output")->copyToHostTensor(&output_tensor_user);

    auto embeds_size = std::accumulate(
        std::begin(_m_output_shape), std::end(_m_output_shape), 1, std::multiplies());
    text_embeddings.resize(embeds_size);
    auto img_embeds_val = output_tensor_user.host<float>();
    for (auto idx = 0; idx < embeds_size; ++idx) {
        text_embeddings[idx] = img_embeds_val[idx];
    }

    return StatusCode::OK;
}

/***
 *
 * @param input_text
 * @param token_ids
 * @param attn_mask
 */
void OpenAiClipTextEncoder::Impl::tokenize(
    const std::string& input_text, std::vector<int32_t>& token_ids, std::vector<int32_t>& attn_mask) {
    if (_m_context_length <= 0) {
        LOG(ERROR) << "invalid context length: " << _m_context_length;
        return;
    }
    const size_t context_length = static_cast<size_t>(_m_context_length);

    std::vector<int32_t> text_tokens;
    _m_tokenizer.tokenize(input_text, text_tokens);

    // truncate BEFORE writing into the fixed-size buffers: never write past
    // context_length (regression: the old code wrote first and truncated after)
    if (text_tokens.size() > context_length) {
        if (_m_truncate_token) {
            const int32_t last_token = text_tokens.back();
            text_tokens.resize(context_length);
            text_tokens.back() = last_token;
        } else {
            text_tokens.resize(context_length);
        }
    }

    token_ids.resize(context_length);
    attn_mask.resize(context_length);
    for (size_t idx = 0; idx < context_length; ++idx) {
        token_ids[idx] = 0;
        attn_mask[idx] = 0;
    }
    for (size_t idx = 0; idx < text_tokens.size(); ++idx) {
        token_ids[idx] = text_tokens[idx];
        attn_mask[idx] = 1;
    }
}

/***
 *
 */
OpenAiClipTextEncoder::OpenAiClipTextEncoder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
OpenAiClipTextEncoder::~OpenAiClipTextEncoder() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode OpenAiClipTextEncoder::init(const toml::table& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_text
 * @param text_embeddings
 * @return
 */
StatusCode OpenAiClipTextEncoder::encode(const std::string& input_text, std::vector<float>& text_embeddings) {
    return _m_pimpl->encode(input_text, text_embeddings);
}

/***
 *
 * @return
 */
std::vector<int> OpenAiClipTextEncoder::get_encoder_input_shape() const {
    return _m_pimpl->get_encoder_input_shape();
}

/***
 *
 * @return
 */
bool OpenAiClipTextEncoder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}  // namespace clip
}  // namespace models
}  // namespace jinq
