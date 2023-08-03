/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: openai_clip_text_encoder.cpp
 * Date: 23-6-26
 ************************************************/

#include "openai_clip_text_encoder.h"

#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/clip/simple_tokenizer.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;
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
    StatusCode init(const decltype(toml::parse(""))& cfg);

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
    // model file path
    std::string _m_model_path;

    // model compute thread nums
    uint16_t _m_thread_nums = 1;

    // model backend device
    std::string _m_model_device;

    // model input/output names
    std::string _m_input_ids_name;
    std::string _m_input_attention_mask_name;
    std::string _m_output_name;

    // tokenizer
    std::unique_ptr<SimpleTokenizer> _m_tokenizer;
    int _m_context_length = 77;
    bool _m_truncate_token = true;

    // model session
    std::unique_ptr<MNN::Interpreter> _m_net;
    MNN::Session* _m_session = nullptr;
    MNN::Tensor* _m_input_tensor = nullptr;
    MNN::Tensor* _m_output_tensor = nullptr;

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
    void tokenize(const std::string& input_text, std::vector<int32_t >& token_ids, std::vector<int32_t >& attn_mask);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode OpenAiClipTextEncoder::Impl::init(const decltype(toml::parse("")) &cfg) {
    // init text encoder configs
    auto cfg_content = cfg.at("OPENAI_CLIP_TEXT_ENCODER");
    _m_model_path = cfg_content["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "openai clip text encoder model file path: " << _m_model_path << " not exists";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init session
    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_path.c_str()));
    _m_thread_nums = cfg_content["model_threads_num"].as_integer();
    _m_model_device = cfg_content["compute_backend"].as_string();
    MNN::ScheduleConfig mnn_config;
    mnn_config.numThread = _m_thread_nums;
    mnn_config.type = MNN_FORWARD_CPU;
    if (std::strcmp(_m_model_device.c_str(), "cuda") == 0) {
        mnn_config.type = MNN_FORWARD_CUDA;
    }
    MNN::BackendConfig backend_config;
    if (!cfg_content.contains("backend_precision_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_precision_mode field default Precision_Normal";
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(cfg_content.at("backend_precision_mode").as_integer());
    }
    if (!cfg_content.contains("backend_power_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_power_mode field default Power_Normal";
        backend_config.power = MNN::BackendConfig::Power_Normal;
    } else {
        backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(cfg_content.at("backend_power_mode").as_integer());
    }
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);

    // fetch input/output tensors
    _m_input_ids_name = "input";
    _m_input_tensor = _m_net->getSessionInput(_m_session, _m_input_ids_name.c_str());
    _m_input_shape = _m_input_tensor->shape();
    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "fetch input ids tensor failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_output_name = "output";
    _m_output_tensor = _m_net->getSessionOutput(_m_session, _m_output_name.c_str());
    _m_output_shape = _m_output_tensor->shape();
    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "fetch output tensor failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init tokenizer
    _m_tokenizer = std::make_unique<SimpleTokenizer>();
    auto status = _m_tokenizer->init(cfg);
    if (!_m_tokenizer->is_successfully_initialized()) {
        LOG(ERROR) << "init simple tokenizer failed, status code: " << status;
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_context_length = static_cast<int>(cfg.at("TOKENIZER").at("context_length").as_integer());
    _m_truncate_token = cfg.at("TOKENIZER").at("truncate_context").as_boolean();

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load openai clip vit encoder";
    return StatusCode::OJBK;
}

/***
 *
 * @param input_image
 * @param text_embeddings
 * @return
 */
StatusCode OpenAiClipTextEncoder::Impl::encode(
    const std::string& input_text,
    std::vector<float> &text_embeddings) {
    // tokenize input text
    std::vector<int32_t > token_ids;
    std::vector<int32_t > attn_masks;
    tokenize(input_text, token_ids, attn_masks);
    if (token_ids.size() < 3) {
        LOG(ERROR) << "tokenization failed, source text: " << input_text;
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // run encoder
    auto input_tensor_user = MNN::Tensor(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, token_ids.data(), input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);

    _m_net->runSession(_m_session);

    MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    _m_output_tensor->copyToHostTensor(&output_tensor_user);

    auto embeds_size = std::accumulate(
        std::begin(_m_output_shape), std::end(_m_output_shape), 1, std::multiplies());
    text_embeddings.resize(embeds_size);
    auto img_embeds_val = output_tensor_user.host<float>();
    for (auto idx = 0; idx < embeds_size; ++idx) {
        text_embeddings[idx] = img_embeds_val[idx];
    }

    return StatusCode::OJBK;
}

/***
 *
 * @param input_text
 * @param token_ids
 * @param attn_mask
 */
void OpenAiClipTextEncoder::Impl::tokenize(
    const std::string &input_text, std::vector<int32_t> &token_ids, std::vector<int32_t> &attn_mask) {
    std::vector<int32_t> text_tokens;
    _m_tokenizer->tokenize(input_text, text_tokens);

    token_ids.resize(_m_context_length);
    attn_mask.resize(_m_context_length);
    for (auto idx = 0; idx < _m_context_length; ++idx) {
        token_ids[idx] = 0;
        attn_mask[idx] = 0;
    }
    for (int idx = 0; idx < text_tokens.size(); ++idx) {
        token_ids[idx] = text_tokens[idx];
        attn_mask[idx] = 1;
    }
    if (text_tokens.size() > _m_context_length && _m_truncate_token) {
        token_ids[_m_context_length - 1] = text_tokens[text_tokens.size() - 1];
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
StatusCode OpenAiClipTextEncoder::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_text
 * @param text_embeddings
 * @return
 */
StatusCode OpenAiClipTextEncoder::encode(const std::string& input_text, std::vector<float> &text_embeddings) {
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

}
}
}