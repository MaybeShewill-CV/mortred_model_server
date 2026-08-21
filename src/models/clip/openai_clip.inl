/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: openai_clip.inl
 * Date: 23-6-26
 ************************************************/

#include "openai_clip.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <type_traits>

#include "glog/logging.h"

#include "common/cv_utils.h"
#include "models/cv_image_input.h"

namespace jinq {
namespace models {
namespace clip {

using jinq::models::backend::InferenceSession;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorInfo;
using jinq::common::StatusCode;

namespace {

constexpr float k_embedding_norm_eps = 1.0e-12f;
constexpr float k_clip_logit_cap = 80.0f;

} // namespace

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::on_init(const toml::table& params) {
    _m_context_length = static_cast<int>(params["context_length"].value_or<int64_t>(77));
    _m_truncate_context = params["truncate_context"].value_or<bool>(true);
    if (_m_context_length <= 0) {
        LOG(ERROR) << "openai clip context_length must be positive";
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto tokenizer_status = _m_tokenizer.init(params);
    if (tokenizer_status != StatusCode::OK || !_m_tokenizer.is_successfully_initialized()) {
        LOG(ERROR) << "init openai clip tokenizer failed, status: " << tokenizer_status;
        return tokenizer_status;
    }

    _m_visual_encoder = this->make_session("visual_backend");
    _m_text_encoder = this->make_session("text_backend");
    if (_m_visual_encoder == nullptr || _m_text_encoder == nullptr) {
        _m_visual_encoder.reset();
        _m_text_encoder.reset();
        return StatusCode::MODEL_INIT_FAILED;
    }

    auto status = validate_visual_io(*_m_visual_encoder);
    if (status != StatusCode::OK) {
        _m_visual_encoder.reset();
        _m_text_encoder.reset();
        return status;
    }
    status = validate_text_io(*_m_text_encoder);
    if (status != StatusCode::OK) {
        _m_visual_encoder.reset();
        _m_text_encoder.reset();
        return status;
    }

    const auto* text_input = find_info(*_m_text_encoder, "input");
    if (text_input == nullptr ||
        jinq::models::backend::shape_volume(text_input->shape) !=
            static_cast<int64_t>(_m_context_length)) {
        LOG(ERROR) << "openai clip context length " << _m_context_length
                   << " mismatches text encoder input";
        _m_visual_encoder.reset();
        _m_text_encoder.reset();
        return StatusCode::MODEL_INIT_FAILED;
    }

    LOG(INFO) << "Successfully load openai clip model";
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
const TensorInfo* OpenAiClip<INPUT, OUTPUT>::find_info(
    const InferenceSession& session, const std::string& name) {
    const auto input_iter = std::find_if(
        session.inputs().begin(), session.inputs().end(),
        [&name](const TensorInfo& info) { return info.name == name; });
    if (input_iter != session.inputs().end()) {
        return &*input_iter;
    }
    const auto output_iter = std::find_if(
        session.outputs().begin(), session.outputs().end(),
        [&name](const TensorInfo& info) { return info.name == name; });
    if (output_iter != session.outputs().end()) {
        return &*output_iter;
    }
    return nullptr;
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::validate_visual_io(const InferenceSession& session) {
    const auto* input = find_info(session, "input");
    const auto* output = find_info(session, "output");
    if (input == nullptr || output == nullptr ||
        input->dtype != jinq::models::backend::DType::F32 ||
        output->dtype != jinq::models::backend::DType::F32 ||
        input->shape.size() != 4 || input->shape[1] != 3 ||
        output->shape.size() != 2 || input->dynamic || output->dynamic) {
        LOG(ERROR) << "invalid openai clip visual encoder io";
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::validate_text_io(const InferenceSession& session) {
    const auto* input = find_info(session, "input");
    const auto* output = find_info(session, "output");
    if (input == nullptr || output == nullptr ||
        input->dtype != jinq::models::backend::DType::I32 ||
        output->dtype != jinq::models::backend::DType::F32 ||
        input->shape.size() != 2 || output->shape.size() != 2 ||
        input->dynamic || output->dynamic) {
        LOG(ERROR) << "invalid openai clip text encoder io";
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
const NamedTensor* OpenAiClip<INPUT, OUTPUT>::find_output(
    const std::vector<NamedTensor>& outputs, const std::string& name) const {
    const auto iter = std::find_if(
        outputs.begin(), outputs.end(),
        [&name](const NamedTensor& item) { return item.name == name; });
    return iter == outputs.end() ? nullptr : &*iter;
}

template<typename INPUT, typename OUTPUT>
bool OpenAiClip<INPUT, OUTPUT>::normalize_embedding(std::vector<float>& embedding) {
    const auto squared_norm = std::inner_product(
        embedding.begin(), embedding.end(), embedding.begin(), 0.0f);
    const auto norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm <= k_embedding_norm_eps) {
        return false;
    }
    std::transform(
        embedding.begin(), embedding.end(), embedding.begin(),
        [norm](float value) { return value / norm; });
    return true;
}

template<typename INPUT, typename OUTPUT>
bool OpenAiClip<INPUT, OUTPUT>::embeddings_compatible(
    const std::vector<float>& lhs, const std::vector<float>& rhs) {
    return !lhs.empty() && lhs.size() == rhs.size();
}

template<typename INPUT, typename OUTPUT>
cv::Mat OpenAiClip<INPUT, OUTPUT>::preprocess_image(const cv::Mat& input_image) const {
    if (input_image.empty() || input_image.channels() != 3) {
        LOG(ERROR) << "openai clip visual input must be a non-empty 3-channel image";
        return {};
    }

    const auto& input_info = _m_visual_encoder->inputs().front();
    cv::Mat result;
    cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);
    cv::resize(
        result, result,
        cv::Size(static_cast<int>(input_info.shape[3]),
                 static_cast<int>(input_info.shape[2])));
    result.convertTo(result, CV_32FC3);
    cv::divide(result, 255.0, result);
    cv::subtract(result, cv::Scalar(0.48145466, 0.4578275, 0.40821073), result);
    cv::divide(result, cv::Scalar(0.26862954, 0.26130258, 0.27577711), result);
    return result;
}

template<typename INPUT, typename OUTPUT>
void OpenAiClip<INPUT, OUTPUT>::tokenize(
    const std::string& input_text, std::vector<int32_t>& token_ids) const {
    token_ids.clear();
    std::vector<int32_t> text_tokens;
    const auto tokenize_status = _m_tokenizer.tokenize(input_text, text_tokens);
    if (tokenize_status != StatusCode::OK) {
        LOG(ERROR) << "tokenize openai clip text failed, status: " << tokenize_status;
        return;
    }

    const auto context_length = static_cast<size_t>(_m_context_length);
    if (text_tokens.size() > context_length) {
        const auto last_token = text_tokens.back();
        text_tokens.resize(context_length);
        if (_m_truncate_context) {
            text_tokens.back() = last_token;
        }
    }

    token_ids.assign(context_length, 0);
    std::copy_n(text_tokens.begin(), text_tokens.size(), token_ids.begin());
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::encode_image(
    const cv::Mat& input_image, std::vector<float>& image_embeddings) const {
    image_embeddings.clear();
    const auto preprocessed_image = preprocess_image(input_image);
    if (preprocessed_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(preprocessed_image);
    const auto& input_info = _m_visual_encoder->inputs().front();
    NamedTensor named;
    named.name = input_info.name;
    named.tensor = Tensor::make<float>(input_info.shape);
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "openai clip visual input buffer size mismatch";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());

    std::vector<NamedTensor> inputs;
    inputs.push_back(std::move(named));
    std::vector<NamedTensor> outputs;
    const auto run_status = _m_visual_encoder->run(inputs, outputs);
    if (run_status != StatusCode::OK) {
        return run_status;
    }
    const auto* output = find_output(outputs, "output");
    if (output == nullptr || output->tensor.dtype != jinq::models::backend::DType::F32 ||
        output->tensor.element_count() <= 0) {
        LOG(ERROR) << "openai clip visual encoder output is invalid";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto* data = output->tensor.template data<float>();
    image_embeddings.assign(data, data + output->tensor.element_count());
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::encode_text(
    const std::string& input_text, std::vector<float>& text_embeddings) const {
    text_embeddings.clear();
    std::vector<int32_t> token_ids;
    tokenize(input_text, token_ids);
    if (token_ids.size() < 3) {
        LOG(ERROR) << "openai clip tokenization failed, source text: " << input_text;
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    const auto& input_info = _m_text_encoder->inputs().front();
    NamedTensor named;
    named.name = input_info.name;
    named.tensor = Tensor::make<int32_t>(input_info.shape);
    if (token_ids.size() * sizeof(int32_t) != named.tensor.byte_size()) {
        LOG(ERROR) << "openai clip text input buffer size mismatch";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    std::memcpy(named.tensor.buffer.data(), token_ids.data(), named.tensor.byte_size());

    std::vector<NamedTensor> inputs;
    inputs.push_back(std::move(named));
    std::vector<NamedTensor> outputs;
    const auto run_status = _m_text_encoder->run(inputs, outputs);
    if (run_status != StatusCode::OK) {
        return run_status;
    }
    const auto* output = find_output(outputs, "output");
    if (output == nullptr || output->tensor.dtype != jinq::models::backend::DType::F32 ||
        output->tensor.element_count() <= 0) {
        LOG(ERROR) << "openai clip text encoder output is invalid";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto* data = output->tensor.template data<float>();
    text_embeddings.assign(data, data + output->tensor.element_count());
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::get_visual_embedding(
    const cv::Mat& input_image, std::vector<float>& image_embeddings) {
    if (_m_visual_encoder == nullptr) {
        return StatusCode::MODEL_INIT_FAILED;
    }
    return encode_image(input_image, image_embeddings);
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::get_textual_embedding(
    const std::string& input_text, std::vector<float>& text_embeddings) {
    if (_m_text_encoder == nullptr) {
        return StatusCode::MODEL_INIT_FAILED;
    }
    return encode_text(input_text, text_embeddings);
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::texts2img(
    const std::vector<std::string>& input_texts, const cv::Mat& input_image,
    std::vector<float>& simi_scores) {
    simi_scores.clear();
    if (input_texts.empty()) {
        LOG(ERROR) << "openai clip texts2img input texts are empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    std::vector<float> visual_embedding;
    auto status = get_visual_embedding(input_image, visual_embedding);
    if (status != StatusCode::OK) {
        return status;
    }
    if (!normalize_embedding(visual_embedding)) {
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    std::vector<float> logits;
    logits.reserve(input_texts.size());
    for (const auto& text : input_texts) {
        std::vector<float> text_embedding;
        status = get_textual_embedding(text, text_embedding);
        if (status != StatusCode::OK) {
            return status;
        }
        if (!normalize_embedding(text_embedding) ||
            !embeddings_compatible(text_embedding, visual_embedding)) {
            LOG(ERROR) << "openai clip text/visual embedding shapes are incompatible";
            return StatusCode::MODEL_EMPTY_OUTPUT;
        }
        const auto cosine = std::inner_product(
            text_embedding.begin(), text_embedding.end(), visual_embedding.begin(), 0.0f);
        logits.push_back(std::min(100.0f * cosine, k_clip_logit_cap));
    }

    const auto max_logit = *std::max_element(logits.begin(), logits.end());
    const auto score_sum = std::accumulate(
        logits.begin(), logits.end(), 0.0f, [max_logit](float sum, float value) {
            return sum + std::exp(value - max_logit);
        });
    if (!std::isfinite(score_sum) || score_sum <= 0.0f) {
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    simi_scores.reserve(logits.size());
    for (const auto logit : logits) {
        simi_scores.push_back(std::exp(logit - max_logit) / score_sum);
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::imgs2text(
    const std::vector<cv::Mat>& input_images, const std::string& input_text,
    std::vector<float>& simi_scores) {
    simi_scores.clear();
    if (input_images.empty()) {
        LOG(ERROR) << "openai clip imgs2text input images are empty";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    std::vector<float> text_embedding;
    auto status = get_textual_embedding(input_text, text_embedding);
    if (status != StatusCode::OK) {
        return status;
    }
    if (!normalize_embedding(text_embedding)) {
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    std::vector<float> logits;
    logits.reserve(input_images.size());
    for (const auto& image : input_images) {
        std::vector<float> visual_embedding;
        status = get_visual_embedding(image, visual_embedding);
        if (status != StatusCode::OK) {
            return status;
        }
        if (!normalize_embedding(visual_embedding) ||
            !embeddings_compatible(visual_embedding, text_embedding)) {
            LOG(ERROR) << "openai clip text/visual embedding shapes are incompatible";
            return StatusCode::MODEL_EMPTY_OUTPUT;
        }
        const auto cosine = std::inner_product(
            text_embedding.begin(), text_embedding.end(), visual_embedding.begin(), 0.0f);
        logits.push_back(std::min(100.0f * cosine, k_clip_logit_cap));
    }

    const auto max_logit = *std::max_element(logits.begin(), logits.end());
    const auto score_sum = std::accumulate(
        logits.begin(), logits.end(), 0.0f, [max_logit](float sum, float value) {
            return sum + std::exp(value - max_logit);
        });
    if (!std::isfinite(score_sum) || score_sum <= 0.0f) {
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    simi_scores.reserve(logits.size());
    for (const auto logit : logits) {
        simi_scores.push_back(std::exp(logit - max_logit) / score_sum);
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::run_sessions(const INPUT& input, OUTPUT& output) {
    ClipInput internal_input{};
    if constexpr (std::is_same_v<INPUT, ClipInput>) {
        internal_input = input;
    } else if constexpr (
        std::is_same_v<INPUT, jinq::models::io_define::common_io::mat_input> ||
        std::is_same_v<INPUT, jinq::models::io_define::common_io::file_input> ||
        std::is_same_v<INPUT, jinq::models::io_define::common_io::base64_input>) {
        internal_input.task_type = ClipTaskType::IMAGE_EMBEDDING;
        internal_input.image = jinq::models::cv_input::load_image(input);
    } else {
        LOG(ERROR) << "openai clip input type is unsupported";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    ClipOutput internal_output{};
    auto status = StatusCode::OK;
    switch (internal_input.task_type) {
        case ClipTaskType::TEXT_EMBEDDING:
            status = get_textual_embedding(internal_input.text, internal_output.embeddings);
            break;
        case ClipTaskType::IMAGE_EMBEDDING:
            status = get_visual_embedding(internal_input.image, internal_output.embeddings);
            break;
        case ClipTaskType::TEXTS_TO_IMAGE:
            status = texts2img(
                internal_input.texts, internal_input.image, internal_output.simi_scores);
            break;
        case ClipTaskType::IMAGES_TO_TEXT:
            status = imgs2text(
                internal_input.images, internal_input.text, internal_output.simi_scores);
            break;
        default:
            status = StatusCode::MODEL_RUN_SESSION_FAILED;
            break;
    }
    if (status != StatusCode::OK) {
        return status;
    }
    output = std::move(internal_output);
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
StatusCode OpenAiClip<INPUT, OUTPUT>::postprocess(
    const std::vector<jinq::models::backend::NamedTensor>& outputs, OUTPUT& output) {
    (void)outputs;
    (void)output;
    LOG(ERROR) << "openai clip is a multi-session model and must run through run_sessions";
    return StatusCode::MODEL_RUN_SESSION_FAILED;
}

template<typename INPUT, typename OUTPUT>
OpenAiClip<INPUT, OUTPUT>::OpenAiClip()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("OPENAI_CLIP") {}

} // namespace clip
} // namespace models
} // namespace jinq
