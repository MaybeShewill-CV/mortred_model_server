/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_vit_encoder.cpp
 * Date: 23-6-7
 ************************************************/

#include "sam_vit_encoder.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"

#include "common/cv_utils.h"
#include "models/backend/tensor.h"

namespace jinq {
namespace models {
namespace segment_anything {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::models::backend::InferenceSession;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;

class SamVitEncoder::Impl {
  public:
    explicit Impl(InferenceSession *session) : _m_session(session) {}

    ~Impl() = default;

    StatusCode init() {
        if (_m_session == nullptr) {
            LOG(ERROR) << "sam encoder session is null";
            return StatusCode::MODEL_INIT_FAILED;
        }
        if (_m_session->inputs().size() != 1u || _m_session->outputs().size() != 1u) {
            LOG(ERROR) << "sam encoder must have one input and one output";
            return StatusCode::MODEL_INIT_FAILED;
        }

        const auto &input = _m_session->inputs().front();
        const auto &output = _m_session->outputs().front();
        if (input.dtype != jinq::models::backend::DType::F32 || output.dtype != jinq::models::backend::DType::F32 ||
            input.shape.size() != 4 || input.shape[1] != 3 || output.shape.size() != 4 || input.dynamic || output.dynamic) {
            LOG(ERROR) << "invalid sam encoder io: " << input.to_string() << " / " << output.to_string();
            return StatusCode::MODEL_INIT_FAILED;
        }

        _m_input_shape.reserve(input.shape.size());
        for (const auto dim : input.shape) {
            _m_input_shape.push_back(static_cast<int>(dim));
        }
        _m_input_size_host.height = static_cast<int>(input.shape[2]);
        _m_input_size_host.width = static_cast<int>(input.shape[3]);
        _m_successfully_initialized = true;
        return StatusCode::OK;
    }

    StatusCode encode(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
        if (!_m_successfully_initialized) {
            return StatusCode::MODEL_INIT_FAILED;
        }
        const auto preprocessed_image = preprocess_image(input_image);
        if (preprocessed_image.empty()) {
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }

        const auto chw_data = CvUtils::convert_to_chw_vec(preprocessed_image);
        NamedTensor named;
        named.name = _m_session->inputs().front().name;
        named.tensor = Tensor::make<float>(_m_session->inputs().front().shape);
        if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
            LOG(ERROR) << "sam encoder input buffer size mismatch";
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }
        std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());

        std::vector<NamedTensor> inputs;
        inputs.push_back(std::move(named));
        std::vector<NamedTensor> outputs;
        const auto run_status = _m_session->run(inputs, outputs);
        if (run_status != StatusCode::OK) {
            return run_status;
        }
        if (outputs.empty() || outputs.front().tensor.dtype != jinq::models::backend::DType::F32 ||
            outputs.front().tensor.element_count() <= 0) {
            LOG(ERROR) << "sam encoder output is invalid";
            return StatusCode::MODEL_EMPTY_OUTPUT;
        }

        const auto *data = outputs.front().tensor.template data<float>();
        image_embeddings.assign(data, data + outputs.front().tensor.element_count());
        return StatusCode::OK;
    }

    std::vector<int> get_encoder_input_shape() const { return _m_input_shape; }

    bool is_successfully_initialized() const { return _m_successfully_initialized; }

  private:
    cv::Mat preprocess_image(const cv::Mat &input_image) const {
        if (input_image.empty() || input_image.channels() != 3) {
            LOG(ERROR) << "sam encoder input must be a non-empty 3-channel image";
            return {};
        }

        const auto input_height = static_cast<float>(input_image.rows);
        const auto input_width = static_cast<float>(input_image.cols);
        const auto long_side = std::max(input_image.rows, input_image.cols);
        const auto scale = static_cast<float>(_m_input_size_host.height) / static_cast<float>(long_side);
        const auto target_width = static_cast<int>(std::round(scale * input_width));
        const auto target_height = static_cast<int>(std::round(scale * input_height));
        if (target_width <= 0 || target_height <= 0) {
            LOG(ERROR) << "sam encoder resized image is empty";
            return {};
        }

        cv::Mat result;
        cv::resize(input_image, result, cv::Size(target_width, target_height));
        result.convertTo(result, CV_32FC3);
        cv::subtract(result, cv::Scalar(123.675, 116.28, 103.53), result);
        cv::divide(result, cv::Scalar(58.395, 57.12, 57.375), result);
        cv::copyMakeBorder(result, result, 0, _m_input_size_host.height - target_height, 0, _m_input_size_host.width - target_width,
                           cv::BORDER_CONSTANT, 0.0);
        return result;
    }

    InferenceSession *_m_session = nullptr;
    std::vector<int> _m_input_shape;
    cv::Size _m_input_size_host;
    bool _m_successfully_initialized = false;
};

SamVitEncoder::SamVitEncoder(backend::InferenceSession *session) : _m_pimpl(std::make_unique<Impl>(session)) {}

SamVitEncoder::~SamVitEncoder() = default;

StatusCode SamVitEncoder::init() { return _m_pimpl->init(); }

StatusCode SamVitEncoder::encode(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    return _m_pimpl->encode(input_image, image_embeddings);
}

std::vector<int> SamVitEncoder::get_encoder_input_shape() const { return _m_pimpl->get_encoder_input_shape(); }

bool SamVitEncoder::is_successfully_initialized() const { return _m_pimpl->is_successfully_initialized(); }

} // namespace segment_anything
} // namespace models
} // namespace jinq
