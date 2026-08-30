/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_predictor.inl
 * Date: 23-5-26
 ************************************************/

#include "sam_prompt_decoder.h"
#include "sam_vit_encoder.h"

#include <type_traits>

#include "glog/logging.h"

#include "models/cv_image_input.h"

namespace jinq {
namespace models {
namespace segment_anything {

using jinq::common::StatusCode;
using SamInput = jinq::models::io_define::segment_anything::sam_prompt_input;
using SamOutput = jinq::models::io_define::segment_anything::std_sam_prompt_output;

template <typename INPUT, typename OUTPUT> std::vector<jinq::models::backend::SessionSpec> SamPredictor<INPUT, OUTPUT>::sessions() {
    return {
        // both engines validate their own IO: the encoder names its tensors
        // per backend (input_image/image_embeddings on TRT, others on MNN) and
        // the decoder exposes an optional input (orig_im_size) plus
        // alternative outputs (masks / low_res_masks)
        {"encoder", "encoder_backend", {}, {}},
        {"decoder", "decoder_backend", {}, {}},
    };
}

template <typename INPUT, typename OUTPUT> StatusCode SamPredictor<INPUT, OUTPUT>::on_init(const toml::table &params) {
    (void)params;
    const auto session_status = this->init_sessions();
    if (session_status != StatusCode::OK) {
        return session_status;
    }

    _m_encoder = std::make_unique<SamVitEncoder>(this->session("encoder"));
    _m_decoder = std::make_unique<SamPromptDecoder>(this->session("decoder"));

    auto status = _m_encoder->init();
    if (status != StatusCode::OK) {
        return status;
    }
    const auto shape = _m_encoder->get_encoder_input_shape();
    if (shape.size() != 4) {
        LOG(ERROR) << "invalid sam encoder input shape";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_encoder_input_size.height = shape[2];
    _m_encoder_input_size.width = shape[3];

    status = _m_decoder->init();
    if (status != StatusCode::OK) {
        return status;
    }
    _m_decoder->set_encoder_input_size(_m_encoder_input_size);
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
StatusCode SamPredictor<INPUT, OUTPUT>::predict(const cv::Mat &input_image, const std::vector<cv::Rect> &bboxes,
                                                std::vector<cv::Mat> &predicted_masks) {
    if (_m_encoder == nullptr || _m_decoder == nullptr) {
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    _m_ori_image_size = input_image.size();
    std::vector<float> image_embeddings;
    auto status = _m_encoder->encode(input_image, image_embeddings);
    if (status != StatusCode::OK) {
        return status;
    }

    _m_decoder->set_ori_image_size(_m_ori_image_size);
    return _m_decoder->decode(image_embeddings, transform_bboxes(bboxes, _m_encoder_input_size.height), predicted_masks);
}

template <typename INPUT, typename OUTPUT>
StatusCode SamPredictor<INPUT, OUTPUT>::predict(const cv::Mat &input_image, const std::vector<std::vector<cv::Point2f>> &prompt_points,
                                                std::vector<cv::Mat> &predicted_masks) {
    if (_m_encoder == nullptr || _m_decoder == nullptr) {
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    _m_ori_image_size = input_image.size();
    std::vector<float> image_embeddings;
    auto status = _m_encoder->encode(input_image, image_embeddings);
    if (status != StatusCode::OK) {
        return status;
    }

    _m_decoder->set_ori_image_size(_m_ori_image_size);
    return _m_decoder->decode(image_embeddings, transform_points(prompt_points, _m_encoder_input_size.height), predicted_masks);
}

template <typename INPUT, typename OUTPUT>
StatusCode SamPredictor<INPUT, OUTPUT>::get_embedding(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    if (_m_encoder == nullptr) {
        return StatusCode::MODEL_INIT_FAILED;
    }
    return _m_encoder->encode(input_image, image_embeddings);
}

template <typename INPUT, typename OUTPUT>
std::vector<cv::Rect2f> SamPredictor<INPUT, OUTPUT>::transform_bboxes(const std::vector<cv::Rect> &bboxes, int target_size) const {
    const auto long_side = std::max(_m_ori_image_size.height, _m_ori_image_size.width);
    const auto scale = static_cast<float>(target_size) / static_cast<float>(long_side);
    std::vector<cv::Rect2f> result;
    result.reserve(bboxes.size());
    for (const auto &bbox : bboxes) {
        result.emplace_back(bbox.x * scale, bbox.y * scale, bbox.width * scale, bbox.height * scale);
    }
    return result;
}

template <typename INPUT, typename OUTPUT>
std::vector<std::vector<cv::Point2f>> SamPredictor<INPUT, OUTPUT>::transform_points(const std::vector<std::vector<cv::Point2f>> &points,
                                                                                    int target_size) const {
    const auto long_side = std::max(_m_ori_image_size.height, _m_ori_image_size.width);
    const auto scale = static_cast<float>(target_size) / static_cast<float>(long_side);
    std::vector<std::vector<cv::Point2f>> result;
    result.reserve(points.size());
    for (const auto &object_points : points) {
        std::vector<cv::Point2f> transformed_points;
        transformed_points.reserve(object_points.size());
        for (const auto &point : object_points) {
            transformed_points.emplace_back(point.x * scale, point.y * scale);
        }
        result.push_back(std::move(transformed_points));
    }
    return result;
}

template <typename INPUT, typename OUTPUT> StatusCode SamPredictor<INPUT, OUTPUT>::run_sessions(const INPUT &input, OUTPUT &output) {
    SamInput internal_input{};
    if constexpr (std::is_same_v<INPUT, SamInput>) {
        internal_input = input;
    } else if constexpr (std::is_same_v<INPUT, jinq::models::io_define::common_io::mat_input> ||
                         std::is_same_v<INPUT, jinq::models::io_define::common_io::file_input> ||
                         std::is_same_v<INPUT, jinq::models::io_define::common_io::base64_input>) {
        StatusCode image_status = StatusCode::OK;
        std::string image_error;
        internal_input.image = this->load_model_image(input, &image_status, &image_error);
        if (internal_input.image.empty()) {
            LOG(ERROR) << image_error;
            return image_status == StatusCode::OK ? StatusCode::MODEL_EMPTY_INPUT_IMAGE : image_status;
        }
    } else {
        LOG(ERROR) << "sam predictor input type is unsupported";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    if (internal_input.image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    SamOutput internal_output{};
    StatusCode status = StatusCode::OK;
    if (!internal_input.bboxes.empty()) {
        status = predict(internal_input.image, internal_input.bboxes, internal_output);
    } else if (!internal_input.prompt_points.empty()) {
        status = predict(internal_input.image, internal_input.prompt_points, internal_output);
    }
    if (status != StatusCode::OK) {
        return status;
    }
    output = std::move(internal_output);
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
StatusCode SamPredictor<INPUT, OUTPUT>::postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                                                    const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) {
    (void)outputs;
    (void)output;
    LOG(ERROR) << "sam predictor is a multi-session model and must run through run_sessions";
    return StatusCode::MODEL_RUN_SESSION_FAILED;
}

template <typename INPUT, typename OUTPUT>
SamPredictor<INPUT, OUTPUT>::SamPredictor()
    : jinq::models::backend::MultiSessionModel<SamPredictor<INPUT, OUTPUT>, INPUT, OUTPUT>("SAM_PREDICTOR") {}

template <typename INPUT, typename OUTPUT> SamPredictor<INPUT, OUTPUT>::~SamPredictor() = default;

} // namespace segment_anything
} // namespace models
} // namespace jinq
