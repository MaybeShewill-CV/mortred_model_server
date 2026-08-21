/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_automask_generator.inl
 * Date: 23-10-13
 ************************************************/

#include "sam_amg_decoder.h"
#include "models/cv_image_input.h"
#include "models/segment_anything/sam_prediction/sam_vit_encoder.h"

#include <type_traits>

#include "glog/logging.h"

namespace jinq {
namespace models {
namespace segment_anything {

using jinq::common::StatusCode;
using AmgOutput = jinq::models::io_define::segment_anything::sam_amg_output;

template<typename INPUT, typename OUTPUT>
StatusCode SamAutoMaskGenerator<INPUT, OUTPUT>::on_init(const toml::table& params) {
    _m_points_per_side = static_cast<int>(params["points_per_side"].value_or<int64_t>(32));
    _m_pred_iou_thresh = static_cast<float>(params["pred_iou_thresh"].value_or<double>(0.88));
    _m_stability_score_thresh =
        static_cast<float>(params["stability_score_thresh"].value_or<double>(0.95));
    _m_box_nms_thresh = static_cast<float>(params["box_nms_thresh"].value_or<double>(0.7));
    _m_min_mask_region_area =
        static_cast<int>(params["min_mask_region_area"].value_or<int64_t>(0));
    if (_m_points_per_side <= 0) {
        LOG(ERROR) << "sam amg points_per_side must be positive";
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_encoder = std::make_unique<SamVitEncoder>(
        this->make_session("encoder_backend"));
    auto status = _m_encoder->init();
    if (status != StatusCode::OK) {
        _m_encoder.reset();
        _m_decoder.reset();
        return status;
    }
    const auto input_shape = _m_encoder->get_encoder_input_shape();
    if (input_shape.size() != 4) {
        LOG(ERROR) << "invalid sam amg encoder input shape";
        _m_encoder.reset();
        _m_decoder.reset();
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_encoder_input_size.height = input_shape[2];
    _m_encoder_input_size.width = input_shape[3];

    _m_decoder = std::make_unique<SamAmgDecoder>();
    status = _m_decoder->init(this->model_section());
    if (status != StatusCode::OK) {
        _m_encoder.reset();
        _m_decoder.reset();
        return status;
    }
    _m_decoder->set_encoder_input_size(_m_encoder_input_size);
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
StatusCode SamAutoMaskGenerator<INPUT, OUTPUT>::generate(
    const cv::Mat& input_image, AmgOutput& amg_output) {
    if (_m_encoder == nullptr || _m_decoder == nullptr) {
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    std::vector<float> image_embeddings;
    auto status = _m_encoder->encode(input_image, image_embeddings);
    if (status != StatusCode::OK) {
        return status;
    }
    _m_decoder->set_ori_image_size(input_image.size());
    return _m_decoder->decode_everything(
        image_embeddings, amg_output, _m_points_per_side, _m_pred_iou_thresh,
        _m_stability_score_thresh, _m_box_nms_thresh, _m_min_mask_region_area);
}

template<typename INPUT, typename OUTPUT>
StatusCode SamAutoMaskGenerator<INPUT, OUTPUT>::run_sessions(const INPUT& input, OUTPUT& output) {
    cv::Mat image;
    if constexpr (std::is_same_v<INPUT, cv::Mat>) {
        image = input;
    } else if constexpr (
        std::is_same_v<INPUT, jinq::models::io_define::common_io::mat_input> ||
        std::is_same_v<INPUT, jinq::models::io_define::common_io::file_input> ||
        std::is_same_v<INPUT, jinq::models::io_define::common_io::base64_input>) {
        image = jinq::models::cv_input::load_image(input);
    } else {
        LOG(ERROR) << "sam amg input type is unsupported";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    AmgOutput internal_output{};
    const auto status = generate(image, internal_output);
    if (status != StatusCode::OK) {
        return status;
    }
    output = std::move(internal_output);
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
StatusCode SamAutoMaskGenerator<INPUT, OUTPUT>::postprocess(
    const std::vector<jinq::models::backend::NamedTensor>& outputs, OUTPUT& output) {
    (void)outputs;
    (void)output;
    LOG(ERROR) << "sam amg is a multi-session model and must run through run_sessions";
    return StatusCode::MODEL_RUN_SESSION_FAILED;
}

template<typename INPUT, typename OUTPUT>
SamAutoMaskGenerator<INPUT, OUTPUT>::SamAutoMaskGenerator()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("SAM_AMG") {}

template<typename INPUT, typename OUTPUT>
SamAutoMaskGenerator<INPUT, OUTPUT>::~SamAutoMaskGenerator() = default;

} // namespace segment_anything
} // namespace models
} // namespace jinq
