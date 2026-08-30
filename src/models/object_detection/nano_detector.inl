/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: nano_detector.inl
 * Date: 22-6-10
 ************************************************/

#include "nano_detector.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "glog/logging.h"
#include "models/backend/model_runtime.h"
#include "models/object_detection/detector_common.h"

namespace jinq {
namespace models {
namespace object_detection {

using DetectionOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode NanoDetector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.4f;
    _m_detection_params.nms_threshold = 0.35f;
    _m_detection_params.keep_top_k = 250;
    _m_detection_params.class_nums = 80;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid nanodet detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected nanodet input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "nanodet input shape has dynamic/invalid H/W: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }

    generate_grid_center_priors();
    cv::Size configured_size;
    if (!parse_model_input_size(params, &configured_size, &param_error) ||
        (params.contains("model_input_image_size") && configured_size != _m_input_size_host)) {
        LOG(ERROR) << "invalid nanodet input size: " << (param_error.empty() ? "configured size mismatches model input" : param_error);
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> NanoDetector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // resize / colour / normalize, emitted as f32 nchw
    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .resize(_m_input_size_host)
                      .to_float()
                      .scale(1.0f / 255.0f)
                      .subtract({0.406f, 0.456f, 0.485f})
                      .divide({0.225f, 0.224f, 0.229f})
                      .nchw(this->session().inputs().front().name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode NanoDetector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                    const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    const int num_points = static_cast<int>(_m_center_priors.size());
    const int num_channels = _m_detection_params.class_nums + (_m_reg_max + 1) * 4;
    F32OutputView output_view;
    const auto output_status = validated_f32_output(
        outputs, "output", {jinq::models::backend::DType::F32, 3, {1, num_points, num_channels}}, "nanodet", &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const float *tensor_preds_host = output_view.data;

    DetectionGeometryScale geometry_scale;
    std::string geometry_error;
    if (!make_detection_geometry_scale(context, &geometry_scale, &geometry_error)) {
        LOG(ERROR) << "nanodet " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    DetectionOutput result;

    for (int idx = 0; idx < num_points; idx++) {
        const int ct_x = _m_center_priors[idx].x;
        const int ct_y = _m_center_priors[idx].y;
        const int stride = _m_center_priors[idx].stride;

        const float *scores = tensor_preds_host + (idx * num_channels);
        const auto max_score_iter = std::max_element(scores, scores + _m_detection_params.class_nums);
        const float score = *max_score_iter;
        const int cur_label = static_cast<int>(std::distance(scores, max_score_iter));

        if (score > _m_detection_params.score_threshold) {
            const float *bbox_pred = tensor_preds_host + idx * num_channels + _m_detection_params.class_nums;
            const auto obj_box_coords = refine_bbox_coords(bbox_pred, ct_x, ct_y, stride, context, geometry_scale);
            jinq::models::io_define::object_detection::bbox obj_box;
            obj_box.score = score;
            obj_box.class_id = cur_label;
            obj_box.bbox = cv::Rect2f(obj_box_coords[0], obj_box_coords[1], obj_box_coords[2], obj_box_coords[3]);
            result.push_back(obj_box);
        }
    }

    DetectionOutput nms_result = finalize_detections(std::move(result), _m_detection_params);
    output = std::move(nms_result);
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
std::vector<float> NanoDetector<INPUT, OUTPUT>::refine_bbox_coords(const float *preds, int x, int y, int stride,
                                                                   const jinq::models::backend::InferenceContext &context,
                                                                   const DetectionGeometryScale &geometry_scale) const {
    const auto ct_x = static_cast<float>(x * stride);
    const auto ct_y = static_cast<float>(y * stride);
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    std::vector<float> dis_after_sm(_m_reg_max + 1);

    for (int i = 0; i < 4; i++) {
        float dis = 0;
        activation_function_softmax(preds + i * (_m_reg_max + 1), dis_after_sm.data(), _m_reg_max + 1);

        for (int j = 0; j < _m_reg_max + 1; j++) {
            dis += static_cast<float>(j) * dis_after_sm[j];
        }

        dis *= static_cast<float>(stride);
        dis_pred[i] = dis;
    }

    const auto bbox = scale_detection_bbox(
        {std::max(ct_x - dis_pred[0], 0.0f), std::max(ct_y - dis_pred[1], 0.0f),
         std::min(ct_x + dis_pred[2], static_cast<float>(context.network_size.width)) - std::max(ct_x - dis_pred[0], 0.0f),
         std::min(ct_y + dis_pred[3], static_cast<float>(context.network_size.height)) - std::max(ct_y - dis_pred[1], 0.0f)},
        geometry_scale);
    return {bbox.x, bbox.y, bbox.width, bbox.height};
}

template <typename INPUT, typename OUTPUT> void NanoDetector<INPUT, OUTPUT>::generate_grid_center_priors() {
    for (const auto &stride : _m_strides) {
        const int feat_w = std::ceil(static_cast<float>(_m_input_size_host.width) / static_cast<float>(stride));
        const int feat_h = std::ceil(static_cast<float>(_m_input_size_host.height) / static_cast<float>(stride));

        for (int y = 0; y < feat_h; y++) {
            for (int x = 0; x < feat_w; x++) {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                _m_center_priors.push_back(ct);
            }
        }
    }
}

template <typename INPUT, typename OUTPUT> float NanoDetector<INPUT, OUTPUT>::fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

template <typename INPUT, typename OUTPUT>
void NanoDetector<INPUT, OUTPUT>::activation_function_softmax(const float *src, float *dst, int length) {
    const float alpha = *std::max_element(src, src + length);
    float denominator{0};

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
NanoDetector<INPUT, OUTPUT>::NanoDetector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("NanoDet") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
