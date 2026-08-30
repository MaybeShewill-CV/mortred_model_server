/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.inl
 * Date: 22-6-10
 ************************************************/

#include "libface_detector.h"

#include <algorithm>
#include <cmath>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "models/backend/model_runtime.h"
#include "models/object_detection/detector_common.h"

namespace jinq {
namespace models {
namespace object_detection {

using FaceBBox = jinq::models::io_define::object_detection::face_bbox;
using FaceOutput = jinq::models::io_define::object_detection::std_face_detection_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode LibFaceDetector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.6f;
    _m_detection_params.nms_threshold = 0.3f;
    _m_detection_params.keep_top_k = 250;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid libface detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected libface input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    cv::Size configured_size;
    if (!parse_model_input_size(params, &configured_size, &param_error) ||
        (params.contains("model_input_image_size") && configured_size != _m_input_size_host)) {
        LOG(ERROR) << "invalid libface input size: " << (param_error.empty() ? "configured size mismatches model input" : param_error);
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> auto LibFaceDetector<INPUT, OUTPUT>::generate_prior_anchors() const -> std::vector<FaceAnchor> {
    const std::vector<std::vector<double>> min_sizes = {{10., 16., 24.}, {32., 48.}, {64., 96.}, {128., 192., 256.}};
    const std::vector<double> steps = {8., 16., 32., 64.};

    const auto in_h = _m_input_size_host.height;
    const auto in_w = _m_input_size_host.width;
    const std::vector<int> feature_map_2th = {static_cast<int>((in_h + 1) / 2 / 2), static_cast<int>((in_w + 1) / 2 / 2)};
    const std::vector<int> feature_map_3th = {feature_map_2th[0] / 2, feature_map_2th[1] / 2};
    const std::vector<int> feature_map_4th = {feature_map_3th[0] / 2, feature_map_3th[1] / 2};
    const std::vector<int> feature_map_5th = {feature_map_4th[0] / 2, feature_map_4th[1] / 2};
    const std::vector<int> feature_map_6th = {feature_map_5th[0] / 2, feature_map_5th[1] / 2};
    const std::vector<std::vector<int>> feature_maps = {feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th};

    std::vector<FaceAnchor> anchors;
    for (size_t k = 0; k < feature_maps.size(); ++k) {
        const auto &feature_map = feature_maps[k];
        const auto &feature_min_sizes = min_sizes[k];
        for (int i = 0; i < feature_map[0]; ++i) {
            for (int j = 0; j < feature_map[1]; ++j) {
                for (const auto min_size : feature_min_sizes) {
                    FaceAnchor anchor;
                    anchor.s_kx = min_size / in_w;
                    anchor.s_ky = min_size / in_h;
                    anchor.cx = (static_cast<double>(j) + 0.5) * steps[k] / in_w;
                    anchor.cy = (static_cast<double>(i) + 0.5) * steps[k] / in_h;
                    anchors.push_back(anchor);
                }
            }
        }
    }
    return anchors;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> LibFaceDetector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // resize / colour / normalize, emitted as f32 nchw
    auto result =
        jinq::models::backend::ImagePipeline(input_image).resize(_m_input_size_host).to_float().nchw(this->session().inputs().front().name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode LibFaceDetector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                       const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    const auto *loc = jinq::models::backend::find_output(outputs, "loc");
    const auto *conf = jinq::models::backend::find_output(outputs, "conf");
    if (loc == nullptr || conf == nullptr) {
        LOG(ERROR) << "libface outputs loc/conf missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    std::string contract_error;
    if (!jinq::models::backend::validate_output_tensor(*loc, {jinq::models::backend::DType::F32, 3, {1, -1, 14}}, &contract_error) ||
        !jinq::models::backend::validate_output_tensor(*conf, {jinq::models::backend::DType::F32, 3, {1, loc->tensor.shape[1], 2}},
                                                       &contract_error)) {
        LOG(ERROR) << "libface output contract failed: " << contract_error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }

    const auto anchor_count = static_cast<size_t>(loc->tensor.shape[1]);
    const float *loc_data = nullptr;
    const float *conf_data = nullptr;
    if (!jinq::models::backend::get_f32_data(loc->tensor, &loc_data, &contract_error) ||
        !jinq::models::backend::get_f32_data(conf->tensor, &conf_data, &contract_error) ||
        !jinq::models::backend::require_finite_f32(loc_data, loc->tensor.element_count(), loc->name, &contract_error) ||
        !jinq::models::backend::require_finite_f32(conf_data, conf->tensor.element_count(), conf->name, &contract_error)) {
        LOG(ERROR) << "libface output contract failed: " << contract_error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }

    // MNN rank-3 host output is row-major [N, anchors, channels].  A real
    // 640x480-weight probe found 37 threshold hits with this interleaved view
    // versus 8776 with the old channel-major view; only the former matches the
    // five-box golden result after anchor decoding and NMS.

    const auto priors = generate_prior_anchors();
    if (priors.size() != anchor_count) {
        LOG(ERROR) << "libface anchor count " << priors.size() << " mismatches output anchor count " << anchor_count;
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    DetectionGeometryScale geometry_scale;
    std::string geometry_error;
    if (!make_detection_geometry_scale(context, &geometry_scale, &geometry_error)) {
        LOG(ERROR) << "libface " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    std::vector<FaceBBox> decode_result;
    for (size_t bbox_index = 0; bbox_index < anchor_count; ++bbox_index) {
        const auto &prior = priors[bbox_index];
        const auto raw_conf = conf_data[bbox_index * 2 + 1];
        if (raw_conf <= _m_detection_params.score_threshold) {
            continue;
        }

        const auto raw_bbox_x = loc_data[bbox_index * 14];
        const auto raw_bbox_y = loc_data[bbox_index * 14 + 1];
        const auto raw_bbox_w = loc_data[bbox_index * 14 + 2];
        const auto raw_bbox_h = loc_data[bbox_index * 14 + 3];
        auto pred_bbox_x = prior.cx + raw_bbox_x * 0.1 * prior.s_kx;
        auto pred_bbox_y = prior.cy + raw_bbox_y * 0.1 * prior.s_ky;
        auto pred_bbox_w = prior.s_kx * std::exp(raw_bbox_w * 0.2);
        auto pred_bbox_h = prior.s_ky * std::exp(raw_bbox_h * 0.2);
        pred_bbox_x = (pred_bbox_x - pred_bbox_w / 2.0) * context.network_size.width;
        pred_bbox_y = (pred_bbox_y - pred_bbox_h / 2.0) * context.network_size.height;
        pred_bbox_w *= context.network_size.width;
        pred_bbox_h *= context.network_size.height;

        FaceBBox face_box;
        face_box.score = raw_conf;
        face_box.bbox = cv::Rect2f(static_cast<float>(pred_bbox_x), static_cast<float>(pred_bbox_y), static_cast<float>(pred_bbox_w),
                                   static_cast<float>(pred_bbox_h));
        for (size_t landmark_index = 4; landmark_index < 14; landmark_index += 2) {
            const auto raw_landmark_x = loc_data[bbox_index * 14 + landmark_index];
            const auto raw_landmark_y = loc_data[bbox_index * 14 + landmark_index + 1];
            const auto pred_landmark_x = (prior.cx + raw_landmark_x * 0.1 * prior.s_kx) * context.network_size.width;
            const auto pred_landmark_y = (prior.cy + raw_landmark_y * 0.1 * prior.s_ky) * context.network_size.height;
            face_box.landmarks.emplace_back(static_cast<float>(pred_landmark_x), static_cast<float>(pred_landmark_y));
        }
        face_box.class_id = 0;
        decode_result.push_back(std::move(face_box));
    }

    auto nms_result = finalize_detections(std::move(decode_result), _m_detection_params);
    for (auto &face_box : nms_result) {
        face_box.bbox = scale_detection_bbox(face_box.bbox, geometry_scale);
        for (auto &landmark : face_box.landmarks) {
            landmark = scale_detection_point(landmark, geometry_scale);
        }
        face_box.category = "face";
    }

    FaceOutput faces;
    faces.reserve(nms_result.size());
    for (const auto &face_box : nms_result) {
        faces.push_back(face_box);
    }
    output = std::move(faces);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
LibFaceDetector<INPUT, OUTPUT>::LibFaceDetector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("LIBFACE") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
