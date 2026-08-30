/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: centerface_detector.inl
 * Date: 23-10-18
 ************************************************/

#include "centerface_detector.h"

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

template <typename INPUT, typename OUTPUT> StatusCode CenterFaceDetector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.6f;
    _m_detection_params.nms_threshold = 0.3f;
    _m_detection_params.keep_top_k = 250;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid centerface detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (params.contains("model_input_image_size")) {
        LOG(ERROR) << "centerface uses dynamic input sizes; model_input_image_size is invalid";
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected centerface input shape: " << input_info.to_string() << ", expected dynamic [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
std::vector<NamedTensor> CenterFaceDetector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // bgr -> rgb, dynamic resize to a multiple of 32 (the session resizes itself)
    const auto width_resized = static_cast<int>(std::ceil(static_cast<float>(input_image.cols) / 32.0f) * 32);
    const auto height_resized = static_cast<int>(std::ceil(static_cast<float>(input_image.rows) / 32.0f) * 32);

    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .bgr_to_rgb()
                      .resize(cv::Size(width_resized, height_resized))
                      .to_float()
                      .nchw(this->session().inputs().front().name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode CenterFaceDetector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                          const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    const auto *heatmap = jinq::models::backend::find_output(outputs, "537");
    const auto *scale = jinq::models::backend::find_output(outputs, "538");
    const auto *offset = jinq::models::backend::find_output(outputs, "539");
    const auto *landmark = jinq::models::backend::find_output(outputs, "540");
    if (heatmap == nullptr || scale == nullptr || offset == nullptr || landmark == nullptr) {
        LOG(ERROR) << "centerface outputs 537/538/539/540 missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    DetectionGeometryScale geometry_scale;
    std::string geometry_error;
    if (!make_detection_geometry_scale(context, &geometry_scale, &geometry_error)) {
        LOG(ERROR) << "centerface " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // heatmap layout: [1,1,H,W] over the /4 feature map
    const jinq::models::backend::Tensor &heat_tensor = heatmap->tensor;
    std::string contract_error;
    if (!jinq::models::backend::validate_output_tensor(*heatmap, {jinq::models::backend::DType::F32, 4, {1, 1, -1, -1}}, &contract_error)) {
        LOG(ERROR) << "centerface heatmap contract failed: " << contract_error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }
    const int output_height = static_cast<int>(heat_tensor.shape[2]);
    const int output_width = static_cast<int>(heat_tensor.shape[3]);
    const int channel_step = output_width * output_height;
    const std::vector<std::pair<const jinq::models::backend::NamedTensor *, int>> contracted = {{scale, 2}, {offset, 2}, {landmark, 10}};
    for (const auto &item : contracted) {
        if (!jinq::models::backend::validate_output_tensor(
                *item.first, {jinq::models::backend::DType::F32, 4, {1, item.second, output_height, output_width}}, &contract_error)) {
            LOG(ERROR) << "centerface output contract failed: " << contract_error;
            return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
        }
    }
    const float *heat_data = nullptr;
    const float *scale_data = nullptr;
    const float *offset_data = nullptr;
    const float *landmark_data = nullptr;
    if (!jinq::models::backend::get_f32_data(heat_tensor, &heat_data, &contract_error) ||
        !jinq::models::backend::get_f32_data(scale->tensor, &scale_data, &contract_error) ||
        !jinq::models::backend::get_f32_data(offset->tensor, &offset_data, &contract_error) ||
        !jinq::models::backend::get_f32_data(landmark->tensor, &landmark_data, &contract_error) ||
        !jinq::models::backend::require_finite_f32(heat_data, heat_tensor.element_count(), heatmap->name, &contract_error) ||
        !jinq::models::backend::require_finite_f32(scale_data, scale->tensor.element_count(), scale->name, &contract_error) ||
        !jinq::models::backend::require_finite_f32(offset_data, offset->tensor.element_count(), offset->name, &contract_error) ||
        !jinq::models::backend::require_finite_f32(landmark_data, landmark->tensor.element_count(), landmark->name, &contract_error)) {
        LOG(ERROR) << "centerface output contract failed: " << contract_error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }

    std::vector<FaceBBox> decode_result;
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            const int index = h * output_width + w;
            const float score = heat_data[index];
            if (score < _m_detection_params.score_threshold) {
                continue;
            }
            const float s0 = 4 * std::exp(scale_data[index]);
            const float s1 = 4 * std::exp(scale_data[index + channel_step]);
            const float o0 = offset_data[index];
            const float o1 = offset_data[index + channel_step];

            const float ymin = std::max(0.0f, static_cast<float>(4 * (h + o0 + 0.5) - 0.5 * s0));
            const float xmin = std::max(0.0f, static_cast<float>(4 * (w + o1 + 0.5) - 0.5 * s1));
            const float ymax = std::min(ymin + s0, static_cast<float>(context.network_size.height));
            const float xmax = std::min(xmin + s1, static_cast<float>(context.network_size.width));

            FaceBBox face_info;
            face_info.score = score;
            face_info.bbox.x = xmin;
            face_info.bbox.y = ymin;
            face_info.bbox.width = (xmax - xmin);
            face_info.bbox.height = (ymax - ymin);
            for (int num = 0; num < 5; ++num) {
                cv::Point2f point;
                point.x = s1 * landmark_data[(2 * num + 1) * channel_step + index] + xmin;
                point.y = s0 * landmark_data[(2 * num + 0) * channel_step + index] + ymin;
                face_info.landmarks.push_back(point);
            }
            face_info.class_id = 0;
            decode_result.push_back(face_info);
        }
    }

    // refine bbox coords back into the user image space
    for (auto &face_box : decode_result) {
        face_box.bbox = scale_detection_bbox(face_box.bbox, geometry_scale);
        for (auto &point : face_box.landmarks) {
            point = scale_detection_point(point, geometry_scale);
        }
    }

    auto nms_result = finalize_detections(std::move(decode_result), _m_detection_params);
    for (auto &bbox : nms_result) {
        bbox.category = "face";
    }
    FaceOutput faces;
    faces.reserve(nms_result.size());
    for (const auto &bbox : nms_result) {
        faces.push_back(bbox);
    }
    output = std::move(faces);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
CenterFaceDetector<INPUT, OUTPUT>::CenterFaceDetector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("CENTER_FACE") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
