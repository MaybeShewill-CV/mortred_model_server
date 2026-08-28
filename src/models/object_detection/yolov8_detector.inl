/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_detector.inl
 * Date: 24-3-13
 ************************************************/

#include "yolov8_detector.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace object_detection {

using DetectionOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode YoloV8Detector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.4f;
    _m_detection_params.nms_threshold = 0.35f;
    _m_detection_params.keep_top_k = 250;
    _m_detection_params.class_nums = 80;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid yolov8 detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected yolov8 input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "yolov8 input shape has dynamic/invalid H/W: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    cv::Size configured_size;
    if (!parse_model_input_size(params, &configured_size, &param_error) ||
        (params.contains("model_input_image_size") && configured_size != _m_input_size_host)) {
        LOG(ERROR) << "invalid yolov8 input size: " << (param_error.empty() ? "configured size mismatches model input" : param_error);
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> YoloV8Detector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // bgr -> rgb -> resize -> [0,1] normalize, emitted as f32 nchw
    cv::Mat tmp;
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);
    cv::resize(tmp, tmp, _m_input_size_host);
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    tmp /= 255.0;

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>({1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data size mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
cv::Rect2f YoloV8Detector<INPUT, OUTPUT>::transform_bboxes(const cv::Rect2d &bbox,
                                                           const jinq::models::backend::InferenceContext &context) const {
    const auto w_scale = static_cast<float>(context.source_size.width) / static_cast<float>(context.network_size.width);
    const auto h_scale = static_cast<float>(context.source_size.height) / static_cast<float>(context.network_size.height);
    cv::Rect2f result;
    result.x = static_cast<float>(bbox.x * w_scale);
    result.y = static_cast<float>(bbox.y * h_scale);
    result.width = static_cast<float>(bbox.width * w_scale);
    result.height = static_cast<float>(bbox.height * h_scale);
    return result;
}

template <typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                      const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "yolov8 output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto &tensor = outputs.front().tensor;
    std::string contract_error;
    if (!jinq::models::backend::validate_output_tensor(
            outputs.front(), {jinq::models::backend::DType::F32, 3, {1, _m_detection_params.class_nums + 4, -1}}, &contract_error)) {
        LOG(ERROR) << "yolov8 output contract failed: " << contract_error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }
    const float *out_data = nullptr;
    if (!jinq::models::backend::get_f32_data(tensor, &out_data, &contract_error) ||
        !jinq::models::backend::require_finite_f32(out_data, static_cast<size_t>(tensor.element_count()), outputs.front().name,
                                                   &contract_error)) {
        LOG(ERROR) << "yolov8 output contract failed: " << contract_error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }
    const auto row_size = tensor.shape[1];
    const auto proposal_counts = tensor.shape[2];

    // collect class-filtered candidates in the network (host) coordinate
    // space; NMS runs before mapping the kept boxes back to the input image
    DetectionOutput candidates;
    for (int64_t i = 0; i < proposal_counts; ++i) {
        const float cx = out_data[0 * proposal_counts + i];
        const float cy = out_data[1 * proposal_counts + i];
        const float w = out_data[2 * proposal_counts + i];
        const float h = out_data[3 * proposal_counts + i];

        float cls_score = 0.0f;
        int cls_id = -1;
        // class scores occupy rows 4..row_size-1 (all of them, the last 4
        // classes must not be dropped)
        for (int64_t j = 4; j < row_size; ++j) {
            const float score = out_data[j * proposal_counts + i];
            if (score > cls_score) {
                cls_score = score;
                cls_id = static_cast<int>(j - 4);
            }
        }
        if (cls_score < _m_detection_params.score_threshold) {
            continue;
        }

        jinq::models::io_define::object_detection::bbox candidate;
        candidate.bbox = cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h);
        candidate.score = cls_score;
        candidate.class_id = cls_id;
        candidates.push_back(candidate);
    }

    // shared per-class cv::dnn::NMSBoxes suppression used by all detectors
    DetectionOutput nms_result =
        jinq::common::CvUtils::nms_boxes_per_class(candidates, _m_detection_params.score_threshold, _m_detection_params.nms_threshold);
    if (nms_result.size() > static_cast<size_t>(_m_detection_params.keep_top_k)) {
        nms_result.resize(static_cast<size_t>(_m_detection_params.keep_top_k));
    }

    // rescale kept boxes from the network space to the original image size
    for (auto &bbox : nms_result) {
        bbox.bbox = transform_bboxes(cv::Rect2d(bbox.bbox), context);
        if (bbox.class_id >= 0 && bbox.class_id < static_cast<int>(_m_detection_params.class_names.size())) {
            bbox.category = _m_detection_params.class_names[static_cast<size_t>(bbox.class_id)];
        }
    }
    output = std::move(nms_result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
YoloV8Detector<INPUT, OUTPUT>::YoloV8Detector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("YOLOV8") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
