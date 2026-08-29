/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_detector.inl
 * Date: 24-3-13
 ************************************************/

#include "yolov8_detector.h"

#include <algorithm>

#include "glog/logging.h"
#include "models/object_detection/detector_common.h"

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

    std::vector<NamedTensor> inputs;
    NamedTensor named;
    if (!make_nchw_input(this->session().inputs().front().name, tmp, &named)) {
        return {};
    }
    inputs.push_back(std::move(named));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                      const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    F32OutputView output_view;
    const auto output_status = validated_f32_output(
        outputs, "output0", {jinq::models::backend::DType::F32, 3, {1, _m_detection_params.class_nums + 4, -1}}, "yolov8", &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto &tensor = *output_view.tensor;
    const float *out_data = output_view.data;
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

    DetectionGeometryScale geometry_scale;
    std::string geometry_error;
    if (!make_detection_geometry_scale(context, &geometry_scale, &geometry_error)) {
        LOG(ERROR) << "yolov8 " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    DetectionOutput nms_result = finalize_detections(std::move(candidates), _m_detection_params);

    // rescale kept boxes from the network space to the original image size
    for (auto &bbox : nms_result) {
        bbox.bbox = scale_detection_bbox(bbox.bbox, geometry_scale);
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
