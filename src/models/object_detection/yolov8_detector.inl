/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_detector.inl
 * Date: 24-3-13
 ************************************************/

#include "yolov8_detector.h"

#include "glog/logging.h"
#include "models/backend/model_runtime.h"
#include "models/object_detection/detector_common.h"
#include "models/object_detection/yolo_decode.h"

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
    this->set_image_decode_hint(_m_input_size_host, parse_image_decode_upscale(params, "yolov8"));
    this->set_gpu_preprocess(yolo_letterbox_gpu_preprocess(input_info.dtype));
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> YoloV8Detector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    return yolo_letterbox_nchw(input_image, _m_input_size_host, this->session().inputs().front());
}

template <typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                      const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    F32OutputView output_view;
    const auto output_status = backend::validated_f32_named_output(
        outputs, "output0", {jinq::models::backend::DType::F32, 3, {1, _m_detection_params.class_nums + 4, -1}}, "yolov8", &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto &tensor = *output_view.tensor;
    const float *out_data = output_view.data;
    const auto row_size = tensor.shape[1];
    const auto proposal_counts = tensor.shape[2];

    DetectionOutput candidates;
    collect_yolov8_candidates(out_data, row_size, proposal_counts, _m_detection_params.score_threshold,
                              _m_detection_params.min_box_area_px, &candidates);

    LetterboxGeometry letterbox;
    std::string geometry_error;
    if (!backend::make_letterbox_geometry(context, &letterbox, &geometry_error)) {
        LOG(ERROR) << "yolov8 " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    DetectionOutput nms_result = finalize_detections(std::move(candidates), _m_detection_params, context);
    unmap_letterbox_detections(nms_result, letterbox, context.source_size);
    output = std::move(nms_result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
YoloV8Detector<INPUT, OUTPUT>::YoloV8Detector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("YOLOV8") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
