/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov6_detector.inl
 * Date: 23-3-3
 ************************************************/

#include "yolov6_detector.h"

#include <utility>

#include "glog/logging.h"
#include "models/backend/model_runtime.h"
#include "models/object_detection/detector_common.h"
#include "models/object_detection/yolo_cxcywh_decode.h"

namespace jinq {
namespace models {
namespace object_detection {

using DetectionOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode YoloV6Detector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.4f;
    _m_detection_params.nms_threshold = 0.35f;
    _m_detection_params.keep_top_k = 250;
    _m_detection_params.class_nums = 80;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid yolov6 detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected yolov6 input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "yolov6 input shape has dynamic/invalid H/W: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    cv::Size configured_size;
    if (!parse_model_input_size(params, &configured_size, &param_error) ||
        (params.contains("model_input_image_size") && configured_size != _m_input_size_host)) {
        LOG(ERROR) << "invalid yolov6 input size: " << (param_error.empty() ? "configured size mismatches model input" : param_error);
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> YoloV6Detector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // letterbox / colour / normalize, emitted as f32 nchw
    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .letterbox(_m_input_size_host)
                      .bgr_to_rgb()
                      .to_float()
                      .scale(1.0f / 255.0f)
                      .nchw(this->session().inputs().front().name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode YoloV6Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                      const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    F32OutputView output_view;
    const auto output_status = backend::validated_f32_named_output(
        outputs, "outputs", {jinq::models::backend::DType::F32, 3, {1, -1, _m_detection_params.class_nums + 5}}, "yolov6", &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto &tensor = *output_view.tensor;
    const float *output_tensordata = output_view.data;
    const auto batch_nums = tensor.shape[0];
    const auto raw_pred_bbox_nums = tensor.shape[1];

    LetterboxGeometry letterbox;
    std::string geometry_error;
    if (!backend::make_letterbox_geometry(context, &letterbox, &geometry_error)) {
        LOG(ERROR) << "yolov6 " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    DetectionOutput decode_result;
    collect_yolo_cxcywh_obj_candidates(output_tensordata, batch_nums, raw_pred_bbox_nums, _m_detection_params.class_nums,
                                       _m_detection_params.score_threshold, _m_detection_params.min_box_area_px, &decode_result);
    unmap_letterbox_detections(decode_result, letterbox, context.source_size);

    DetectionOutput nms_result = finalize_detections(std::move(decode_result), _m_detection_params, context);
    output = std::move(nms_result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
YoloV6Detector<INPUT, OUTPUT>::YoloV6Detector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("YOLOV6") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
