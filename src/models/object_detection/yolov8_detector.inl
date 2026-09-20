/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_detector.inl
 * Date: 24-3-13
 ************************************************/

#include "yolov8_detector.h"

#include "glog/logging.h"
#include "models/backend/gpu_jpeg_decoder.h"
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
    // W4 JPEG DCT-domain reduced decode: strict by default (never
    // upsamples); "budget" allows one notch of letterbox upsampling within
    // image_decode_budget_upscale (an accuracy tradeoff owned by this
    // model's config, gated by the golden test in docs/perf)
    float decode_upscale = 1.0f;
    const auto decode_mode = params.contains("image_decode_mode")
                                 ? params["image_decode_mode"].value<std::string>()
                                 : std::optional<std::string>{};
    if (decode_mode.has_value() && *decode_mode == "budget") {
        decode_upscale = 1.25f;
        if (params.contains("image_decode_budget_upscale")) {
            const auto configured = params["image_decode_budget_upscale"].value<double>();
            if (configured.has_value() && *configured > 1.0 && *configured <= 2.0) {
                decode_upscale = static_cast<float>(*configured);
            }
        }
        LOG(INFO) << "yolov8 reduced JPEG decode enabled (budget_upscale=" << decode_upscale << ")";
    }
    // S1 GPU decode backend: "auto" uses nvjpeg only when the startup perf
    // race beat cv::imdecode; "gpu" forces it (capability probe only);
    // default "cpu" keeps the OpenCV path. Per-request CPU fallback applies
    // in every mode.
    int decode_gpu = 0;
    const auto decode_backend = params.contains("image_decode_backend")
                                    ? params["image_decode_backend"].value<std::string>()
                                    : std::optional<std::string>{};
    if (decode_backend.has_value()) {
        if (*decode_backend == "gpu") {
            decode_gpu = 2;
        } else if (*decode_backend == "auto") {
            decode_gpu = 1;
        }
    }
    if (decode_gpu != 0) {
        LOG(INFO) << "yolov8 gpu jpeg decode mode=" << (decode_gpu == 2 ? "force" : "auto")
                  << " (probe: " << jinq::models::backend::gpu_jpeg::backend_name() << ", recommended="
                  << (jinq::models::backend::gpu_jpeg::recommended() ? "yes" : "no") << ")";
    }
    this->set_image_decode_hint(_m_input_size_host, decode_upscale, decode_gpu);
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> YoloV8Detector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // letterbox / colour / normalize, emitted in the session's input dtype -
    // fused single-pass kernel first (numerically identical for f32 engines,
    // 5x less memory traffic), fluent chain kept as the f32 fallback
    const auto &input_info = this->session().inputs().front();
    auto fused = jinq::models::backend::letterbox_bgr_nchw(input_image, _m_input_size_host, input_info.name, input_info.dtype);
    if (fused.ok()) {
        return {std::move(fused.value)};
    }
    LOG(ERROR) << fused.error;
    if (input_info.dtype != jinq::models::backend::DType::F32) {
        return {};
    }
    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .bgr_to_rgb()
                      .letterbox(_m_input_size_host)
                      .to_float()
                      .scale(1.0f / 255.0f)
                      .nchw(input_info.name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
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
