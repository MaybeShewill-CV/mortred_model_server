/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: rtdetr_detector.inl
 * Date: 2026-08-30
 ************************************************/

#include "rtdetr_detector.h"

#include <vector>

#include "glog/logging.h"

#include "models/backend/model_runtime.h"

namespace jinq {
namespace models {
namespace object_detection {

template <typename INPUT, typename OUTPUT>
RtdetrDetector<INPUT, OUTPUT>::RtdetrDetector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("RTDETR") {}

/*********** Implement Model Public Func Sets ***********/

template <typename INPUT, typename OUTPUT>
std::vector<jinq::models::backend::NamedTensor> RtdetrDetector<INPUT, OUTPUT>::preprocess(const cv::Mat &image) {
    // TODO(new_model): convert the request image into the named input tensors this model expects.
    // Reference shape: src/models/classification/mobilenetv2.inl
    //
    // Derive the network input size from session().inputs() and keep it in a
    // member you declare, then pack with the Phase 1 runtime toolkit:
    //
    //   return jinq::models::backend::ImagePipeline(image)
    //       .resize(input_size)
    //       .to_float()
    //       .scale(1.0f / 255.0f)
    //       .nchw(session().inputs().front().name);
    (void)image;
    LOG(ERROR) << "RTDETR::preprocess is not implemented yet";
    return {};
}

template <typename INPUT, typename OUTPUT>
StatusCode RtdetrDetector<INPUT, OUTPUT>::postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                                                      const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    // TODO(new_model): validate and decode the named output tensors.
    // Reference shape: src/models/object_detection/yolov8_detector.inl
    //
    // Always go through OutputReader so a malformed tensor becomes
    // MODEL_OUTPUT_CONTRACT_FAILED instead of a partially decoded result:
    //
    //   auto view = jinq::models::backend::OutputReader(outputs, "<output name>")
    //                   .f32()
    //                   .shape({1, N})
    //                   .finite()
    //                   .read();
    //   if (!view.ok()) { return view.status; }
    (void)outputs;
    (void)context;
    (void)output;
    LOG(ERROR) << "RTDETR::postprocess is not implemented yet";
    return StatusCode::MODEL_NOT_IMPLEMENTED;
}

template <typename INPUT, typename OUTPUT> StatusCode RtdetrDetector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    // TODO(new_model): read model specific keys from [RTDETR.params].
    // Reference shape: src/models/classification/mobilenetv2.inl
    //
    //   const auto reader = jinq::models::backend::ParamReader(params, "[RTDETR.params]")
    //                           .get("model_input_image_size", &_m_input_size)
    //                           .min(1);
    //   return reader.status().status;
    (void)params;
    return StatusCode::MODEL_NOT_IMPLEMENTED;
}

} // namespace object_detection
} // namespace models
} // namespace jinq
