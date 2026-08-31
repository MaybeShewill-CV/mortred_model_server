#ifndef MORTRED_MODELS_OBJECT_DETECTION_DETECTOR_COMMON_H
#define MORTRED_MODELS_OBJECT_DETECTION_DETECTOR_COMMON_H

#include <cstring>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "common/cv_utils.h"
#include "common/status_code.h"
#include "models/backend/f32_output.h"
#include "models/backend/inference_context.h"
#include "models/backend/request_geometry.h"
#include "models/backend/tensor_contract.h"
#include "models/object_detection/detection_params.h"

namespace jinq {
namespace models {
namespace object_detection {

using jinq::common::StatusCode;

/***
 * Request-scoped scale from the network input space to the source image space.
 * Keep validation separate from multiplication so callers can reject malformed
 * contexts once instead of producing NaN/Inf boxes deep inside a decoder.
 */
using backend::GeometryScale;
using DetectionGeometryScale = GeometryScale;

inline bool make_detection_geometry_scale(const backend::InferenceContext &context, DetectionGeometryScale *scale, std::string *error) {
    return backend::make_geometry_scale(context, scale, error);
}

inline cv::Rect2f scale_detection_bbox(const cv::Rect2f &bbox, const DetectionGeometryScale &scale) {
    return backend::scale_bbox(bbox, scale);
}

inline cv::Point2f scale_detection_point(const cv::Point2f &point, const DetectionGeometryScale &scale) {
    return backend::scale_point(point, scale);
}

using backend::F32OutputView;

inline StatusCode validated_f32_output(const std::vector<backend::NamedTensor> &outputs, const std::string &name,
                                       const backend::TensorContract &contract, const std::string &log_prefix,
                                       F32OutputView *view = nullptr) {
    return backend::validated_f32_named_output(outputs, name, contract, log_prefix, view);
}

/***
 * Shared detector tail: per-class NMS, top-k truncation and category filling.
 * Model-specific decoding deliberately remains in each detector.
 */
template <typename T> inline std::vector<T> finalize_detections(std::vector<T> detections, const DetectionParams &params) {
    auto result = jinq::common::CvUtils::nms_boxes_per_class(detections, params.score_threshold, params.nms_threshold);
    if (result.size() > static_cast<size_t>(params.keep_top_k)) {
        result.resize(static_cast<size_t>(params.keep_top_k));
    }
    for (auto &detection : result) {
        if (detection.class_id >= 0 && detection.class_id < static_cast<int>(params.class_names.size())) {
            detection.category = params.class_names[static_cast<size_t>(detection.class_id)];
        }
    }
    return result;
}

/***
 * Request-aware tail: overlays per-request overrides (validated against the
 * detection ParamSpec schema in the task catalog) on top of the config
 * defaults, then runs the shared finalize path. nullptr params (legacy
 * single-image path) keeps pure config behavior.
 */
template <typename T>
inline std::vector<T> finalize_detections(std::vector<T> detections, const DetectionParams &params,
                                          const backend::InferenceContext &context) {
    if (context.params == nullptr) {
        return finalize_detections(std::move(detections), params);
    }
    DetectionParams effective = params;
    effective.score_threshold = context.params->get_f32("score_threshold", params.score_threshold);
    effective.nms_threshold = context.params->get_f32("nms_threshold", params.nms_threshold);
    effective.keep_top_k = context.params->get_i32("top_k", params.keep_top_k);
    return finalize_detections(std::move(detections), effective);
}

/***
 * Pack a preprocessed CV_32FC3 HWC image into a [1,3,H,W] f32 tensor.
 * Resize/color conversion/normalization stay model-specific.
 */
inline bool make_nchw_input(const std::string &input_name, const cv::Mat &image, backend::NamedTensor *output) {
    if (output == nullptr) {
        LOG(ERROR) << "NCHW input output pointer is null";
        return false;
    }
    if (input_name.empty()) {
        LOG(ERROR) << "NCHW input tensor name is empty";
        return false;
    }
    if (image.empty()) {
        LOG(ERROR) << "NCHW input image is empty";
        return false;
    }
    if (image.type() != CV_32FC3) {
        LOG(ERROR) << "NCHW input image type is " << image.type() << ", expected CV_32FC3";
        return false;
    }

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(image);
    output->name = input_name;
    output->tensor = backend::Tensor::make<float>({1, 3, image.rows, image.cols});
    if (chw_data.size() * sizeof(float) != output->tensor.byte_size()) {
        LOG(ERROR) << "NCHW input data size mismatches the input tensor";
        return false;
    }
    std::memcpy(output->tensor.buffer.data(), chw_data.data(), output->tensor.byte_size());
    return true;
}

} // namespace object_detection
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_OBJECT_DETECTION_DETECTOR_COMMON_H
