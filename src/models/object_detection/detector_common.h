#ifndef MORTRED_MODELS_OBJECT_DETECTION_DETECTOR_COMMON_H
#define MORTRED_MODELS_OBJECT_DETECTION_DETECTOR_COMMON_H

#include <cstring>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "common/cv_utils.h"
#include "common/status_code.h"
#include "models/backend/inference_context.h"
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
struct DetectionGeometryScale {
    float width = 0.0f;
    float height = 0.0f;
};

inline bool make_detection_geometry_scale(const backend::InferenceContext &context, DetectionGeometryScale *scale, std::string *error) {
    if (scale == nullptr) {
        if (error != nullptr) {
            *error = "detection geometry scale output pointer is null";
        }
        return false;
    }
    if (context.source_size.width <= 0 || context.source_size.height <= 0 || context.network_size.width <= 0 ||
        context.network_size.height <= 0) {
        if (error != nullptr) {
            *error = "invalid request geometry: source=" + std::to_string(context.source_size.width) + "x" +
                     std::to_string(context.source_size.height) + ", network=" + std::to_string(context.network_size.width) + "x" +
                     std::to_string(context.network_size.height);
        }
        return false;
    }
    scale->width = static_cast<float>(context.source_size.width) / static_cast<float>(context.network_size.width);
    scale->height = static_cast<float>(context.source_size.height) / static_cast<float>(context.network_size.height);
    return true;
}

inline cv::Rect2f scale_detection_bbox(const cv::Rect2f &bbox, const DetectionGeometryScale &scale) {
    return {bbox.x * scale.width, bbox.y * scale.height, bbox.width * scale.width, bbox.height * scale.height};
}

inline cv::Point2f scale_detection_point(const cv::Point2f &point, const DetectionGeometryScale &scale) {
    return {point.x * scale.width, point.y * scale.height};
}

/***
 * A named f32 output which passed its shape/buffer/finite-value contract.
 * The tensor remains owned by the caller's output vector.
 */
struct F32OutputView {
    const backend::Tensor *tensor = nullptr;
    const float *data = nullptr;
};

inline StatusCode validated_f32_output(const std::vector<backend::NamedTensor> &outputs, const std::string &name,
                                       const backend::TensorContract &contract, const std::string &log_prefix,
                                       F32OutputView *view = nullptr) {
    const auto *named = backend::find_output(outputs, name);
    if (named == nullptr) {
        LOG(ERROR) << log_prefix << " output tensor '" << name << "' is missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    std::string error;
    if (!backend::validate_output_tensor(*named, contract, &error)) {
        LOG(ERROR) << log_prefix << " output contract failed: " << error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }

    const float *data = nullptr;
    if (!backend::get_f32_data(named->tensor, &data, &error) ||
        !backend::require_finite_f32(data, static_cast<size_t>(named->tensor.element_count()), named->name, &error)) {
        LOG(ERROR) << log_prefix << " output contract failed: " << error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }
    if (view != nullptr) {
        view->tensor = &named->tensor;
        view->data = data;
    }
    return StatusCode::OK;
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
