#ifndef MORTRED_MODELS_BACKEND_REQUEST_GEOMETRY_H
#define MORTRED_MODELS_BACKEND_REQUEST_GEOMETRY_H

#include <string>

#include "common/status_code.h"
#include "models/backend/inference_context.h"

namespace jinq {
namespace models {
namespace backend {

using jinq::common::StatusCode;

/*** request-scoped scale from network coordinates to source-image coordinates ***/
struct GeometryScale {
    float width = 0.0f;
    float height = 0.0f;
};

inline bool make_geometry_scale(const InferenceContext &context, GeometryScale *scale, std::string *error) {
    if (scale == nullptr) {
        if (error != nullptr) {
            *error = "geometry scale output pointer is null";
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

inline cv::Rect2f scale_bbox(const cv::Rect2f &bbox, const GeometryScale &scale) {
    return {bbox.x * scale.width, bbox.y * scale.height, bbox.width * scale.width, bbox.height * scale.height};
}

inline cv::Point2f scale_point(const cv::Point2f &point, const GeometryScale &scale) {
    return {point.x * scale.width, point.y * scale.height};
}

/*** validates the destination geometry used to map dense outputs to the source ***/
inline StatusCode validated_source_size(const InferenceContext &context, const std::string &log_prefix, cv::Size *source_size = nullptr) {
    if (context.source_size.width <= 0 || context.source_size.height <= 0) {
        LOG(ERROR) << log_prefix << " invalid source size " << context.source_size;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    if (source_size != nullptr) {
        *source_size = context.source_size;
    }
    return StatusCode::OK;
}

} // namespace backend
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_BACKEND_REQUEST_GEOMETRY_H
