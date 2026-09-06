#ifndef MORTRED_MODELS_BACKEND_REQUEST_GEOMETRY_H
#define MORTRED_MODELS_BACKEND_REQUEST_GEOMETRY_H

#include <algorithm>
#include <cmath>
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

/***
 * Ultralytics YOLO export letterbox (auto=False, center pad).
 * scale is the uniform ratio; pad_x/pad_y are the left/top borders applied
 * with int(round(d - 0.1)) / int(round(d + 0.1)). Stretch models keep
 * GeometryScale and must not use this type.
 */
struct LetterboxGeometry {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int pad_right = 0;
    int pad_bottom = 0;
    cv::Size unpadded;
    cv::Size network;
    cv::Size source;
};

inline LetterboxGeometry compute_letterbox_geometry(const cv::Size &source, const cv::Size &network) {
    LetterboxGeometry geom;
    geom.source = source;
    geom.network = network;
    const double ratio =
        std::min(static_cast<double>(network.height) / static_cast<double>(source.height),
                 static_cast<double>(network.width) / static_cast<double>(source.width));
    geom.scale = static_cast<float>(ratio);
    int unpadded_w = static_cast<int>(std::round(static_cast<double>(source.width) * ratio));
    int unpadded_h = static_cast<int>(std::round(static_cast<double>(source.height) * ratio));
    if (unpadded_w > network.width) {
        unpadded_w = network.width;
    }
    if (unpadded_h > network.height) {
        unpadded_h = network.height;
    }
    geom.unpadded = cv::Size(unpadded_w, unpadded_h);
    const double dw = (static_cast<double>(network.width) - static_cast<double>(unpadded_w)) / 2.0;
    const double dh = (static_cast<double>(network.height) - static_cast<double>(unpadded_h)) / 2.0;
    geom.pad_x = static_cast<int>(std::round(dw - 0.1));
    geom.pad_y = static_cast<int>(std::round(dh - 0.1));
    geom.pad_right = static_cast<int>(std::round(dw + 0.1));
    geom.pad_bottom = static_cast<int>(std::round(dh + 0.1));
    if (geom.pad_x < 0) {
        geom.pad_x = 0;
    }
    if (geom.pad_y < 0) {
        geom.pad_y = 0;
    }
    if (geom.pad_right < 0) {
        geom.pad_right = 0;
    }
    if (geom.pad_bottom < 0) {
        geom.pad_bottom = 0;
    }
    return geom;
}

inline bool make_letterbox_geometry(const InferenceContext &context, LetterboxGeometry *geom, std::string *error) {
    if (geom == nullptr) {
        if (error != nullptr) {
            *error = "letterbox geometry output pointer is null";
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
    *geom = compute_letterbox_geometry(context.source_size, context.network_size);
    return true;
}

inline cv::Rect2f unmap_letterbox_bbox(const cv::Rect2f &bbox, const LetterboxGeometry &geom, const cv::Size &source) {
    if (geom.scale <= 0.0f) {
        return {};
    }
    const float pad_x = static_cast<float>(geom.pad_x);
    const float pad_y = static_cast<float>(geom.pad_y);
    float x1 = (bbox.x - pad_x) / geom.scale;
    float y1 = (bbox.y - pad_y) / geom.scale;
    float x2 = (bbox.x + bbox.width - pad_x) / geom.scale;
    float y2 = (bbox.y + bbox.height - pad_y) / geom.scale;
    const float max_x = static_cast<float>(std::max(source.width, 0));
    const float max_y = static_cast<float>(std::max(source.height, 0));
    x1 = std::min(std::max(x1, 0.0f), max_x);
    y1 = std::min(std::max(y1, 0.0f), max_y);
    x2 = std::min(std::max(x2, 0.0f), max_x);
    y2 = std::min(std::max(y2, 0.0f), max_y);
    return {x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1)};
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
