/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_decode.h
 * Date: 26-9-7
 ************************************************/

#ifndef MORTRED_MODELS_OBJECT_DETECTION_YOLOV8_DECODE_H
#define MORTRED_MODELS_OBJECT_DETECTION_YOLOV8_DECODE_H

#include <cstdint>
#include <vector>

#include "models/io/object_detection.h"
#include "models/object_detection/detection_params.h"

namespace jinq {
namespace models {
namespace object_detection {

/***
 * Collect class-filtered YOLOv8 candidates from a packed [1, 4+C, N] f32
 * output0 tensor. Boxes stay in network (letterboxed) coordinates so NMS
 * can run before letterbox unmap. row_size is 4 + class_nums. min_box_area_px
 * is width*height in that space.
 */
inline void collect_yolov8_candidates(const float *out_data, int64_t row_size, int64_t proposal_counts, float score_threshold,
                                      float min_box_area_px,
                                      jinq::models::io_define::object_detection::std_object_detection_output *candidates) {
    if (out_data == nullptr || candidates == nullptr || row_size < 5 || proposal_counts <= 0) {
        return;
    }
    // the [4+C, N] layout is column-major per anchor, so a per-anchor class
    // loop strides a full row (N * 4B) between scores - a cache miss per
    // read. Scan each class row sequentially instead and keep a running
    // per-anchor max; identical output (same strict '>' with ascending class
    // order resolves ties the same way, and the filters are side-effect
    // free, so their order is not observable).
    const int64_t n = proposal_counts;
    const int64_t class_count = row_size - 4;
    std::vector<float> best_score(n, 0.0f);
    std::vector<int32_t> best_class(n, -1);
    for (int64_t j = 0; j < class_count; ++j) {
        const float *row = out_data + (4 + j) * n;
        for (int64_t i = 0; i < n; ++i) {
            const float score = row[i];
            if (score > best_score[i]) {
                best_score[i] = score;
                best_class[i] = static_cast<int32_t>(j);
            }
        }
    }
    const std::vector<float> cx(out_data + 0 * n, out_data + 1 * n);
    const std::vector<float> cy(out_data + 1 * n, out_data + 2 * n);
    const std::vector<float> box_w(out_data + 2 * n, out_data + 3 * n);
    const std::vector<float> box_h(out_data + 3 * n, out_data + 4 * n);
    for (int64_t i = 0; i < n; ++i) {
        const float w = box_w[i];
        const float h = box_h[i];
        if (w <= 0.0f || h <= 0.0f) {
            continue;
        }
        const float cls_score = best_score[i];
        if (cls_score < score_threshold) {
            continue;
        }

        jinq::models::io_define::object_detection::bbox candidate;
        candidate.bbox = cv::Rect2f(cx[i] - w / 2.0f, cy[i] - h / 2.0f, w, h);
        candidate.score = cls_score;
        candidate.class_id = best_class[i];
        if (!passes_min_box_area(candidate.bbox, min_box_area_px)) {
            continue;
        }
        candidates->push_back(candidate);
    }
}

} // namespace object_detection
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_OBJECT_DETECTION_YOLOV8_DECODE_H
