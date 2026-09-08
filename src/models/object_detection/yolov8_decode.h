/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_decode.h
 * Date: 26-9-7
 ************************************************/

#ifndef MORTRED_MODELS_OBJECT_DETECTION_YOLOV8_DECODE_H
#define MORTRED_MODELS_OBJECT_DETECTION_YOLOV8_DECODE_H

#include <cstdint>

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
    for (int64_t i = 0; i < proposal_counts; ++i) {
        const float cx = out_data[0 * proposal_counts + i];
        const float cy = out_data[1 * proposal_counts + i];
        const float w = out_data[2 * proposal_counts + i];
        const float h = out_data[3 * proposal_counts + i];
        if (w <= 0.0f || h <= 0.0f) {
            continue;
        }

        float cls_score = 0.0f;
        int cls_id = -1;
        for (int64_t j = 4; j < row_size; ++j) {
            const float score = out_data[j * proposal_counts + i];
            if (score > cls_score) {
                cls_score = score;
                cls_id = static_cast<int>(j - 4);
            }
        }
        if (cls_score < score_threshold) {
            continue;
        }

        jinq::models::io_define::object_detection::bbox candidate;
        candidate.bbox = cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h);
        candidate.score = cls_score;
        candidate.class_id = cls_id;
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
