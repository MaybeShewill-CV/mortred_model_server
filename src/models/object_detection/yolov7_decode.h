/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov7_decode.h
 * Date: 26-9-9
 ************************************************/

#ifndef MORTRED_MODELS_OBJECT_DETECTION_YOLOV7_DECODE_H
#define MORTRED_MODELS_OBJECT_DETECTION_YOLOV7_DECODE_H

#include <cmath>

#include "models/io/object_detection.h"
#include "models/object_detection/detection_params.h"

namespace jinq {
namespace models {
namespace object_detection {

/***
 * Collect one YOLOv7 raw head [1, 3, H, W, 5+C] into network (letterboxed)
 * pixels: (col, row) * stride plus the usual (2*sigmoid-0.5) offset and
 * (2*sigmoid)^2 * anchor size. min_box_area_px is width*height in that space,
 * before letterbox unmap.
 */
inline void collect_yolov7_head_candidates(const float *data, int anchor_nums, int grid_h, int grid_w, int attrs, int stride,
                                           const float anchors[][2], float score_threshold, float min_box_area_px,
                                           jinq::models::io_define::object_detection::std_object_detection_output *candidates) {
    if (data == nullptr || candidates == nullptr || anchors == nullptr || anchor_nums <= 0 || grid_h <= 0 || grid_w <= 0 ||
        attrs < 6 || stride <= 0) {
        return;
    }
    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    const int used_anchors = anchor_nums < 3 ? anchor_nums : 3;
    for (int a = 0; a < used_anchors; ++a) {
        const float anchor_w = anchors[a][0];
        const float anchor_h = anchors[a][1];
        for (int row = 0; row < grid_h; ++row) {
            for (int col = 0; col < grid_w; ++col) {
                const float *p = data + (((a * grid_h + row) * grid_w + col) * attrs);
                const float obj_score = sigmoid(p[4]);
                if (obj_score < 0.05f) {
                    continue;
                }
                int class_id = -1;
                float max_cls_score = 0.0f;
                for (int c = 5; c < attrs; ++c) {
                    const float cls_score = sigmoid(p[c]);
                    if (cls_score > max_cls_score) {
                        max_cls_score = cls_score;
                        class_id = c - 5;
                    }
                }
                const float bbox_score = obj_score * max_cls_score;
                if (bbox_score < score_threshold) {
                    continue;
                }
                const float center_x = (2.0f * sigmoid(p[0]) - 0.5f + static_cast<float>(col)) * static_cast<float>(stride);
                const float center_y = (2.0f * sigmoid(p[1]) - 0.5f + static_cast<float>(row)) * static_cast<float>(stride);
                const float box_w = std::pow(2.0f * sigmoid(p[2]), 2.0f) * anchor_w;
                const float box_h = std::pow(2.0f * sigmoid(p[3]), 2.0f) * anchor_h;
                if (box_w <= 0.0f || box_h <= 0.0f) {
                    continue;
                }
                jinq::models::io_define::object_detection::bbox tmp_bbox;
                tmp_bbox.class_id = class_id;
                tmp_bbox.score = bbox_score;
                tmp_bbox.bbox.x = center_x - box_w / 2.0f;
                tmp_bbox.bbox.y = center_y - box_h / 2.0f;
                tmp_bbox.bbox.width = box_w;
                tmp_bbox.bbox.height = box_h;
                if (!passes_min_box_area(tmp_bbox.bbox, min_box_area_px)) {
                    continue;
                }
                candidates->push_back(tmp_bbox);
            }
        }
    }
}

} // namespace object_detection
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_OBJECT_DETECTION_YOLOV7_DECODE_H
