/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolo_decode.h
 * Date: 26-9-20
 ************************************************/

#ifndef MORTRED_MODELS_OBJECT_DETECTION_YOLO_DECODE_H
#define MORTRED_MODELS_OBJECT_DETECTION_YOLO_DECODE_H

#include <cmath>
#include <cstdint>
#include <vector>

#include "models/io/object_detection.h"
#include "models/object_detection/detection_params.h"

namespace jinq {
namespace models {
namespace object_detection {

/*** ── YOLOv5 / YOLOv6: packed-row [B, N, 5+C] ─────────────────────────
 * Each row is cx, cy, w, h, obj, cls... in network (letterboxed) pixels.
 * Score = obj × best_cls. min_box_area_px is in network space.
 */
inline void collect_yolo_cxcywh_obj_candidates(const float *out_data, int64_t batch_nums, int64_t proposal_counts, int class_nums,
                                               float score_threshold, float min_box_area_px,
                                               jinq::models::io_define::object_detection::std_object_detection_output *candidates) {
    if (out_data == nullptr || candidates == nullptr || class_nums < 1 || proposal_counts <= 0 || batch_nums <= 0) {
        return;
    }
    const int64_t row_size = static_cast<int64_t>(class_nums) + 5;
    for (int64_t batch_num = 0; batch_num < batch_nums; ++batch_num) {
        const int64_t batch_offset = batch_num * proposal_counts * row_size;
        for (int64_t bbox_index = 0; bbox_index < proposal_counts; ++bbox_index) {
            const int64_t offset = batch_offset + bbox_index * row_size;
            int class_id = -1;
            float max_cls_score = 0.0f;
            for (int cls_idx = 0; cls_idx < class_nums; ++cls_idx) {
                const float cls_score = out_data[offset + cls_idx + 5];
                if (cls_score > max_cls_score) {
                    max_cls_score = cls_score;
                    class_id = cls_idx;
                }
            }
            const float bbox_score = out_data[offset + 4] * max_cls_score;
            if (bbox_score < score_threshold) {
                continue;
            }
            const float box_w = out_data[offset + 2];
            const float box_h = out_data[offset + 3];
            if (box_w <= 0.0f || box_h <= 0.0f) {
                continue;
            }
            jinq::models::io_define::object_detection::bbox tmp_bbox;
            tmp_bbox.class_id = class_id;
            tmp_bbox.score = bbox_score;
            tmp_bbox.bbox = cv::Rect2f(out_data[offset + 0] - box_w / 2.0f, out_data[offset + 1] - box_h / 2.0f, box_w, box_h);
            if (!passes_min_box_area(tmp_bbox.bbox, min_box_area_px)) {
                continue;
            }
            candidates->push_back(tmp_bbox);
        }
    }
}

/*** ── YOLOv7: raw head [1, 3, H, W, attrs] ─────────────────────────────
 * Per-anchor, per-grid-cell decoding with sigmoid activation and anchor
 * scaling: center = (2*sigmoid(x) - 0.5 + grid) * stride,
 * size = (2*sigmoid(w))² * anchor. Score = sigmoid(obj) × sigmoid(cls).
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

/*** ── YOLOv8: column-major [1, 4+C, N] ─────────────────────────────────
 * No objectness, cls score is direct. Row-sequential scan (W3): the
 * [4+C, N] layout strides a full row between scores per anchor, so scan
 * each class row sequentially with a running per-anchor max instead.
 * Identical output (same strict '>' tie-break, side-effect-free filters).
 */
inline void collect_yolov8_candidates(const float *out_data, int64_t row_size, int64_t proposal_counts, float score_threshold,
                                      float min_box_area_px,
                                      jinq::models::io_define::object_detection::std_object_detection_output *candidates) {
    if (out_data == nullptr || candidates == nullptr || row_size < 5 || proposal_counts <= 0) {
        return;
    }
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

#endif // MORTRED_MODELS_OBJECT_DETECTION_YOLO_DECODE_H
