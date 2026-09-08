/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolo_cxcywh_decode.h
 * Date: 26-9-9
 ************************************************/

#ifndef MORTRED_MODELS_OBJECT_DETECTION_YOLO_CXCYWH_DECODE_H
#define MORTRED_MODELS_OBJECT_DETECTION_YOLO_CXCYWH_DECODE_H

#include <cstdint>

#include "models/io/object_detection.h"
#include "models/object_detection/detection_params.h"

namespace jinq {
namespace models {
namespace object_detection {

/***
 * Collect class-filtered YOLO packed-row candidates from a [B, N, 5+C] f32
 * tensor (YOLOv5 / YOLOv6). Each row is cx, cy, w, h, obj, cls... in network
 * (letterboxed) pixels. min_box_area_px is width*height in that space, before
 * letterbox unmap.
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

} // namespace object_detection
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_OBJECT_DETECTION_YOLO_CXCYWH_DECODE_H
