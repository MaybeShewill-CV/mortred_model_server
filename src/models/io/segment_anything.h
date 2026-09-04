/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: segment_anything.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_SEGMENT_ANYTHING_H
#define MORTRED_MODELS_IO_SEGMENT_ANYTHING_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace segment_anything {

// segment anything

struct sam_prompt_input {
    cv::Mat image;
    std::vector<cv::Rect> bboxes;
    std::vector<std::vector<cv::Point2f>> prompt_points;
};
using std_sam_prompt_output = std::vector<cv::Mat>;

struct sam_amg_output {
    std::vector<cv::Mat> segmentations;
    std::vector<int32_t> areas;
    std::vector<cv::Rect> bboxes;
    std::vector<float> preds_ious;
    std::vector<float> preds_stability_scores;
    std::vector<cv::Point2f> point_coords;
};
using std_sam_amg_output = sam_amg_output;

using std_fast_sam_output = cv::Mat;

} // namespace segment_anything
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_SEGMENT_ANYTHING_H
