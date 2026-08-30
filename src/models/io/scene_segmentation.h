/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: scene_segmentation.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_SCENE_SEGMENTATION_H
#define MORTRED_MODELS_IO_SCENE_SEGMENTATION_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace scene_segmentation {

// image scene segmentation

struct seg_output {
    cv::Mat segmentation_result;
    cv::Mat colorized_seg_mask;
};
using std_scene_segmentation_output = seg_output;

} // namespace scene_segmentation
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_SCENE_SEGMENTATION_H
