/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: object_detection.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_OBJECT_DETECTION_H
#define MORTRED_MODELS_IO_OBJECT_DETECTION_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace object_detection {

// image object detection

struct bbox {
    cv::Rect2f bbox;
    float score;
    int32_t class_id;
    std::string category;
};
using std_object_detection_output = std::vector<bbox>;

struct face_bbox {
    cv::Rect2f bbox;
    float score;
    int32_t class_id;
    std::string category;
    std::vector<cv::Point2f> landmarks;
};
using std_face_detection_output = std::vector<face_bbox>;

} // namespace object_detection
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_OBJECT_DETECTION_H
