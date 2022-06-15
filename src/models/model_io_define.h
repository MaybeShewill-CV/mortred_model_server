/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: model_io_define.h
 * Date: 22-6-7
 ************************************************/

#ifndef MM_AI_SERVER_MODEL_IO_DEFINE_H
#define MM_AI_SERVER_MODEL_IO_DEFINE_H

#include <string>

#include <opencv2/opencv.hpp>

namespace morted {
namespace models {
namespace io_define {

// common io
namespace common_io {

struct mat_input {
    cv::Mat input_image;
};

struct file_input {
    std::string input_image_path;
};

struct base64_input {
    std::string input_image_content;
};
} // namespace common_io

// image ocr
namespace ocr {

struct text_region {
    cv::Rect2f bbox;
    std::vector<cv::Point2f> polygon;
    float score;
};
using common_text_regions = std::vector<text_region>;

struct common_out {
    cv::Rect2f bbox;
    std::vector<cv::Point2f> polygon;
    float score;
};
} // namespace ocr

// image object detection
namespace object_detection {

struct common_out {
    cv::Rect2f bbox;
    float score;
    int32_t class_id;
};
} // namespace object_detection

// image scene segmentation
namespace scene_segmentation {

struct common_out {
    cv::Mat segmentation_result;
};
} // namespace scene_segmentation

// image enhancement
namespace enhancement {

struct common_out {
    cv::Mat enhancement_result;
};
} // namespace enhancement

// image enhancement
namespace classification {

struct common_out {
    int class_id;
    std::vector<float> scores;
};
} // namespace classification

// image feature point
namespace feature_point {

struct common_out {
    cv::Point2f location;
    std::vector<float> descriptor;
    float score;
};
} // namespace feature_point

} // namespace io_define
} // namespace models
} // namespace morted

#endif // MM_AI_SERVER_MODEL_IO_DEFINE_H
