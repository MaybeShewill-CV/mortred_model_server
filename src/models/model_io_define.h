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
}

// image ocr
namespace image_ocr {
struct common_out {
    cv::Rect2f bbox;
    std::vector<cv::Point2f> polygon;
    float score;
};
}

}
}
}

#endif //MM_AI_SERVER_MODEL_IO_DEFINE_H
