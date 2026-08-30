/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ocr.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_OCR_H
#define MORTRED_MODELS_IO_OCR_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace ocr {

// image ocr

struct text_region {
    cv::Rect2f bbox;
    std::vector<cv::Point2f> polygon;
    float score;
};
using std_text_regions_output = std::vector<text_region>;

} // namespace ocr
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_OCR_H
