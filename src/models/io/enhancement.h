/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: enhancement.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_ENHANCEMENT_H
#define MORTRED_MODELS_IO_ENHANCEMENT_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace mortred {
namespace models {
namespace io_define {
namespace enhancement {

// image enhancement

struct enhance_output {
    cv::Mat enhancement_result;
};
using std_enhancement_output = enhance_output;

} // namespace enhancement
} // namespace io_define
} // namespace models
} // namespace mortred

#endif // MORTRED_MODELS_IO_ENHANCEMENT_H
