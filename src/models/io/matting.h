/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: matting.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_MATTING_H
#define MORTRED_MODELS_IO_MATTING_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace matting {

// image matting

struct matting_output {
    cv::Mat matting_result;
};
using std_matting_output = matting_output;

} // namespace matting
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_MATTING_H
