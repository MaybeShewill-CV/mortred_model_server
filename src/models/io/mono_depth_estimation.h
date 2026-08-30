/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: mono_depth_estimation.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_MONO_DEPTH_ESTIMATION_H
#define MORTRED_MODELS_IO_MONO_DEPTH_ESTIMATION_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace mono_depth_estimation {

// mono depth estimation

struct mde_output {
    cv::Mat confidence_map;
    cv::Mat depth_map;
    cv::Mat colorized_depth_map;
};
using std_mde_output = mde_output;

} // namespace mono_depth_estimation
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_MONO_DEPTH_ESTIMATION_H
