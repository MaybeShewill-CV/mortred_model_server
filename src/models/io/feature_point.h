/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: feature_point.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_FEATURE_POINT_H
#define MORTRED_MODELS_IO_FEATURE_POINT_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace feature_point {

// image feature point

struct fp {
    cv::Point2f location;
    std::vector<float> descriptor;
    float score;
};
using std_feature_point_output = std::vector<fp>;

struct matched_fp {
    std::pair<fp, fp> m_fp;
    float match_score;
};
using std_feature_point_match_output = std::vector<matched_fp>;

} // namespace feature_point
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_FEATURE_POINT_H
