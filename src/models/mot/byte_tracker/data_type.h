/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: data_type.h
 * Date: 24-5-30
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DATA_TYPE_H
#define MORTRED_MODEL_SERVER_DATA_TYPE_H

#include <cstddef>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace mot {
namespace byte_tracker {

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXES;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURES;

// Kalmanfilter
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

// tracker:
using TRACKER_DATA = std::pair<int, FEATURES>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t {
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
} TRACKER_MATCHED;

// linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMIC;

using RESULT_DATA = std::pair<int, DETECTBOX>;

using Object = jinq::models::io_define::object_detection::bbox;

enum TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
};

}
}
}
}

#endif // MORTRED_MODEL_SERVER_DATA_TYPE_H
