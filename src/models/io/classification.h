/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: classification.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_CLASSIFICATION_H
#define MORTRED_MODELS_IO_CLASSIFICATION_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace classification {

// image classification

struct cls_output {
    int class_id;
    std::string category;
    std::vector<float> scores;
};
using std_classification_output = cls_output;

} // namespace classification
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_CLASSIFICATION_H
