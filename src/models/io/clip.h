/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: clip.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_CLIP_H
#define MORTRED_MODELS_IO_CLIP_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace clip {

// image clip

enum class ClipTaskType {
    TEXT_EMBEDDING = 0,
    IMAGE_EMBEDDING = 1,
    TEXTS_TO_IMAGE = 2,
    IMAGES_TO_TEXT = 3,
};

struct clip_input {
    ClipTaskType task_type = ClipTaskType::TEXT_EMBEDDING;
    std::string text;
    cv::Mat image;
    std::vector<std::string> texts;
    std::vector<cv::Mat> images;
};
struct clip_output {
    std::vector<float> embeddings;
    std::vector<float> simi_scores;
};

} // namespace clip
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_CLIP_H
