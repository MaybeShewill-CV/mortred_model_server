/************************************************
 * Author: Codex
 * File: cv_image_input.h
 * Date: 2026-08-13
 ************************************************/

#ifndef MORTRED_MODELS_CV_IMAGE_INPUT_H
#define MORTRED_MODELS_CV_IMAGE_INPUT_H

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace cv_input {

/***
 * file_input -> cv::Mat: reads with original channels after existence check
 */
inline cv::Mat load_image(const io_define::common_io::file_input& in) {
    cv::Mat ret;
    if (!jinq::common::FilePathUtil::is_file_exist(in.input_image_path)) {
        DLOG(WARNING) << "input image: " << in.input_image_path << " not exist";
        return ret;
    }
    return cv::imread(in.input_image_path, cv::IMREAD_UNCHANGED);
}

/***
 * mat_input -> cv::Mat: refcounted shallow copy, zero overhead
 */
inline cv::Mat load_image(const io_define::common_io::mat_input& in) {
    return in.input_image;
}

/***
 * base64_input -> cv::Mat: base64 decode then imdecode with original channels
 */
inline cv::Mat load_image(const io_define::common_io::base64_input& in) {
    return jinq::common::cv_utils::decode_base64_str_into_cvmat(
        in.input_image_content, cv::IMREAD_UNCHANGED);
}

}  // namespace cv_input
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_CV_IMAGE_INPUT_H
