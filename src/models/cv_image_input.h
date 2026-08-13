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

#include "common/base64.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace cv_input {

/***
 * file_input → cv::Mat：校验文件存在后按原通道读取。
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
 * mat_input → cv::Mat：引用计数浅拷贝，零拷贝开销。
 */
inline cv::Mat load_image(const io_define::common_io::mat_input& in) {
    return in.input_image;
}

/***
 * base64_input → cv::Mat：base64 解码后按原通道 imdecode。
 */
inline cv::Mat load_image(const io_define::common_io::base64_input& in) {
    cv::Mat ret;
    auto decoded = jinq::common::Base64::base64_decode(in.input_image_content);
    if (decoded.empty()) {
        DLOG(WARNING) << "image data empty";
        return ret;
    }
    std::vector<uchar> image_vec_data(decoded.begin(), decoded.end());
    cv::imdecode(image_vec_data, cv::IMREAD_UNCHANGED).copyTo(ret);
    return ret;
}

}  // namespace cv_input
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_CV_IMAGE_INPUT_H
