/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_trt_benchmark.cpp
 * Date: 23-9-7
 ************************************************/
// test sam trt model

#include <string>
#include <chrono>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "opencv2/opencv.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/segment_anything/sam_vit_trt_encoder.h"

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::models::segment_anything::SamVitTrtEncoder;

int main(int argc, char** argv) {
    google::InstallFailureSignalHandler();

    if (argc < 2) {
        LOG(INFO) << "usage exe config_file";
        return -1;
    }

    // test
    std::string config_file_path = argv[1];
    if (!FilePathUtil::is_file_exist(config_file_path)) {
        LOG(ERROR) << "config file path: " << config_file_path << " not exists";
        return -1;
    }
    SamVitTrtEncoder sam_vit_trt_encoder;
    auto cfg = toml::parse(config_file_path);
    sam_vit_trt_encoder.init(cfg);
    if (!sam_vit_trt_encoder.is_successfully_initialized()) {
        LOG(ERROR) << "init sam trt encoder failed";
        return -1;
    }

    std::string input_image_path = "../demo_data/model_test_input/sam/truck.jpg";
    if (argc >= 3) {
        input_image_path = argv[2];
    }
    if (!FilePathUtil::is_file_exist(input_image_path)) {
        LOG(ERROR) << "input image file path: " << input_image_path << " not exists";
        return -1;
    }
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);

    std::vector<cv::Mat> masks;
    std::vector<float> img_embeds;

    LOG(INFO) << "Start benchmarking vit encoder interface ...";
    for (auto idx = 0; idx < 10; ++idx) {
        auto t_start = std::chrono::system_clock::now();
        sam_vit_trt_encoder.encode(input_image, img_embeds);
        auto t_end = std::chrono::system_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << " .... iter: " << idx + 1 << " encoding cost time: " << t_cost << " ms";
    }
    LOG(INFO) << img_embeds[0] << " "
              << img_embeds[1] << " "
              << img_embeds[2] << " "
              << img_embeds[3] << " "
              << img_embeds[4];
}