/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: fast_sam_benchmark.cpp
 * Date: 23-6-29
 ************************************************/
// test fast sam

#include <string>
#include <chrono>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "opencv2/opencv.hpp"

#include "common/cv_utils.h"
#include "models/segment_anything/fast_sam_segmentor.h"

using jinq::common::CvUtils;
using jinq::models::segment_anything::FastSamSegmentor;

int main(int argc, char** argv) {
    // test
    std::string config_file_path = argv[1];

    FastSamSegmentor fast_sam_model;
    auto cfg = toml::parse(config_file_path);
    fast_sam_model.init(cfg);
    if (!fast_sam_model.is_successfully_initialized()) {
        LOG(ERROR) << "init sam failed";
    }

    std::string input_image_path = "../demo_data/model_test_input/sam/truck.jpg";
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);

    std::vector<cv::Mat> masks;
    LOG(INFO) << "Start benchmarking sam predict interface ...";
    for (auto i = 0; i < 100; ++i) {
        auto t_start = std::chrono::system_clock::now();
        fast_sam_model.predict(
            input_image,
            {
                cv::Rect(483, 683, 158, 132),
                cv::Rect(220, 327, 430, 122),
                cv::Rect(77, 78, 58, 176),
                cv::Rect(972, 464, 111, 52)
            },
            masks);
        auto t_end = std::chrono::system_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count();
        LOG(INFO) << "... infer: " << i << ", cost time: " << t_cost;
    }


    LOG(INFO) << "complete";
}
