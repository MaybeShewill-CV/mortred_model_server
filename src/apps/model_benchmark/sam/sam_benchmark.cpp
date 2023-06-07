/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: test_sam.cpp
 * Date: 23-5-24
 ************************************************/
// test sam

#include <string>
#include <chrono>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "opencv2/opencv.hpp"

#include "common/cv_utils.h"
#include "models/segment_anything/sam_segmentor.h"

using jinq::common::CvUtils;
using jinq::models::segment_anything::SamSegmentor;

int main(int argc, char** argv) {
    // test
    SamSegmentor sam_model;
    auto cfg = toml::parse("../conf/model/segment_anything/sam_vit_l_config.ini");
    sam_model.init(cfg);
    if (!sam_model.is_successfully_initialized()) {
        LOG(ERROR) << "init sam failed";
    }

    std::string input_image_path = "../demo_data/model_test_input/sam/truck.jpg";
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);

    std::vector<cv::Mat> masks;
    std::vector<float> img_embeds;

    LOG(INFO) << "Start benchmarking vit encoder interface ...";
    for (auto idx = 0; idx < 20; ++idx) {
        auto t_start = std::chrono::system_clock::now();
        sam_model.get_embedding(input_image, img_embeds);
        auto t_end = std::chrono::system_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << " .... iter: " << idx + 1 << " encoding cost time: " << t_cost << " ms";
    }

    LOG(INFO) << "Start benchmarking sam predict interface ...";
    sam_model.predict(
        input_image,
        {
            cv::Rect(483, 683, 158, 132),
            cv::Rect(220, 327, 430, 122),
            cv::Rect(77, 78, 58, 176),
            cv::Rect(972, 464, 111, 52)
        },
        masks);
    LOG(INFO) << masks[0].size();

    cv::Mat color_output;
    CvUtils::visualize_sam_output_masks(input_image, masks, color_output);
    cv::imwrite("../demo_data/model_test_input/sam/truck_sam_output.jpg", color_output);

    LOG(INFO) << "complete";
}
