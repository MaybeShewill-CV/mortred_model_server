/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_amg_benchmark.cpp
 * Date: 23-9-20
 ************************************************/
// test sam amg model

#include <string>
#include <chrono>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "opencv2/opencv.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/segment_anything/sam_automask_generator/sam_automask_generator.h"

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::models::segment_anything::SamAutoMaskGenerator;
using jinq::models::segment_anything::AmgMaskOutput;

int main(int argc, char** argv) {
    google::InstallFailureSignalHandler();

    if (argc < 2) {
        LOG(INFO) << "usage exe config_file";
        return -1;
    }

    // read input image
    std::string input_image_path = "../demo_data/model_test_input/sam/truck.jpg";
    if (argc >= 3) {
        input_image_path = argv[2];
    }
    if (!FilePathUtil::is_file_exist(input_image_path)) {
        LOG(ERROR) << "input image file path: " << input_image_path << " not exists";
        return -1;
    }
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);

    // test
    std::string config_file_path = argv[1];
    if (!FilePathUtil::is_file_exist(config_file_path)) {
        LOG(ERROR) << "config file path: " << config_file_path << " not exists";
        return -1;
    }
    auto cfg = toml::parse(config_file_path);
    SamAutoMaskGenerator sam_amg;
    sam_amg.init(cfg);
    if (!sam_amg.is_successfully_initialized()) {
        LOG(ERROR) << "init sam auto mask generator model failed";
        return -1;
    }

    AmgMaskOutput amg_output;
    for (auto idx = 0; idx < 10; ++idx) {
        auto t_start = std::chrono::high_resolution_clock::now();
        sam_amg.generate(input_image, amg_output);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds >(t_end - t_start).count();
        LOG(INFO) << " .... iter: " << idx + 1 << ", amg generate mask cost time: " << t_cost << " ms";
    }

    auto seg_mask_counts = static_cast<int>(amg_output.segmentations.size());
    auto color_pool = CvUtils::generate_color_map(seg_mask_counts + 1);
    cv::Mat color_mask = cv::Mat::zeros(amg_output.segmentations[0].size(), CV_8UC3);
    for (auto idx = 0; idx < amg_output.segmentations.size(); ++idx) {
        auto mask = amg_output.segmentations[idx];
        auto mask_bbox = amg_output.bboxes[idx];
        auto color = color_pool[idx];
        for (auto row = 0; row < mask.rows; ++row) {
            auto mask_row = mask.ptr<float>(row);
            auto color_mask_row = color_mask.ptr<cv::Vec3b>(row);
            for (auto col = 0; col < mask.cols; ++col) {
                if (mask_row[col] == 255.0f) {
                    color_mask_row[col][0] = static_cast<uchar>(color[0]);
                    color_mask_row[col][1] = static_cast<uchar>(color[1]);
                    color_mask_row[col][2] = static_cast<uchar>(color[2]);
                }
            }
        }
        cv::rectangle(color_mask, mask_bbox, color, 2);
    }

    std::string output_file_name = FilePathUtil::get_file_name(input_image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_sam_amg_output.png";
    std::string output_path = FilePathUtil::concat_path(
        "../demo_data/model_test_input/sam", output_file_name);
    cv::Mat output_mask;
    cv::addWeighted(input_image, 0.6, color_mask, 0.4, 0.0, output_mask);
    cv::imwrite(output_path, output_mask);
    LOG(INFO) << "sam amg prediction result image has been written into: " << output_path;
}