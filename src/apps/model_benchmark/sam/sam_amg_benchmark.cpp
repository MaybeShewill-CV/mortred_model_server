/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_amg_benchmark.cpp
 * Date: 23-9-20
 ************************************************/
// test sam trt model

#include <string>
#include <chrono>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "opencv2/opencv.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/segment_anything/sam_trt_amg_decoder.h"
#include "models/segment_anything/sam_vit_trt_encoder.h"
#include "models/segment_anything/sam_trt_decoder.h"

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::models::segment_anything::SamTrtAmgDecoder;
using jinq::models::segment_anything::SamVitTrtEncoder;
using jinq::models::segment_anything::SamTrtDecoder;

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
    SamTrtAmgDecoder sam_trt_amg_decoder;
    sam_trt_amg_decoder.init(cfg);
    if (!sam_trt_amg_decoder.is_successfully_initialized()) {
        LOG(ERROR) << "init sam trt amg decoder failed";
        return -1;
    }
    sam_trt_amg_decoder.set_encoder_input_size(cv::Size(1024, 1024));
    sam_trt_amg_decoder.set_ori_image_size(input_image.size());

    SamVitTrtEncoder sam_vit_encoder;
    sam_vit_encoder.init(cfg);

    SamTrtDecoder sam_trt_decoder;
    sam_trt_decoder.init(cfg);
    sam_trt_decoder.set_encoder_input_size(cv::Size(1024, 1024));
    sam_trt_decoder.set_ori_image_size(input_image.size());

    std::vector<cv::Mat> masks;
    std::vector<float> img_embeds;

    sam_vit_encoder.encode(input_image, img_embeds);

    LOG(INFO) << "Start benchmarking prompt mask decoder interface ...";
    float scale = 1024.f / static_cast<float>(input_image.cols);
    std::vector<std::vector<cv::Point2f> > prompt_points = {
        {cv::Point2f(562 * scale, 749 * scale), },
        {cv::Point2f(435 * scale, 388 * scale), },
        {cv::Point2f(105 * scale, 166 * scale), },
        {cv::Point2f(1028 * scale, 490 * scale), },
        {cv::Point2f(562 * scale, 749 * scale), },
        {cv::Point2f(435 * scale, 388 * scale), },
        {cv::Point2f(105 * scale, 166 * scale), },
        {cv::Point2f(1028 * scale, 490 * scale), },
        {cv::Point2f(562 * scale, 749 * scale), },
        {cv::Point2f(435 * scale, 388 * scale), },
        {cv::Point2f(105 * scale, 166 * scale), },
        {cv::Point2f(1028 * scale, 490 * scale), },
    };
    for (auto idx = 0; idx < 100; ++idx) {
        auto t_start = std::chrono::high_resolution_clock::now();
        sam_trt_decoder.decode(img_embeds, prompt_points, masks);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << " .... iter: " << idx + 1 << " decode mask cost time: " << t_cost << " ms";
    }
    for (auto idx = 0; idx < 100; ++idx) {
        masks.clear();
        auto t_start = std::chrono::high_resolution_clock::now();
        sam_trt_amg_decoder.decode(img_embeds, prompt_points, masks);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << " .... iter: " << idx + 1 << " amg decode mask cost time: " << t_cost << " ms";
    }
    std::string output_file_name = FilePathUtil::get_file_name(input_image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_sam_trt_amg_output.png";
    std::string output_path = FilePathUtil::concat_path(
        "../demo_data/model_test_input/sam", output_file_name);
    cv::Mat color_output;
    CvUtils::visualize_sam_output_masks(input_image, masks, color_output);
    cv::imwrite(output_path, color_output);

    LOG(INFO) << "sam amg prediction result image has been written into: " << output_path;
}