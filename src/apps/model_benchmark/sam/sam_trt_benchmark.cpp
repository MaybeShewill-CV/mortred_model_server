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
#include "models/segment_anything/sam_trt_segmentor.h"
//#include "models/segment_anything/sam_vit_trt_encoder.h"
//#include "models/segment_anything/sam_trt_decoder.h"

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::models::segment_anything::SamTrtSegmentor;
//using jinq::models::segment_anything::SamVitTrtEncoder;
//using jinq::models::segment_anything::SamTrtDecoder;

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
    SamTrtSegmentor sam_trt_segmentor;
    auto cfg = toml::parse(config_file_path);
    sam_trt_segmentor.init(cfg);
    if (!sam_trt_segmentor.is_successfully_initialized()) {
        LOG(ERROR) << "init sam trt segmentor failed";
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
        sam_trt_segmentor.get_embedding(input_image, img_embeds);
        auto t_end = std::chrono::system_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << " .... iter: " << idx + 1 << " encoding cost time: " << t_cost << " ms";
    }

    LOG(INFO) << "Start benchmarking prompt mask decoder interface ...";
    std::vector<std::vector<cv::Point2f> > prompt_points = {
        {cv::Point2f(562, 749), },
    };
    for (auto idx = 0; idx < 10; ++idx) {
        auto t_start = std::chrono::system_clock::now();
        sam_trt_segmentor.decode_masks(img_embeds, input_image.size(), prompt_points, masks);
        auto t_end = std::chrono::system_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << " .... iter: " << idx + 1 << " decode mask cost time: " << t_cost << " ms";
    }

    LOG(INFO) << "Start benchmarking sam predict interface ...";
//    sam_trt_segmentor.predict(
//        input_image,
//        {
//            cv::Rect(483, 683, 158, 132),
//            cv::Rect(220, 327, 430, 122),
//            cv::Rect(77, 78, 58, 176),
//            cv::Rect(972, 464, 111, 52)
//        },
//        masks);
    sam_trt_segmentor.predict(input_image, prompt_points, masks);

    std::string output_file_name = FilePathUtil::get_file_name(input_image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_sam_trt_output.png";
    std::string output_path = FilePathUtil::concat_path(
        "../demo_data/model_test_input/sam", output_file_name);
    cv::Mat color_output;
    CvUtils::visualize_sam_output_masks(input_image, masks, color_output);
    cv::imwrite(output_path, color_output);
    LOG(INFO) << "sam prediction result image has been written into: " << output_path;
}