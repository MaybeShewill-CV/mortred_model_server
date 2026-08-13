/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: fast_sam_benchmark.cpp
 * Date: 23-9-14
 ************************************************/
// test fast sam

#include <string>
#include <chrono>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "opencv2/opencv.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "factory/sam_task.h"

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::segment_anything::std_fast_sam_output;
using jinq::factory::segment_anything::create_fast_sam_segmentor;

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
    auto fast_sam_model = create_fast_sam_segmentor<mat_input, std_fast_sam_output>("fast_sam");
    auto cfg_parsed = toml::parse_file(config_file_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(cfg_parsed.error().description());
        return -1;
    }
    auto cfg = std::move(cfg_parsed).table();
    fast_sam_model->init(cfg);
    if (!fast_sam_model->is_successfully_initialized()) {
        LOG(ERROR) << "init fast-sam failed";
    }

    std::string input_image_path = "../demo_data/model_test_input/sam/truck.jpg";
    if (argc >= 3) {
        input_image_path = argv[2];
    }
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);

    LOG(INFO) << "Start benchmarking sam predict interface ...";
    cv::Mat everything_mask;
    mat_input model_input{input_image};
    for (auto i = 0; i < 10; ++i) {
        auto t_start = std::chrono::system_clock::now();
        fast_sam_model->run(model_input, everything_mask);
        auto t_end = std::chrono::system_clock::now();
        auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << "... infer: " << i << ", cost time: " << t_cost;
    }
    cv::Mat everything_color_mask;
    CvUtils::colorize_sam_everything_mask(everything_mask, everything_color_mask);
    cv::Mat merge_result;
    cv::addWeighted(input_image, 0.65, everything_color_mask, 0.35, 0.0, merge_result);

    std::string output_file_name = FilePathUtil::get_file_name(input_image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_fastsam_everything_result.png";
    std::string output_path = FilePathUtil::concat_path(
        "../demo_data/model_test_input/sam", output_file_name);
    cv::imwrite(output_path, merge_result);
    LOG(INFO) << "fast-sam everything result image has been written into: " << output_path;

    return 0;
}
