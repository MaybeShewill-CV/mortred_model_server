/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: metric3d_benchmark.cpp
* Date: 23-10-27
************************************************/

// metric3d benckmark tool

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "factory/mono_depth_estimate_task.h"

using jinq::common::FilePathUtil;
using jinq::common::Timestamp;
using jinq::common::CvUtils;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::mono_depth_estimation::std_mde_output;
using jinq::factory::mono_depth_estimation::create_metric3d_estimator;

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path [test_image_path]";
        return -1;
    }

    std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }

    std::string input_image_path;
    if (argc == 3) {
        input_image_path = argv[2];
        LOG(INFO) << "input test image path: " << input_image_path;
    } else {
        input_image_path = "../demo_data/model_test_input/mono_depth_estimation/0000000005.png";
        LOG(INFO) << "use default input test image path: " << input_image_path;
    }
    if (!FilePathUtil::is_file_exist(input_image_path)) {
        LOG(INFO) << "test input image file: " << input_image_path << " not exist";
        return -1;
    }

    // construct model input
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_COLOR);
    struct mat_input model_input {
        input_image
    };
    std_mde_output model_output;
    // construct detector
    auto estimator = create_metric3d_estimator<mat_input, std_mde_output>("metric3d");
    auto cfg = toml::parse(cfg_file_path);
    estimator->init(cfg);
    if (!estimator->is_successfully_initialized()) {
        LOG(INFO) << "metric3d estimator init failed";
        return -1;
    }

    // run benchmark
    int loop_times = 10;
    LOG(INFO) << "input test image size: " << input_image.size();
    LOG(INFO) << "estimator run loop times: " << loop_times;
    LOG(INFO) << "start metric3d benchmark at: " << Timestamp::now().to_format_str();
    auto ts = Timestamp::now();
    for (int i = 0; i < loop_times; ++i) {
        estimator->run(model_input, model_output);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;

    // save colorized depth map
    cv::Mat colorized_depth_map;
    CvUtils::colorize_depth_map(model_output.depth_map, colorized_depth_map);
    std::string output_file_name = FilePathUtil::get_file_name(input_image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_metric3d_depth_result.png";
    std::string output_path = FilePathUtil::concat_path(
        "../demo_data/model_test_input/mono_depth_estimation", output_file_name);
    cv::imwrite(output_path, colorized_depth_map);
    LOG(INFO) << "prediction result image has been written into: " << output_path;

    return 0;
}



