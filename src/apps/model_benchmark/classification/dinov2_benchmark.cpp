/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: dinov2_benchmark.cpp
 * Date: 23-6-27
 ************************************************/

// dinov2 benckmark tool

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "factory/classification_task.h"

using jinq::common::FilePathUtil;
using jinq::common::Timestamp;
using jinq::common::CvUtils;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::classification::std_classification_output;
using jinq::factory::classification::create_dinov2_classifier;

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
        input_image_path = "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG";
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
    std_classification_output model_output{};
    // construct detector
    auto classifier = create_dinov2_classifier<mat_input, std_classification_output>("mobilenetv2");
    auto cfg = toml::parse(cfg_file_path);
    classifier->init(cfg);

    if (!classifier->is_successfully_initialized()) {
        LOG(INFO) << "mobilenetv2 classifier init failed";
        return -1;
    }

    // run benchmark
    int loop_times = 1;
    LOG(INFO) << "input test image size: " << input_image.size();
    LOG(INFO) << "classifier run loop times: " << loop_times;
    LOG(INFO) << "start mobilenetv2 benchmark at: " << Timestamp::now().to_format_str();
    auto ts = Timestamp::now();

    for (int i = 0; i < loop_times; ++i) {
        classifier->run(model_input, model_output);
    }

    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;

    LOG(INFO) << "dinov2 image embedding features dims: " << model_output.scores.size();
    LOG(INFO) << "dinov2 image embedding features:" << " "
              << model_output.scores[0] << " "
              << model_output.scores[1] << " "
              << model_output.scores[2] << " "
              << model_output.scores[3] << " "
              << model_output.scores[4];

    return 0;
}