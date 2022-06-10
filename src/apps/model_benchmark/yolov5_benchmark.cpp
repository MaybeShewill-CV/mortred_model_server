/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov5_benchmark.cpp
* Date: 22-6-10
************************************************/

// yolov5 benckmark tool

#include <random>

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "factory/obj_detection_task.h"

using morted::common::FilePathUtil;
using morted::common::Timestamp;
using morted::common::CvUtils;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::object_detection::common_out;
using morted::factory::object_detection::create_yolov5_detector;

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
        input_image_path = "../demo_data/model_test_input/object_detection/bus.jpg";
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
    std::vector<common_out> model_output{};
    // construct detector
    auto detector = create_yolov5_detector<mat_input, common_out>("yolov5");
    auto cfg = toml::parse(cfg_file_path);
    detector->init(cfg);

    if (!detector->is_successfully_initialized()) {
        LOG(INFO) << "yolov5 detector init failed";
        return -1;
    }

    // run benchmark
    int loop_times = 100;
    LOG(INFO) << "input test image size: " << input_image.size();
    LOG(INFO) << "detector run loop times: " << loop_times;
    LOG(INFO) << "start yolov5 benchmark at: " << Timestamp::now().to_format_str();
    auto ts = Timestamp::now();

    for (int i = 0; i < loop_times; ++i) {
        detector->run(model_input, model_output);
    }

    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;

    CvUtils::vis_object_detection(input_image, model_output, 80);
    std::string output_path = "../demo_data/model_test_input/object_detection/bus_yolov5_ret.jpg";
    cv::imwrite(output_path, input_image);
    LOG(INFO) << "detection result image has been written into: " << output_path;

    return 1;
}



