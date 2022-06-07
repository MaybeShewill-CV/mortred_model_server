/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov5_test.cpp
* Date: 22-6-7
************************************************/

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/time_stamp.h"
#include "models/image_object_detection/yolov5_detector.h"

using morted::common::Timestamp;
using morted::models::io_define::common_io::file_input;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::common_io::base64_input;
using morted::models::io_define::image_object_detection::common_out;
using morted::models::image_object_detection::YoloV5Detector;

int main(int argc, char** argv) {

    if (argc != 2) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path";
        return -1;
    }

    LOG(INFO) << "cfg file path: " << argv[1];

    auto cfg = toml::parse(argv[1]);

    file_input file_in{};
    file_in.input_image_path = "../demo_data/model_test_input/image_ocr/db_text/test.jpg";
    mat_input mat_in{};
    base64_input base64_in{};
    std::vector<common_out> out;

    YoloV5Detector<file_input, common_out> yolov5_1;
    yolov5_1.init(cfg);
    Timestamp ts;
    for (int i = 0; i < 50; ++i) {
        yolov5_1.run(file_in, out);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "yolov5 file in cost time: " << cost_time << "s";

    for (const auto& bbox : out) {
        LOG(INFO) << bbox.bbox << " " << bbox.score;
    }

    YoloV5Detector<mat_input, common_out> yolov5_2;
    mat_in.input_image = cv::imread("../demo_data/model_test_input/image_ocr/db_text/test.jpg", cv::IMREAD_UNCHANGED);
    yolov5_2.init(cfg);
    out.clear();

    ts = Timestamp::now();
    for (int i = 0; i < 50; ++i) {
        yolov5_2.run(mat_in, out);
    }
    cost_time = Timestamp::now() - ts;
    LOG(INFO) << "yolov5 mat in cost time: " << cost_time << "s";
    LOG(INFO) << "time stamp: " << ts.to_str();
    LOG(INFO) << "time stamp format str: " << ts.to_format_str();

    for (const auto& bbox : out) {
        LOG(INFO) << bbox.bbox << " " << bbox.score;
    }

    return 1;
}