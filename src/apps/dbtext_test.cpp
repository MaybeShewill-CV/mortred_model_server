/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dbtext_test.cpp
* Date: 22-6-6
************************************************/

#include <memory>

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/time_stamp.h"
#include "factory/base_factory.h"

using morted::common::Timestamp;
using morted::models::io_define::common_io::file_input;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::common_io::base64_input;
using morted::models::io_define::image_ocr::common_out;
using morted::models::image_ocr::DBTextDetector;
using morted::factory::DBTextModelFactory;

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

    DBTextModelFactory<file_input, common_out> db_fin_creator;
    auto db_text_1 = db_fin_creator.create_model();
    db_text_1->init(cfg);
    Timestamp ts;
    for (int i = 0; i < 500; ++i) {
        db_text_1->run(file_in, out);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "db text file in cost time: " << cost_time << "s";
    for (const auto& bbox : out) {
        LOG(INFO) << bbox.bbox << " " << bbox.score;
    }

    DBTextModelFactory<mat_input, common_out> db_min_creator;
    auto db_text_2 = db_min_creator.create_model();
    mat_in.input_image = cv::imread("../demo_data/model_test_input/image_ocr/db_text/test.jpg", cv::IMREAD_UNCHANGED);
    db_text_2->init(cfg);
    out.clear();

    ts = Timestamp::now();
    for (int i = 0; i < 500; ++i) {
        db_text_2->run(mat_in, out);
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