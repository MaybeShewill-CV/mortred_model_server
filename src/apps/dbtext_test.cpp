/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dbtext_test.cpp
* Date: 22-6-6
************************************************/

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "factory/image_ocr_task.h"

using morted::common::Timestamp;
using morted::models::io_define::common_io::file_input;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::common_io::base64_input;
using morted::models::io_define::image_ocr::common_out;
using morted::factory::image_ocr::create_dbtext_detector;

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

    auto model_fc = create_dbtext_detector<file_input, common_out>("dbtext_worker_1");
    auto model_tmp = create_dbtext_detector<file_input, common_out>("dbtext_worker_tmp");
    if (model_fc == nullptr) {
        LOG(ERROR) << "empty model ptr";
        return -1;
    }
    model_fc->init(cfg);
//    Timestamp ts;
//    for (int i = 0; i < 500; ++i) {
//        model_fc->run(file_in, out);
//    }
//    auto cost_time = Timestamp::now() - ts;
//    LOG(INFO) << "db text file in cost time: " << cost_time << "s";
//    for (const auto& bbox : out) {
//        LOG(INFO) << bbox.bbox << " " << bbox.score;
//    }


    auto model_mc = create_dbtext_detector<mat_input, common_out>("dbtext_worker_2");
    auto model_mc_tmp = create_dbtext_detector<mat_input, common_out>("dbtext_worker_3");
//    mat_in.input_image = cv::imread(
//            "../demo_data/model_test_input/image_ocr/db_text/test.jpg",cv::IMREAD_UNCHANGED);
//    model_mc->init(cfg);
//    out.clear();
//
//    ts = Timestamp::now();
//    for (int i = 0; i < 500; ++i) {
//        model_mc->run(mat_in, out);
//    }
//    cost_time = Timestamp::now() - ts;
//    LOG(INFO) << "db text mat in cost time: " << cost_time << "s";
//    LOG(INFO) << "time stamp: " << ts.to_str();
//    LOG(INFO) << "time stamp format str: " << ts.to_format_str();
//
//    for (const auto& bbox : out) {
//        LOG(INFO) << bbox.bbox << " " << bbox.score;
//    }

    return 1;
}