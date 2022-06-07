/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dbtext_test.cpp
* Date: 22-6-6
************************************************/

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "models/image_ocr/db_text_detector.h"

using morted::models::io_define::common_io::file_input;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::common_io::base64_input;
using morted::models::io_define::image_ocr::common_out;
using morted::models::image_ocr::DBTextDetector;

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

    DBTextDetector<file_input, common_out> db_text_1;
    db_text_1.init(cfg);
    for (int i = 0; i < 500; ++i) {
        db_text_1.run(file_in, out);
    }
    for (const auto& bbox : out) {
        LOG(INFO) << bbox.bbox << " " << bbox.score;
    }

    DBTextDetector<mat_input, common_out> db_text_2;
    mat_in.input_image = cv::imread("../demo_data/model_test_input/image_ocr/db_text/test.jpg", cv::IMREAD_UNCHANGED);
    db_text_2.init(cfg);
    out.clear();
    for (int i = 0; i < 500; ++i) {
        db_text_2.run(mat_in, out);
    }
    for (const auto& bbox : out) {
        LOG(INFO) << bbox.bbox << " " << bbox.score;
    }

    return 1;
}