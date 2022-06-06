/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dbtext_test.cpp
* Date: 22-6-6
************************************************/

#include <string>

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "models/image_ocr/db_text_detector.h"

using morted::models::image_ocr::dbtext_input;
using morted::models::image_ocr::dbtext_output;
using morted::models::image_ocr::DBTextDetector;

int main(int argc, char** argv) {

    if (argc != 2) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path";
        return -1;
    }

    LOG(INFO) << "cfg file path: " << argv[1];

    auto cfg = toml::parse(argv[1]);
    DBTextDetector dbtext_detector;
    dbtext_detector.init(cfg);

    dbtext_input in{};
    dbtext_output out{};

    struct tmp {};
    tmp tmp_in{};
    tmp tmp_out{};

    dbtext_detector.run(&in, &out);
    dbtext_detector.run(&tmp_in, &tmp_out);

    return 1;
}