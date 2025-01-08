/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: qwen2_vl_benchmark.cpp
 * Date: 25-1-7
 ************************************************/
// qwen2 vl benchmark app

#include <glog/logging.h>
#include "toml/toml.hpp"
#include <opencv2/opencv.hpp>

#include "common/status_code.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "models/llm/qwen2_vl/qwen2_vl.h"

using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::io_define::llm::vlm::file_input;
using jinq::models::io_define::llm::vlm::mat_input;
using jinq::models::io_define::llm::vlm::std_vlm_output;
using jinq::models::llm::qwen2_vl::Qwen2VL;

int main(int argc, char** argv) {

    if (argc != 2 && argc != 3 && argc != 4) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path [image_path] [text_prompt]";
        return -1;
    }

    std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }

    std::string image_path = "../demo_data/model_test_input/llm/qwen2_vl/hehe.jpeg";
    if (argc > 2) {
        image_path = argv[2];
    }
    if (!FilePathUtil::is_file_exist(image_path)) {
        LOG(INFO) << fmt::format("image file: {} not exist", image_path);
        return -1;
    }

    std::string text_prompt = "what's in the picture?";
    if (argc > 3) {
        text_prompt = argv[3];
    }
    if (text_prompt.empty()) {
        LOG(INFO) << "empty text prompt";
        return -1;
    }

    // construct qwen2-vl model
    Qwen2VL<mat_input, std_vlm_output> generator;
    auto cfg = toml::parse(cfg_file_path);
    generator.init(cfg);
    if (!generator.is_successfully_initialized()) {
        LOG(INFO) << "qwen2-vl generator init failed";
        return -1;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    mat_input input{image, text_prompt};
    std_vlm_output output;
    auto status = generator.run(input, output);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "generator run session failed";
        return status;
    } else {
        LOG(INFO) << "user input: ";
        LOG(INFO) << fmt::format("---- {}", input.text);
        LOG(INFO) << "assistant response: ";
        LOG(INFO) << fmt::format("---- {}", output);
    }

    input.text = "what's the title in the top-left corner of the image";
    status = generator.run(input, output);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "generator run session failed";
        return status;
    } else {
        LOG(INFO) << "user input: ";
        LOG(INFO) << fmt::format("---- {}", input.text);
        LOG(INFO) << "assistant response: ";
        LOG(INFO) << fmt::format("---- {}", output);
    }

    return status;
}