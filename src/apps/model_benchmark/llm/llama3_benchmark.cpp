/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llama3_benchmark.cpp
 * Date: 24-11-22
 ************************************************/

// llama3 benchmark

#include <glog/logging.h>
#include "toml/toml.hpp"

#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "models/llm/llama/llama3.h"

using jinq::common::FilePathUtil;
using jinq::common::Timestamp;
using jinq::models::llm::llama::Llama3;

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

    // construct llama3 model
    Llama3<std::string, std::string> model;
    auto cfg = toml::parse(cfg_file_path);
    model.init(cfg);
    if (!model.is_successfully_initialized()) {
        LOG(INFO) << "llama3 model init failed";
        return -1;
    }

    std::string input = "\n<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nWho are you<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nI am an assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nWho creates you?<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n";
    LOG(INFO) << "input prompt text: " << input;
    std::string out;
    model.run(input, out);
    LOG(INFO) << "generated output: " << out;

    return 0;
}
