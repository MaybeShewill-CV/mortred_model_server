/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: tokenizer_benchmark.cpp
 * Date: 24-12-18
 ************************************************/
// llama3 tokenizer benchmark

#include <string>

#include <glog/logging.h>
#include "toml/toml.hpp"
#include "fmt/format.h"

#include "common/time_stamp.h"
#include "common/status_code.h"
#include "common/file_path_util.h"
#include "models/llm/llama/llama3.h"

using jinq::common::Timestamp;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::llm::llama::Llama3;

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3 && argc != 4) {
        LOG(ERROR) << "usage ...";
        LOG(ERROR) << "exe conf_path [tokenize_text] [loop_times]";
        return -1;
    }

    std::string cfg_path = argv[1];
    if (!FilePathUtil::is_file_exist(cfg_path)) {
        LOG(ERROR) << fmt::format("cfg file: {} not exist", cfg_path);
        return -1;
    }
    auto cfg = toml::parse(cfg_path);

    Llama3<std::string, std::string> model;
    model.init(cfg);
    if (!model.is_successfully_initialized()) {
        LOG(ERROR) << "init llama model failed";
        return -1;
    }

    std::string text = "hello world";
    if (argc == 3) {
        text = argv[2];
    }

    int loop_times = 100;
    if (argc == 4) {
        loop_times = std::stoi(argv[3]);
    }
    LOG(INFO) << fmt::format("input text: {}", text);
    LOG(INFO) << fmt::format("tokenizer run loop times: {}", loop_times);
    LOG(INFO) << "start tokenizer benchmark at: " << Timestamp::now().to_format_str();
    std::vector<int32_t > tokens;
    auto ts = Timestamp::now();
    for (int i = 0; i < loop_times; ++i) {
        model.tokenize(text, tokens);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;
    std::ostringstream oss;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << tokens[i];
    }
    LOG(INFO) << fmt::format("output tokens: {}", oss.str());
    return 1;
}
