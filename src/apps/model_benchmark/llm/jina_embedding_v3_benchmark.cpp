/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: jina_embedding_v3_benchmark.cpp
 * Date: 24-12-18
 ************************************************/
// jinq-embedding-v3 benchmark

#include <string>

#include <glog/logging.h>
#include "toml/toml.hpp"
#include "fmt/format.h"

#include "common/time_stamp.h"
#include "common/status_code.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "factory/llm_task.h"

using jinq::common::Timestamp;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::io_define::llm::embedding::pool_type;
using jinq::models::io_define::llm::embedding::std_embedding_input;
using jinq::models::io_define::llm::embedding::std_embedding_output;
using jinq::factory::llm::embedding::create_jina_embedding_v3;

int main(int argc, char** argv) {
    // usage info
    if (argc != 2 && argc != 3 && argc != 4) {
        LOG(ERROR) << "usage ...";
        LOG(ERROR) << "exe conf_path [tokenize_text] [loop_times]";
        return -1;
    }

    // init input params
    std::string cfg_path = argv[1];
    if (!FilePathUtil::is_file_exist(cfg_path)) {
        LOG(ERROR) << fmt::format("cfg file: {} not exist", cfg_path);
        return -1;
    }
    auto cfg = toml::parse(cfg_path);

    std::string text = "hello world";
    if (argc == 3) {
        text = argv[2];
    }

    int loop_times = 100;
    if (argc == 4) {
        loop_times = std::stoi(argv[3]);
    }

    // init embedding model
    auto model = create_jina_embedding_v3<std_embedding_input , std_embedding_output >("embedding");
    model->init(cfg);
    if (!model->is_successfully_initialized()) {
        LOG(ERROR) << "init llama model failed";
        return -1;
    }

    // run benchmark
    LOG(INFO) << fmt::format("input text: {}", text);
    LOG(INFO) << fmt::format("tokenizer run loop times: {}", loop_times);
    LOG(INFO) << "start tokenizer benchmark at: " << Timestamp::now().to_format_str();
    std_embedding_input input {text, pool_type::EMBEDDING_MEAN_POOLING};
    std_embedding_output output;
    auto ts = Timestamp::now();
    for (int i = 0; i < loop_times; ++i) {
        auto status = model->run(input, output);
        if (status != StatusCode::OK) {
            LOG(ERROR) << "benchmark failed";
            return -1;
        }
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;
    std::ostringstream oss;
    for (size_t i = 0; i < output.token_ids.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << output.token_ids[i];
    }
    LOG(INFO) << fmt::format("output tokens: {}", oss.str());

    return 1;
}
