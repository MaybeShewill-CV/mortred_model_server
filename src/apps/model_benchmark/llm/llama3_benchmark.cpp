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
#include "models/llm/llama/llama3_generator.h"

using jinq::common::FilePathUtil;
using jinq::models::llm::chat_template::Dialog;
using jinq::models::llm::chat_template::ChatMessage;
using jinq::models::llm::llama::Llama3Generator;

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
    Llama3Generator generator;
    auto cfg = toml::parse(cfg_file_path);
    generator.init(cfg);
    if (!generator.is_successfully_initialized()) {
        LOG(INFO) << "llama3 generator init failed";
        return -1;
    }

    Dialog dialog;
    dialog.messages = {
        {"system", "You're a smart AI assistant from Mortred Company"},
        {"user", "Who are you?"},
        {"assistant", "I am a ai assistant"},
        {"user", "Who is your favorite singer?"},
    };
    std::string gen_out;
    generator.chat_completion(dialog, gen_out);
    dialog.messages.emplace_back("assistant", gen_out);
    LOG(INFO) << "assistant: " << gen_out;

    Dialog new_dialog;
    new_dialog.messages.emplace_back("user", "answer last question again");
    generator.chat_completion(new_dialog, gen_out);
    dialog.messages.emplace_back("assistant", gen_out);
    LOG(INFO) << "assistant: " << gen_out;

    generator.clear_kv_cache_cell();
    generator.chat_completion(new_dialog, gen_out);
    LOG(INFO) << "assistant: " << gen_out;

    generator.clear_kv_cache_cell();
    auto status = generator.chat_completion(dialog, gen_out);
    LOG(INFO) << "assistant: " << gen_out;

    return status;
}
