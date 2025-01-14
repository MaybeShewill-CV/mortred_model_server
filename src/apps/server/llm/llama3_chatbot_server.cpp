/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llama3_chat_server.cpp
 * Date: 24-11-29
 ************************************************/
// llama3 chat server tool

#include <glog/logging.h>
#include <workflow/WFFacilities.h>

#include "factory/llm_task.h"

using jinq::factory::llm::llama::create_llama3_chat_server;

int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    if (argc != 2) {
        LOG(INFO) << "usage:";
        LOG(INFO) << "exe cfg_path";
        return -1;
    }

    static WFFacilities::WaitGroup wait_group(1);
    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config = toml::parse(config_file_path);
    const auto& server_cfg = config.at("LLAMA3_CHAT_SERVER");
    auto port = server_cfg.at("port").as_integer();
    LOG(INFO) << "serve on port: " << port;

    auto server = create_llama3_chat_server("llama3_chat_server");
    auto status = server->init(config);
    if (status != jinq::common::StatusCode::OK) {
        LOG(INFO) << "llama3 chat server init failed";
        return -1;
    }
    if (server->start(port) == 0) {
        wait_group.wait();
        server->stop();
    } else {
        LOG(ERROR) << "Cannot start server";
        return -1;
    }

    return 0;
}