/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: real_esrgan_server.cpp
* Date: 22-10-18
************************************************/

// real esrgan server tool

#include <glog/logging.h>
#include <workflow/WFFacilities.h>

#include "factory/enhancement_task.h"

using jinq::factory::enhancement::create_realesrgan_server;

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

    WFFacilities::WaitGroup wait_group(1);

    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config = toml::parse(config_file_path);
    const auto& server_cfg = config.at("REAL_ESRGAN_SERVER");
    auto port = server_cfg.at("port").as_integer();
    LOG(INFO) << "serve on port: " << port;

    auto server = create_realesrgan_server("real_esrgan_server");
    server->init(config);
    if (server->start(port) == 0) {
        wait_group.wait();
        server->stop();
    } else {
        LOG(ERROR) << "Cannot start server";
        return -1;
    }

    return 0;
}