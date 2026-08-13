/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: metric3d_estimation_server.cpp
* Date: 23-11-01
************************************************/

// metric3d estimation server tool

#include <glog/logging.h>
#include <workflow/WFFacilities.h>

#include "factory/mono_depth_estimate_task.h"

using jinq::factory::mono_depth_estimation::create_metric3d_estimation_server;

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
    auto config_parsed = toml::parse_file(config_file_path);
    if (!config_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(config_parsed.error().description());
        return -1;
    }
    auto config = std::move(config_parsed).table();
    const auto& server_cfg = config["METRIC3D_ESTIMATION_SERVER"];
    auto port = server_cfg["port"].value_or<int64_t>(0);
    auto host = server_cfg["host"].value_or<std::string>("127.0.0.1");
    LOG(INFO) << "serve on port: " << port;

    auto server = create_metric3d_estimation_server("metric3d_estimate_server");
    auto status = server->init(config);
    if (status != jinq::common::StatusCode::OK) {
        LOG(INFO) << "metric3d estimate server init failed";
        return -1;
    }
    if (server->start(host.c_str(), static_cast<unsigned short>(port)) == 0) {
        wait_group.wait();
        server->stop();
    } else {
        LOG(ERROR) << "Cannot start server";
        return -1;
    }

    return 0;
}