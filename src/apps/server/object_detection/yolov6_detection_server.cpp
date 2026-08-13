/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov6_detection_server.cpp
* Date: 23-3-3
************************************************/

// yolov6 detection server tool

#include <glog/logging.h>
#include <workflow/WFFacilities.h>

#include "factory/obj_detection_task.h"

using jinq::factory::object_detection::create_yolov6_det_server;

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
    auto config_parsed = toml::parse_file(config_file_path);
    if (!config_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(config_parsed.error().description());
        return -1;
    }
    auto config = std::move(config_parsed).table();
    const auto& server_cfg = config["YOLOV6_DETECTION_SERVER"];
    auto port = server_cfg["port"].value_or<int64_t>(0);
    auto host = server_cfg["host"].value_or<std::string>("127.0.0.1");
    LOG(INFO) << "serve on port: " << port;

    auto server = create_yolov6_det_server("yolov6_det_server");
        auto status = server->init(config);
        if (status != jinq::common::StatusCode::OK) {
            LOG(ERROR) << "server init failed, status: " << std::to_string(status);
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
