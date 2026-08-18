/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mobilenetv2_classification_server.cpp
* Date: 22-6-19
************************************************/

// mobilenetv2 classification server tool

#include <glog/logging.h>
#include <workflow/WFFacilities.h>

#include "factory/classification_task.h"

using jinq::factory::classification::create_mobilenetv2_cls_server;

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
        LOG(ERROR) << "parse server config file failed: " << config_file_path << ", error: "
                   << std::string(config_parsed.error().description());
        return -1;
    }
    auto config = std::move(config_parsed).table();
    const toml::table* server_cfg_ptr = config["MOBILENETV2_CLASSIFICATION_SERVER"].as_table();
    if (server_cfg_ptr == nullptr) {
        LOG(ERROR) << "Config section MOBILENETV2_CLASSIFICATION_SERVER missing or not a table";
        return -1;
    }
    const auto& server_cfg = *server_cfg_ptr;
    auto port = server_cfg["port"].value_or<int64_t>(0);
    auto host = server_cfg["host"].value_or<std::string>("127.0.0.1");
    LOG(INFO) << "serve on port: " << port;

    auto server = create_mobilenetv2_cls_server("mobilenetv2_cls_server");
        auto status = server->init(config);
        if (status != jinq::common::StatusCode::OK) {
            LOG(ERROR) << "server init failed, status: " << std::to_string(static_cast<int>(status));
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
