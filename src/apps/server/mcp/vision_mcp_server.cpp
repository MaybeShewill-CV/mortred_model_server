/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: vision_mcp_server.cpp
* Date: 25-4-24
************************************************/

// vision mcp server tool

#include "workflow/WFFacilities.h"
#include <glog/logging.h>

#include "server/tiny_mcp/vision_mcp_server.h"

using jinq::server::tiny_mcp::VisionMcpServer;

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

    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config = toml::parse(config_file_path);
    const auto& server_cfg = config.at("MCP_SERVER");
    auto port = server_cfg.at("port").as_integer();
    LOG(INFO) << "serve on port: " << port;

    auto server = std::make_unique<VisionMcpServer>();
    auto status = server->init(config);
    if (status != jinq::common::StatusCode::OK) {
        LOG(INFO) << "vision mcp server init failed";
        return -1;
    }
    server->run();

    return 0;
}