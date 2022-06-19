/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: resnet_classification_server.cpp
* Date: 22-6-19
************************************************/

// resnet classification server tool

#include <glog/logging.h>
#include <workflow/WFHttpServer.h>
#include <workflow/WFFacilities.h>

#include "server/classification/resnet_server.hpp"

int main(int argc, char** argv) {

    if (argc != 2) {
        LOG(INFO) << "usage:";
        LOG(INFO) << "exe cfg_path";
        return -1;
    }

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    static WFFacilities::WaitGroup wait_group(1);
    WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.compute_threads = -1;
    settings.handler_threads = 50;
    settings.endpoint_params.max_connections = 5000;
    settings.endpoint_params.response_timeout = 30 * 1000;
    WORKFLOW_library_init(&settings);

    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config = toml::parse(config_file_path);
    const auto& server_cfg = config.at("RESNET_CLASSIFICATION_SERVER");
    auto port = server_cfg.at("server_port").as_integer();
    LOG(INFO) << "port: " << port;

    WFHttpServer server(morted::server::classification::server_process);
    if (server.start(port) == 0) {
		wait_group.wait();
		server.stop();
	} else {
		LOG(ERROR) << "Cannot start server";
		return -1;
	}

    return 0;
}