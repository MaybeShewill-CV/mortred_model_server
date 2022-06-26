/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: nanodet_detection_server.cpp
* Date: 22-6-21
************************************************/

// nanodet detection server tool

#include <glog/logging.h>
#include <workflow/WFHttpServer.h>
#include <workflow/WFFacilities.h>

#include "server/object_detection/nano_det_server.h"

using morted::server::object_detection::NanoDetServer;

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
    settings.endpoint_params.max_connections = 500;
    settings.endpoint_params.response_timeout = 30 * 1000;
    WORKFLOW_library_init(&settings);

    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config = toml::parse(config_file_path);
    const auto& server_cfg = config.at("NANODET_DETECTION_SERVER");
    auto port = server_cfg.at("port").as_integer();
    LOG(INFO) << "serve on port: " << port;

    NanoDetServer server;
    server.init(config);
    if (server.start(port) == 0) {
		wait_group.wait();
		server.stop();
	} else {
		LOG(ERROR) << "Cannot start server";
		return -1;
	}

    return 0;
}