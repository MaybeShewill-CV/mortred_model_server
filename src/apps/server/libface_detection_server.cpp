/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: libface_detection_server.cpp
* Date: 22-6-26
************************************************/

// libface detection server tool

#include <glog/logging.h>
#include <workflow/WFHttpServer.h>
#include <workflow/WFFacilities.h>

#include "factory/obj_detection_task.h"

using jinq::factory::object_detection::create_libface_det_server;

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
    const auto& server_cfg = config.at("LIBFACE_DETECTION_SERVER");
    auto port = server_cfg.at("port").as_integer();
    LOG(INFO) << "serve on port: " << port;

    auto server = create_libface_det_server("libface_det_server");
    auto status = server->init(config);
    if (status != jinq::common::StatusCode::OK) {
        LOG(INFO) << "libface detection server init failed";
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