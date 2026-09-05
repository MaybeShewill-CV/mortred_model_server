/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: model_server_main.h
* Date: 26-8-19
************************************************/

// Shared main() body for all model server executables: glog setup, config
// parsing, host/port extraction, server start and wait. Per-server mains
// reduce to a section name + factory callback.
#ifndef MORTRED_APPS_MODEL_SERVER_MAIN_H
#define MORTRED_APPS_MODEL_SERVER_MAIN_H

#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

#include <glog/logging.h>
#include <workflow/WFFacilities.h>
#include "toml/toml.hpp"

#include "server/abstract_server.h"

namespace jinq {
namespace apps {
using jinq::common::StatusCode;

inline int run_model_server_main(
    int argc, char** argv,
    const std::string& server_section,
    const std::function<std::unique_ptr<jinq::server::BaseAiServer>(const std::string&)>&
        make_server) {
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
        LOG(ERROR) << "parse toml config file failed, error: "
                   << std::string(config_parsed.error().description());
        return -1;
    }
    auto config = std::move(config_parsed).table();
    const auto& server_cfg = config[server_section];
    auto port = server_cfg["port"].value_or<int64_t>(0);
    auto host = server_cfg["host"].value_or<std::string>("127.0.0.1");
    // environment overrides (highest precedence): the supervisor injects these
    // to force loopback binding + internal auth for managed children
    if (const char* env = std::getenv("MORTRED_LISTEN_HOST"); env != nullptr && *env != '\0') {
        host = env;
    }
    if (const char* env = std::getenv("MORTRED_LISTEN_PORT"); env != nullptr && *env != '\0') {
        try {
            const int env_port = std::stoi(env);
            if (env_port > 0 && env_port <= 65535) {
                port = env_port;
            }
        } catch (...) {
            LOG(WARNING) << "ignoring invalid MORTRED_LISTEN_PORT: " << env;
        }
    }
    LOG(INFO) << "serve on port: " << port;

    // factory registration key: process-internal registry, no external consumer
    auto server = make_server("server");
    auto status = server->init(config);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "server init failed, status: "
                   << std::to_string(static_cast<int>(status));
        return -1;
    }
    const int rc = server->start(host.c_str(), static_cast<unsigned short>(port));
    if (rc == 0) {
        wait_group.wait();
        server->stop();
        return 0;
    }
    LOG(ERROR) << "Cannot start server on " << host << ":" << port;
    if (errno != 0) {
        LOG(ERROR) << "listen errno=" << errno << ": " << std::strerror(errno);
    } else {
        LOG(ERROR) << "listen failed (port already in use is the usual cause; "
                   << "stop mortred-supervisor or another process on this port)";
    }
    return -1;
}

}  // namespace apps
}  // namespace jinq

#endif  // MORTRED_APPS_MODEL_SERVER_MAIN_H
