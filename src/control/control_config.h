/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: control_config.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_CONTROL_CONFIG_H
#define MORTRED_CONTROL_CONTROL_CONFIG_H

#include <map>
#include <string>

namespace mortred {
namespace control {

struct SupervisorConfig {
    std::string api_host = "127.0.0.1";
    int api_port = 8787;
    bool autostart_default = false;
    int start_concurrency = 1;           // gateway first, then models with this width
    int log_rotate_mb = 10;
    std::string log_dir = "logs";        // relative to project root
    // process layout (source tree defaults; install tree injected via env)
    std::string bin_dir = "_bin";
    std::string lib_dir = "_lib";
    std::string libs_dir = "3rd_party/libs";
};

struct GatewayConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    int request_size_limit_mb = 64;
    int max_connections = 1000;
    int upstream_send_timeout_ms = 180000;
    int upstream_recv_timeout_ms = 180000;
};

/***
 * Per-server supervision policy from [servers.<id>]. Absent values fall back
 * to the supervisor defaults; restart_policy is "on-failure" | "always" | "no".
 */
struct ServerPolicy {
    bool enabled = true;
    bool has_autostart = false;
    bool autostart = false;
    bool has_restart_policy = false;
    std::string restart_policy = "on-failure";
};

struct ControlConfig {
    SupervisorConfig supervisor;
    GatewayConfig gateway;
    std::map<std::string, ServerPolicy> servers;

    /*** resolve the effective autostart/restart policy for a server id */
    ServerPolicy effective_policy(const std::string& id) const;

    /*** parse conf/mortred.toml; false + err on malformed/unknown values */
    static bool load(const std::string& path, ControlConfig* out, std::string* err);
};

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_CONTROL_CONFIG_H
