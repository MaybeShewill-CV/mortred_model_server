/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: gateway_app.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_GATEWAY_APP_H
#define MORTRED_CONTROL_GATEWAY_APP_H

#include <memory>
#include <string>
#include <vector>

#include <workflow/WFHttpServer.h>

#include "control/api_key_manager.h"
#include "control/catalog.h"
#include "control/control_config.h"
#include "server/prometheus_metrics.h"

namespace mortred {
namespace control {

/*** Explicit per-instance configuration. run() fills it from the process
 * environment (MORTRED_* tokens / host / port); tests construct it directly
 * so two instances with different roots, tokens and ports can coexist in one
 * process. Empty string fields mean "not configured" (same semantics the
 * env-driven path had); empty config_path falls back to
 * <project_root>/conf/mortred.toml. */
struct GatewayInitOptions {
    std::string project_root;
    std::string config_path;
    std::string host;
    int port = 0;
    std::string auth_token;      // external bearer token ("" = none)
    std::string admin_token;     // MORTRED_API_TOKEN (UI / mortredctl)
    std::string metrics_token;   // scrape Bearer; mandatory and distinct
    std::string internal_token;  // shared with the model servers
    std::string cors_origins;    // raw comma-separated origin list
    std::string profile;         // "cpu" / "gpu"; "" = MORTRED_PROFILE / gpu
};

struct ResolvedRoute {
    const ServerEntry* entry = nullptr;
    std::string upstream_path;
    std::string allowed_method;
    bool rewrite_job_urls = false;
    bool append_query = false;
};

/*** Data-plane reverse proxy over the model-server catalog. All state is
 * instance-local (the historical file-scope globals are gone); GatewayApp
 * objects are not thread-safe to construct/destroy concurrently, but any
 * number of independent instances may serve at the same time. Lifetime
 * requirement: the object must outlive its in-flight requests - stop_listen()
 * drains them; the destructor calls it. */
class GatewayApp {
public:
    GatewayApp() = default;
    ~GatewayApp();

    GatewayApp(const GatewayApp&) = delete;
    GatewayApp& operator=(const GatewayApp&) = delete;

    // full process entry: env/config-driven init, listen, serve until stop
    int run(int argc, char** argv);

    // test surface: init runs every fail-closed startup check; listen/stop
    // bracket a non-blocking serve window on cfg_.gateway.host:port
    bool init(const GatewayInitOptions& options);
    bool listen();
    void stop_listen();

    const ControlConfig& config() const { return cfg_; }

private:
    void process(WFHttpTask* task);
    void forward_to_model(WFHttpTask* task, const ResolvedRoute& route,
                          const std::string& method, const std::string& query);
    bool resolve_route(const std::string& path, ResolvedRoute* out) const;
    void maybe_add_cors(WFHttpTask* task) const;
    bool origin_allowed(const std::string& origin) const;
    void load_cors_origins(const std::string& raw);

private:
    Catalog catalog_;
    ControlConfig cfg_;
    ApiKeyManager api_keys_;
    jinq::server::PrometheusMetrics metrics_;
    std::string auth_token_;
    std::string admin_token_;
    std::string metrics_token_;
    std::string internal_token_;
    std::vector<std::string> cors_origins_;
    std::unique_ptr<WFHttpServer> server_;
};

/*** data-plane reverse proxy entry (called by the thin main in src/apps) */
int run_gateway(int argc, char** argv);

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_GATEWAY_APP_H
