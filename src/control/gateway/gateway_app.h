/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: gateway_app.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_GATEWAY_APP_H
#define MORTRED_CONTROL_GATEWAY_APP_H

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <workflow/WFHttpServer.h>

#include "common/stage_timing.h"
#include "control/api_key_manager.h"
#include "control/catalog.h"
#include "control/control_config.h"
#include "control/rate_limit/gcra.h"
#include "control/rate_limit/subject.h"
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

    // ST2 seams. Failed auth must leave both at the pre-request value:
    // consume_rate_limit is not entered, and the authorized source limiter
    // has not inserted a cell (no admit-then-release).
    uint64_t api_key_consume_calls() const { return api_keys_.consume_calls(); }
    size_t authorized_source_tracked() const {
        return ip_limiter_ == nullptr ? 0 : ip_limiter_->stats().tracked;
    }

private:
    // Clocks for the gateway stage histogram. fwd and upstream are taken
    // inside forward_to_model, on the same marks as stage_trace.
    struct StageClocks {
        std::chrono::steady_clock::time_point enter{};
        std::chrono::steady_clock::time_point gw_in{};
        std::chrono::steady_clock::time_point route{};
        std::chrono::steady_clock::time_point auth{};
    };
    struct StageSample {
        double ms[5]{};
        double server_time_ms = 0.0;
    };

    void process(WFHttpTask* task);
    // Admit one source limiter. Returns false only when enforce mode already
    // wrote the 429. `limiter` null or `enabled` false admits nothing.
    // Authorized L2 and optional L0 call this with different stores and
    // layer tags; a 401 path does not call it for L2.
    bool admit_source_limit(WFHttpTask* task, const std::string& method,
                            mortred::control::ratelimit::ShardedIpLimiter* limiter,
                            bool enabled, bool shadow, int limit_rate,
                            const mortred::control::ratelimit::TrustedProxies& proxies,
                            const char* layer, const char* limited_message);
    void forward_to_model(WFHttpTask* task, const ResolvedRoute& route,
                          const std::string& method, const std::string& query,
                          const std::shared_ptr<jinq::common::stage_timing::StageTrace>& trace,
                          const StageClocks& clocks,
                          std::shared_ptr<const ApiKey> inflight_key = nullptr);
    void note_upstream_stage(const StageClocks& clocks,
                             std::chrono::steady_clock::time_point t_fwd,
                             std::chrono::steady_clock::time_point t_upstream,
                             bool job_route, const std::string& status,
                             const void* data, size_t size);
    void record_stage_sample(const double stage_ms[5], double server_time_ms);
    std::string render_stage_metrics() const;
    bool resolve_route(const std::string& path, ResolvedRoute* out) const;
    void maybe_add_cors(WFHttpTask* task) const;
    bool origin_allowed(const std::string& origin) const;
    void load_cors_origins(const std::string& raw);

private:
    Catalog catalog_;
    ControlConfig cfg_;
    ApiKeyManager api_keys_;
    jinq::server::PrometheusMetrics metrics_;
    // L2 authorized per-source GCRA. Constructed at init() when
    // [gateway.rate_limit] is enabled or shadow (non-copyable: sharded
    // mutexes). Null = disabled. Admitted only after auth succeeds.
    std::unique_ptr<mortred::control::ratelimit::ShardedIpLimiter> ip_limiter_;
    mortred::control::ratelimit::TrustedProxies trusted_proxies_;
    bool rate_limit_enabled_ = false;
    bool rate_limit_shadow_ = false;
    int rate_limit_rate_ = 0;
    // L0 optional pre-auth abuse limiter. Separate store from L2; off unless
    // [gateway.abuse_limit] is enabled or shadow.
    std::unique_ptr<mortred::control::ratelimit::ShardedIpLimiter> abuse_limiter_;
    mortred::control::ratelimit::TrustedProxies abuse_proxies_;
    bool abuse_limit_enabled_ = false;
    bool abuse_limit_shadow_ = false;
    int abuse_limit_rate_ = 0;
    std::string auth_token_;
    std::string admin_token_;
    std::string metrics_token_;
    std::string internal_token_;
    std::vector<std::string> cors_origins_;
    std::unique_ptr<WFHttpServer> server_;
    // Paired gateway-stage samples. The histogram sums stay cumulative; the
    // deque keeps at most the newest 4096 rows for the median.
    static constexpr int k_stage_count = 5;
    static constexpr int k_stage_bucket_count = 17;
    static constexpr std::size_t k_stage_sample_cap = 4096;
    mutable std::mutex stage_mu_;
    std::deque<StageSample> stage_samples_;
    std::uint64_t stage_observed_ = 0;
    double stage_sum_[k_stage_count]{};
    std::uint64_t stage_bucket_[k_stage_count][k_stage_bucket_count]{};
};

/*** data-plane reverse proxy entry (called by the thin main in src/apps) */
int run_gateway(int argc, char** argv);

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_GATEWAY_APP_H
