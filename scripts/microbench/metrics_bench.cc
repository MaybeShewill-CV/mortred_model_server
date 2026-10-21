// Baseline microbenchmark for src/server/prometheus_metrics.h (phase 0).
// Measures the per-request metric mix: single-thread latency and 8-thread
// contended throughput. One request ≈ 5 metric calls (the generic CV server's
// sync path: inc_http_requests + observe_http_duration_ms + observe_queue_wait_ms
// + inc_inference_success + set_queue_depth).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "server/prometheus_metrics.h"

using mortred::server::PrometheusMetrics;

static const int kIters = 200000;

static void request_mix(PrometheusMetrics &m) {
    m.inc_http_requests("POST", "0");
    m.observe_http_duration_ms("POST", "0", 12.5);
    m.observe_queue_wait_ms(1.5);
    m.inc_inference_success();
    m.set_queue_depth(3);
}

int main() {
    // single-thread ns per request-equivalent (5 metric calls)
    {
        PrometheusMetrics m;
        m.set_model("bench");
        for (int i = 0; i < 20000; ++i) request_mix(m);  // warmup
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i) request_mix(m);
        const auto t1 = std::chrono::steady_clock::now();
        const double ns_per_call =
            std::chrono::duration<double, std::nano>(t1 - t0).count() / (kIters * 5);
        std::printf("single_thread: %.1f ns/metric-call, %.1f ns/request-equivalent\n",
                    ns_per_call, ns_per_call * 5);
    }
    // 8-thread contended throughput
    {
        PrometheusMetrics m;
        m.set_model("bench");
        std::atomic<bool> go{false};
        std::vector<std::thread> threads;
        std::atomic<long long> total{0};
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&]() {
                while (!go.load()) {}
                long long n = 0;
                const auto t0 = std::chrono::steady_clock::now();
                while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {
                    request_mix(m);
                    ++n;
                }
                total.fetch_add(n);
            });
        }
        go.store(true);
        for (auto &th : threads) th.join();
        std::printf("contended_8t: %lld request-equivalents in 2s = %.0f/s\n", total.load(),
                    total.load() / 2.0);
    }
    return 0;
}
