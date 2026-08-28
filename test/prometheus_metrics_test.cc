/************************************************
 * Author: Codex
 * File: prometheus_metrics_test.cc
 * Date: 2026-08-26
 ************************************************/

#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "server/prometheus_metrics.h"

using jinq::server::PrometheusMetrics;

TEST(prometheus_metrics, counters_and_gauges) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.inc_http_requests("POST", "200");
    m.inc_http_requests("POST", "200");
    m.inc_http_requests("POST", "400");
    m.inc_inference_success();
    m.inc_inference_failure();
    m.inc_inference_requests("6");
    m.set_workers_available(3);
    m.set_workers_busy(1);
    m.set_queue_depth(2);
    m.set_waiting_jobs(2);
    m.inc_received_jobs();
    m.inc_finished_jobs();

    auto text = m.render();
    EXPECT_NE(text.find("mortred_http_requests_total{model=\"resnet\",method=\"POST\",status=\"200\"} 2"), std::string::npos);
    EXPECT_NE(text.find("mortred_http_requests_total{model=\"resnet\",method=\"POST\",status=\"400\"} 1"), std::string::npos);
    EXPECT_NE(text.find("mortred_inference_success_total{model=\"resnet\"} 1"), std::string::npos);
    EXPECT_NE(text.find("mortred_inference_failure_total{model=\"resnet\"} 1"), std::string::npos);
    EXPECT_NE(text.find("mortred_model_output_contract_failures_total{model=\"resnet\"} 1"), std::string::npos);
    EXPECT_NE(text.find("mortred_workers_available{model=\"resnet\"} 3"), std::string::npos);
    EXPECT_NE(text.find("mortred_workers_busy{model=\"resnet\"} 1"), std::string::npos);
    EXPECT_NE(text.find("mortred_queue_depth{model=\"resnet\"} 2"), std::string::npos);
}

TEST(prometheus_metrics, histograms) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.observe_http_duration_ms("POST", "200", 5.0);
    m.observe_http_duration_ms("POST", "200", 50.0);
    m.observe_queue_wait_ms(2.0);
    m.observe_inference_duration_ms(10.0);

    auto text = m.render();
    EXPECT_NE(text.find("mortred_http_request_duration_ms_count"), std::string::npos);
    EXPECT_NE(text.find("mortred_queue_wait_duration_ms_count"), std::string::npos);
    EXPECT_NE(text.find("mortred_inference_duration_ms_count"), std::string::npos);
}

TEST(prometheus_metrics, concurrent_updates_are_safe) {
    PrometheusMetrics m;
    m.set_model("resnet");
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&m]() {
            for (int j = 0; j < 1000; ++j) {
                m.inc_http_requests("POST", "200");
                m.observe_inference_duration_ms(1.0);
                m.set_queue_depth(j % 10);
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    auto text = m.render();
    EXPECT_NE(text.find("mortred_http_requests_total{model=\"resnet\",method=\"POST\",status=\"200\"} 4000"), std::string::npos);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
