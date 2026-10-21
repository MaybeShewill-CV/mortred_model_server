/************************************************
 * Author: Codex
 * File: prometheus_metrics_test.cc
 * Date: 2026-08-26
 ************************************************/

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include <atomic>
#include <gtest/gtest.h>

#include "common/status_code.h"
#include "server/prometheus_metrics.h"

using mortred::common::StatusCode;
using mortred::common::to_underlying;
using mortred::server::PrometheusMetrics;

namespace {

/** Distinct values rendered for a given label name, sorted for deterministic comparison. */
std::vector<std::string> rendered_label_values(const std::string &text, const std::string &label_name) {
    std::vector<std::string> values;
    const std::string prefix = label_name + "=\"";
    size_t pos = text.find(prefix);
    while (pos != std::string::npos) {
        const size_t value_begin = pos + prefix.size();
        const size_t value_end = text.find('"', value_begin);
        if (value_end == std::string::npos) {
            break;
        }
        const std::string value = text.substr(value_begin, value_end - value_begin);
        if (std::find(values.begin(), values.end(), value) == values.end()) {
            values.push_back(value);
        }
        pos = text.find(prefix, value_end);
    }
    std::sort(values.begin(), values.end());
    return values;
}

bool any_rendered_label_value_has_char(const std::string &text, const std::string &label_name, char needle) {
    for (const auto &value : rendered_label_values(text, label_name)) {
        if (value.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(prometheus_metrics, counters_and_gauges) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.inc_http_requests("POST", "200");
    m.inc_http_requests("POST", "200");
    m.inc_http_requests("POST", "400");
    m.inc_inference_success();
    m.inc_inference_failure();
    m.inc_inference_requests(to_underlying(StatusCode::MODEL_OUTPUT_CONTRACT_FAILED));
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
    EXPECT_NE(text.find("mortred_http_request_duration_ms_bucket{model=\"resnet\",method=\"POST\",status=\"200\",le=\"+Inf\"} 2"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_queue_wait_duration_ms_count"), std::string::npos);
    EXPECT_NE(text.find("mortred_inference_duration_ms_count"), std::string::npos);
}

TEST(prometheus_metrics, histogram_plus_inf_matches_count_and_includes_overflow) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.observe_inference_duration_ms(10.0);
    m.observe_inference_duration_ms(10000.0);

    auto text = m.render();
    EXPECT_NE(text.find("mortred_inference_duration_ms_bucket{model=\"resnet\",le=\"5000\"} 1"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_inference_duration_ms_bucket{model=\"resnet\",le=\"+Inf\"} 2"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_inference_duration_ms_count{model=\"resnet\"} 2"), std::string::npos);
}

TEST(prometheus_metrics, contract_failure_counter_uses_status_code_not_string_six) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.inc_inference_requests(0);
    m.inc_inference_requests(to_underlying(StatusCode::MODEL_OUTPUT_CONTRACT_FAILED));

    auto text = m.render();
    EXPECT_NE(text.find("mortred_inference_requests_total{model=\"resnet\",status=\"0\"} 1"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_inference_requests_total{model=\"resnet\",status=\"6\"} 1"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_model_output_contract_failures_total{model=\"resnet\"} 1"),
              std::string::npos);
}

TEST(prometheus_metrics, batch_window_wait_histogram_includes_plus_inf) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.observe_batch_window_wait_ms(10.0);
    m.observe_batch_window_wait_ms(1000.0);

    auto text = m.render();
    EXPECT_NE(text.find("mortred_batch_window_wait_ms_bucket{model=\"resnet\",le=\"500\"} 1"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_batch_window_wait_ms_bucket{model=\"resnet\",le=\"+Inf\"} 2"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_batch_window_wait_ms_count{model=\"resnet\"} 2"), std::string::npos);
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


TEST(prometheus_metrics, ready_independent_of_workers_available) {
    // P0-1: LB probes use mortred_ready; saturation uses worker gauges.
    PrometheusMetrics m;
    m.set_model("yolov8");
    m.set_ready(true);
    m.set_workers_available(0);
    m.set_workers_busy(4);
    auto text = m.render();
    EXPECT_NE(text.find("mortred_ready 1"), std::string::npos);
    EXPECT_NE(text.find("mortred_workers_available{model=\"yolov8\"} 0"), std::string::npos);
    EXPECT_NE(text.find("mortred_workers_busy{model=\"yolov8\"} 4"), std::string::npos);
}

// P1-6: the request method is unauthenticated input; it must not be able to grow the label cardinality.

TEST(prometheus_metrics, unknown_methods_collapse_to_a_single_other_series) {
    PrometheusMetrics m;
    m.set_model("resnet");

    std::vector<std::string> unknown = {"TRACE",  "CONNECT", "PROPFIND", "BREW",   "LOCK", "UNLOCK",
                                        "MKCOL",  "COPY",    "MOVE",     "SEARCH", "REPORT"};
    unknown.push_back("VERYVERYLONGMETHODNAME"); // > 16 chars
    unknown.push_back("BAD\r\nMETHOD");          // control characters
    const size_t calls_per_method = 2;
    for (const auto &method : unknown) {
        for (size_t i = 0; i < calls_per_method; ++i) {
            m.inc_http_requests(method, "200");
        }
    }

    auto text = m.render();
    const auto methods = rendered_label_values(text, "method");
    ASSERT_EQ(methods.size(), 1u);
    EXPECT_EQ(methods[0], "other");
    EXPECT_NE(text.find("mortred_http_requests_total{model=\"resnet\",method=\"other\",status=\"200\"} " +
                        std::to_string(unknown.size() * calls_per_method)),
              std::string::npos);
    EXPECT_FALSE(any_rendered_label_value_has_char(text, "method", '|'));
    for (const auto &method : unknown) {
        EXPECT_EQ(text.find("method=\"" + method + "\""), std::string::npos) << "leaked method: " << method;
    }
}

TEST(prometheus_metrics, mixed_case_known_methods_fold_to_uppercase_not_other) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.inc_http_requests("post", "200");
    m.inc_http_requests("PoSt", "200");
    m.inc_http_requests("  POST  ", "200");

    auto text = m.render();
    EXPECT_NE(text.find("mortred_http_requests_total{model=\"resnet\",method=\"POST\",status=\"200\"} 3"),
              std::string::npos);
    EXPECT_EQ(text.find("method=\"other\""), std::string::npos);
    EXPECT_EQ(text.find("method=\"post\""), std::string::npos);
    EXPECT_EQ(text.find("method=\"PoSt\""), std::string::npos);
}

TEST(prometheus_metrics, method_containing_pipe_cannot_inject_label_values) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.inc_http_requests("A|B", "200");
    m.inc_http_requests("GET|500", "200");
    m.inc_http_requests("|other|", "200");

    auto text = m.render();
    EXPECT_NE(text.find("mortred_http_requests_total{model=\"resnet\",method=\"other\",status=\"200\"} 3"),
              std::string::npos);
    EXPECT_EQ(text.find("method=\"A\""), std::string::npos);
    EXPECT_EQ(text.find("method=\"GET\""), std::string::npos);
    EXPECT_EQ(text.find("method=\"\""), std::string::npos);
    EXPECT_FALSE(any_rendered_label_value_has_char(text, "method", '|'));
    EXPECT_FALSE(any_rendered_label_value_has_char(text, "status", '|'));
    EXPECT_EQ(rendered_label_values(text, "method"), std::vector<std::string>{"other"});
    EXPECT_EQ(rendered_label_values(text, "status"), std::vector<std::string>{"200"});
}

TEST(prometheus_metrics, histogram_method_label_normalizes_like_the_counter) {
    PrometheusMetrics m;
    m.set_model("resnet");
    m.observe_http_duration_ms("TRACE", "200", 5.0);
    m.observe_http_duration_ms("trace", "200", 7.0);
    m.observe_http_duration_ms("tr|ace", "200", 9.0);
    m.observe_http_duration_ms("PoSt", "200", 11.0);

    auto text = m.render();
    EXPECT_NE(text.find("mortred_http_request_duration_ms_bucket{model=\"resnet\",method=\"POST\",status=\"200\",le=\"+Inf\"} 1"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_http_request_duration_ms_count{model=\"resnet\",method=\"other\",status=\"200\"} 3"),
              std::string::npos);
    EXPECT_NE(text.find("mortred_http_request_duration_ms_count{model=\"resnet\",method=\"POST\",status=\"200\"} 1"),
              std::string::npos);
    EXPECT_FALSE(any_rendered_label_value_has_char(text, "method", '|'));
    EXPECT_EQ(rendered_label_values(text, "method"), (std::vector<std::string>{"POST", "other"}));
    EXPECT_EQ(rendered_label_values(text, "status"), std::vector<std::string>{"200"});
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(PrometheusMetricsTest, waiting_depth_bound_to_atomic_shows_live_value) {
    mortred::server::PrometheusMetrics m;
    m.set_model("demo");
    m.set_queue_depth(7);
    m.set_waiting_jobs(7);
    std::atomic<size_t> live{0};
    m.bind_waiting_depth(&live);
    const std::string idle = m.render();
    EXPECT_NE(idle.find("mortred_waiting_jobs{model=\"demo\"} 0"), std::string::npos);
    EXPECT_NE(idle.find("mortred_queue_depth{model=\"demo\"} 0"), std::string::npos);
    live.store(3);
    const std::string busy = m.render();
    EXPECT_NE(busy.find("mortred_waiting_jobs{model=\"demo\"} 3"), std::string::npos);
    EXPECT_NE(busy.find("mortred_queue_depth{model=\"demo\"} 3"), std::string::npos);
}
