/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: prometheus_metrics.h
* Date: 26-8-18
************************************************/

// Minimal thread-safe Prometheus metrics collector for model servers.

#ifndef MORTRED_SERVER_PROMETHEUS_METRICS_H
#define MORTRED_SERVER_PROMETHEUS_METRICS_H

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace jinq {
namespace server {

class PrometheusMetrics {
public:
    PrometheusMetrics() {
        http_duration_buckets_ = {5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000};
        queue_wait_buckets_ = {1, 5, 10, 25, 50, 100, 250, 500};
        inference_duration_buckets_ = {1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000};
    }

    void set_model(const std::string& model) {
        std::lock_guard<std::mutex> lock(mutex_);
        model_ = model;
    }

    void inc_http_requests(const std::string& method, const std::string& status) {
        std::lock_guard<std::mutex> lock(mutex_);
        http_requests_[method + "|" + status]++;
    }

    void observe_http_duration_ms(const std::string& method, const std::string& status, double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        observe_histogram_locked(http_duration_, http_duration_buckets_, method + "|" + status, ms);
    }

    void inc_inference_requests(const std::string& status) {
        std::lock_guard<std::mutex> lock(mutex_);
        inference_requests_[status]++;
    }

    void inc_inference_success() {
        std::lock_guard<std::mutex> lock(mutex_);
        inference_success_++;
    }

    void inc_inference_failure() {
        std::lock_guard<std::mutex> lock(mutex_);
        inference_failure_++;
    }

    void observe_queue_wait_ms(double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        observe_histogram_locked(queue_wait_, queue_wait_buckets_, "", ms);
    }

    void observe_inference_duration_ms(double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        observe_histogram_locked(inference_duration_, inference_duration_buckets_, "", ms);
    }

    void set_workers_available(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        workers_available_ = n;
    }

    void set_workers_busy(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        workers_busy_ = n;
    }

    void set_queue_depth(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_depth_ = n;
    }

    void set_waiting_jobs(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        waiting_jobs_ = n;
    }

    void inc_received_jobs() {
        std::lock_guard<std::mutex> lock(mutex_);
        received_jobs_++;
    }

    void inc_finished_jobs() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_jobs_++;
    }

    std::string render() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream ss;
        ss << "# HELP mortred_up Whether the server process is up\n";
        ss << "# TYPE mortred_up gauge\n";
        ss << "mortred_up 1\n";

        ss << "# HELP mortred_ready Whether the server is ready\n";
        ss << "# TYPE mortred_ready gauge\n";
        ss << "mortred_ready " << (ready_ ? 1 : 0) << "\n";

        ss << "# HELP mortred_http_requests_total Total HTTP requests\n";
        ss << "# TYPE mortred_http_requests_total counter\n";
        for (const auto& kv : http_requests_) {
            const auto& method = kv.first.substr(0, kv.first.find('|'));
            const auto& status = kv.first.substr(kv.first.find('|') + 1);
            ss << "mortred_http_requests_total{model=\"" << model_
               << "\",method=\"" << method
               << "\",status=\"" << status << "\"} " << kv.second << "\n";
        }

        ss << "# HELP mortred_http_request_duration_ms HTTP request duration\n";
        ss << "# TYPE mortred_http_request_duration_ms histogram\n";
        render_histogram_locked(ss, "mortred_http_request_duration_ms", http_duration_, http_duration_buckets_);

        ss << "# HELP mortred_inference_requests_total Inference requests\n";
        ss << "# TYPE mortred_inference_requests_total counter\n";
        for (const auto& kv : inference_requests_) {
            ss << "mortred_inference_requests_total{model=\"" << model_
               << "\",status=\"" << kv.first << "\"} " << kv.second << "\n";
        }

        ss << "# HELP mortred_inference_success_total Successful inference requests\n";
        ss << "# TYPE mortred_inference_success_total counter\n";
        ss << "mortred_inference_success_total{model=\"" << model_ << "\"} " << inference_success_ << "\n";

        ss << "# HELP mortred_inference_failure_total Failed inference requests\n";
        ss << "# TYPE mortred_inference_failure_total counter\n";
        ss << "mortred_inference_failure_total{model=\"" << model_ << "\"} " << inference_failure_ << "\n";

        ss << "# HELP mortred_queue_wait_duration_ms Worker queue wait duration\n";
        ss << "# TYPE mortred_queue_wait_duration_ms histogram\n";
        render_histogram_locked(ss, "mortred_queue_wait_duration_ms", queue_wait_, queue_wait_buckets_);

        ss << "# HELP mortred_inference_duration_ms Model inference duration\n";
        ss << "# TYPE mortred_inference_duration_ms histogram\n";
        render_histogram_locked(ss, "mortred_inference_duration_ms", inference_duration_, inference_duration_buckets_);

        ss << "# HELP mortred_workers_available Available workers\n";
        ss << "# TYPE mortred_workers_available gauge\n";
        ss << "mortred_workers_available{model=\"" << model_ << "\"} " << workers_available_ << "\n";

        ss << "# HELP mortred_workers_busy Busy workers\n";
        ss << "# TYPE mortred_workers_busy gauge\n";
        ss << "mortred_workers_busy{model=\"" << model_ << "\"} " << workers_busy_ << "\n";

        ss << "# HELP mortred_queue_depth Current queue depth\n";
        ss << "# TYPE mortred_queue_depth gauge\n";
        ss << "mortred_queue_depth{model=\"" << model_ << "\"} " << queue_depth_ << "\n";

        ss << "# HELP mortred_waiting_jobs Current waiting jobs\n";
        ss << "# TYPE mortred_waiting_jobs gauge\n";
        ss << "mortred_waiting_jobs{model=\"" << model_ << "\"} " << waiting_jobs_ << "\n";

        ss << "# HELP mortred_received_jobs_total Total received jobs\n";
        ss << "# TYPE mortred_received_jobs_total counter\n";
        ss << "mortred_received_jobs_total{model=\"" << model_ << "\"} " << received_jobs_ << "\n";

        ss << "# HELP mortred_finished_jobs_total Total finished jobs\n";
        ss << "# TYPE mortred_finished_jobs_total counter\n";
        ss << "mortred_finished_jobs_total{model=\"" << model_ << "\"} " << finished_jobs_ << "\n";

        return ss.str();
    }

    void set_ready(bool ready) {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_ = ready;
    }

private:
    struct Histogram {
        std::map<std::string, std::vector<uint64_t>> buckets;
        std::map<std::string, double> sum;
        std::map<std::string, uint64_t> count;
    };

    static void observe_histogram_locked(
        Histogram& hist,
        const std::vector<double>& bucket_limits,
        const std::string& label,
        double value) {
        auto& buckets = hist.buckets[label];
        if (buckets.empty()) {
            buckets.resize(bucket_limits.size(), 0);
        }
        for (size_t i = 0; i < bucket_limits.size(); ++i) {
            if (value <= bucket_limits[i]) {
                buckets[i]++;
            }
        }
        hist.sum[label] += value;
        hist.count[label]++;
    }

    void render_histogram_locked(
        std::ostringstream& ss,
        const std::string& metric_name,
        const Histogram& hist,
        const std::vector<double>& bucket_limits) const {
        for (const auto& kv : hist.buckets) {
            const std::string label = kv.first.empty() ? "" : kv.first;
            const std::string label_suffix = label.empty() ? "" : "_" + label;
            for (size_t i = 0; i < kv.second.size(); ++i) {
                ss << metric_name << "_bucket{model=\"" << model_ << "\"";
                if (!label.empty()) {
                    ss << ",method=\"" << label.substr(0, label.find('|')) << "\"";
                    ss << ",status=\"" << label.substr(label.find('|') + 1) << "\"";
                }
                ss << ",le=\"" << bucket_limits[i] << "\"} " << kv.second[i] << "\n";
            }
            ss << metric_name << "_sum{model=\"" << model_ << "\"";
            if (!label.empty()) {
                ss << ",method=\"" << label.substr(0, label.find('|')) << "\"";
                ss << ",status=\"" << label.substr(label.find('|') + 1) << "\"";
            }
            ss << "} " << (hist.sum.count(label) ? hist.sum.at(label) : 0.0) << "\n";
            ss << metric_name << "_count{model=\"" << model_ << "\"";
            if (!label.empty()) {
                ss << ",method=\"" << label.substr(0, label.find('|')) << "\"";
                ss << ",status=\"" << label.substr(label.find('|') + 1) << "\"";
            }
            ss << "} " << (hist.count.count(label) ? hist.count.at(label) : 0) << "\n";
        }
    }

    mutable std::mutex mutex_;
    std::string model_;

    std::map<std::string, uint64_t> http_requests_;
    Histogram http_duration_;
    std::vector<double> http_duration_buckets_;

    std::map<std::string, uint64_t> inference_requests_;
    uint64_t inference_success_ = 0;
    uint64_t inference_failure_ = 0;
    Histogram inference_duration_;
    std::vector<double> inference_duration_buckets_;
    Histogram queue_wait_;
    std::vector<double> queue_wait_buckets_;

    size_t workers_available_ = 0;
    size_t workers_busy_ = 0;
    size_t queue_depth_ = 0;
    size_t waiting_jobs_ = 0;
    uint64_t received_jobs_ = 0;
    uint64_t finished_jobs_ = 0;
    bool ready_ = false;
};

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_PROMETHEUS_METRICS_H
