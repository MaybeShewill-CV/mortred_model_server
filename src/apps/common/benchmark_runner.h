/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: benchmark_runner.h
* Date: 26-8-19
************************************************/

// Shared benchmark main() driver: arg/config handling, model init, warmup,
// timed loop with mean/p50/p99 statistics, and per-task input/output hooks.
#ifndef MORTRED_APPS_BENCHMARK_RUNNER_H
#define MORTRED_APPS_BENCHMARK_RUNNER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace apps {

template<typename INPUT, typename OUTPUT>
struct BenchmarkSpec {
    std::string model_name;        // factory registration key (informational)
    std::string display_name;      // e.g. "mobilenetv2 classifier"
    std::string usage;             // e.g. "exe config_file_path [test_image_path]"
    int loops = 100;
    // each diffusion-family run is a full sampling pass (seconds to minutes);
    // keep warmup off to match the original no-warmup behavior; standard light
    // models keep the default true
    bool warmup = true;
    // CLI shape check: standard mains use standard_args_ok; diffusion mains
    // override per their own arg counts
    std::function<bool(int argc)> args_ok;
    // input validity check: standard single-image specs check
    // !input_image.empty(); diffusion/lightglue custom inputs return true
    // (make_input guarantees it)
    std::function<bool(const INPUT&)> input_ok;
    // build the model input from CLI + config; standard single-image mains use
    // make_single_image_input
    std::function<INPUT(int argc, char** argv, const toml::table& cfg)> make_input;
    // input image path for output naming/visualization ("" when the benchmark
    // has no image)
    std::function<std::string(int argc, char** argv)> image_path_of;
    // model construction: factory create_X or direct make_unique (diffusion)
    std::function<std::unique_ptr<jinq::models::BaseAiModel<INPUT, OUTPUT>>(const std::string&)>
        make_model;
    // called once after the loop: logging / visualization / saving
    std::function<void(const INPUT& in, const OUTPUT& out, const std::string& image_path)>
        handle_output;
};

inline bool standard_args_ok(int argc) {
    return argc == 2 || argc == 3;
}

inline std::string standard_image_path(int argc, char** argv, const std::string& default_path) {
    if (argc >= 3) {
        LOG(INFO) << "input test image path: " << argv[2];
        return argv[2];
    }
    LOG(INFO) << "use default input test image path: " << default_path;
    return default_path;
}

inline jinq::models::io_define::common_io::mat_input make_single_image_input(
    int argc, char** argv, const std::string& default_path,
    int imread_flags = cv::IMREAD_COLOR) {
    const std::string path = standard_image_path(argc, argv, default_path);
    if (!jinq::common::FilePathUtil::is_file_exist(path)) {
        LOG(INFO) << "test input image file: " << path << " not exist";
        return jinq::models::io_define::common_io::mat_input{};
    }
    jinq::models::io_define::common_io::mat_input input;
    input.input_image = cv::imread(path, imread_flags);
    if (input.input_image.empty()) {
        LOG(ERROR) << "image decode failed: " << path;
    }
    return input;
}

template<typename INPUT, typename OUTPUT>
int run_benchmark(int argc, char** argv, const BenchmarkSpec<INPUT, OUTPUT>& spec) {
    if (!spec.args_ok(argc)) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << spec.usage;
        return -1;
    }
    const std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;
    if (!jinq::common::FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }
    auto cfg_parsed = toml::parse_file(cfg_file_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: "
                   << std::string(cfg_parsed.error().description());
        return -1;
    }
    auto cfg = std::move(cfg_parsed).table();

    // matches the original behavior: exit directly when the input is missing or
    // fails to decode
    INPUT model_input = spec.make_input(argc, argv, cfg);
    if (!spec.input_ok(model_input)) {
        return -1;
    }
    OUTPUT model_output{};

    auto model = spec.make_model(spec.model_name);
    model->init(cfg);
    if (!model->is_successfully_initialized()) {
        LOG(INFO) << spec.display_name << " init failed";
        return -1;
    }

    LOG(INFO) << "start " << spec.display_name << " benchmark at: "
              << jinq::common::Timestamp::now().to_format_str();

    // warmup run: not counted (removes first-iteration cold-start bias)
    if (spec.warmup) {
        model->run(model_input, model_output);
    }

    std::vector<double> iter_ms(static_cast<size_t>(spec.loops));
    auto ts = jinq::common::Timestamp::now();
    for (int i = 0; i < spec.loops; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        model->run(model_input, model_output);
        const auto t1 = std::chrono::steady_clock::now();
        iter_ms[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    auto cost_time = jinq::common::Timestamp::now() - ts;

    std::sort(iter_ms.begin(), iter_ms.end());
    const double mean_ms = std::accumulate(iter_ms.begin(), iter_ms.end(), 0.0) / iter_ms.size();
    const double p50_ms = iter_ms[iter_ms.size() / 2];
    const double p99_ms = iter_ms[static_cast<size_t>(
        std::ceil(0.99 * static_cast<double>(iter_ms.size())) - 1)];
    LOG(INFO) << "benchmark ends at: " << jinq::common::Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << spec.loops / cost_time
              << ", mean: " << mean_ms << " ms, p50: " << p50_ms
              << " ms, p99: " << p99_ms << " ms";

    spec.handle_output(model_input, model_output, spec.image_path_of(argc, argv));
    return 0;
}

}  // namespace apps
}  // namespace jinq

#endif  // MORTRED_APPS_BENCHMARK_RUNNER_H
