/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: mobilenetv2_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// mobilenetv2 benckmark tool

#include <chrono>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "apps/common/benchmark_runner.h"
#include "factory/classification_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::classification::std_classification_output;

int main(int argc, char** argv) {
    // optional batch mode: exe config [image] --batch N
    // (run_batch with N copies of the input; reports ms/batch + img/s)
    int batch_n = 0;
    for (int i = 2; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--batch") == 0) {
            batch_n = std::atoi(argv[i + 1]);
        }
    }
    if (batch_n >= 1) {
        if (argc < 2) {
            LOG(ERROR) << "usage: exe config_file_path [test_image_path] --batch N";
            return -1;
        }
        jinq::apps::BenchmarkSpec<mat_input, std_classification_output> spec;
        spec.model_name = "mobilenetv2";
        spec.display_name = "mobilenetv2 classifier (batch)";
        spec.make_input = [](int argc, char** argv, const toml::table&) {
            return jinq::apps::make_single_image_input(
                argc, argv,
                "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
        };
        spec.make_model = [](const std::string& name) {
            return jinq::factory::classification::create_mobilenetv2_classifier<
                mat_input, std_classification_output>(name);
        };
        auto cfg_parsed = toml::parse_file(argv[1]);
        if (!cfg_parsed) {
            LOG(ERROR) << "parse toml config file failed";
            return -1;
        }
        auto cfg = std::move(cfg_parsed).table();
        mat_input input = spec.make_input(argc, argv, cfg);
        if (input.input_image.empty()) {
            LOG(ERROR) << "input image is empty";
            return -1;
        }
        auto model = spec.make_model(spec.model_name);
        model->init(cfg);
        if (!model->is_successfully_initialized()) {
            LOG(ERROR) << "model init failed";
            return -1;
        }
        const std::vector<mat_input> batch_inputs(
            static_cast<size_t>(batch_n), mat_input{input.input_image});
        std::vector<std_classification_output> batch_outputs;
        model->run_batch(batch_inputs, batch_outputs);  // warmup
        constexpr int kLoops = 200;
        std::vector<double> iter_ms(kLoops);
        const auto t_all = jinq::common::Timestamp::now();
        for (int i = 0; i < kLoops; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto status = model->run_batch(batch_inputs, batch_outputs);
            const auto t1 = std::chrono::steady_clock::now();
            if (status != jinq::common::StatusCode::OK) {
                LOG(ERROR) << "run_batch failed: "
                           << jinq::common::to_underlying(status);
                return -1;
            }
            iter_ms[static_cast<size_t>(i)] =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        const double cost = jinq::common::Timestamp::now() - t_all;
        std::sort(iter_ms.begin(), iter_ms.end());
        const double mean_ms =
            std::accumulate(iter_ms.begin(), iter_ms.end(), 0.0) / iter_ms.size();
        LOG(INFO) << "batch=" << batch_n << " loops=" << kLoops
                  << " cost=" << cost << "s batch/s=" << kLoops / cost
                  << " img/s=" << (static_cast<double>(kLoops) * batch_n) / cost
                  << " mean_batch_ms=" << mean_ms
                  << " mean_img_ms=" << mean_ms / batch_n;
        return 0;
    }

    jinq::apps::BenchmarkSpec<mat_input, std_classification_output> spec;
    spec.model_name = "mobilenetv2";
    spec.display_name = "mobilenetv2 classifier";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 1000;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(
            argc, argv,
            "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(
            argc, argv,
            "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    };
    spec.make_model = [](const std::string& name) {
        return jinq::factory::classification::create_mobilenetv2_classifier<mat_input,
                                                                            std_classification_output>(
            name);
    };
    spec.handle_output =
        [](const mat_input&, const std_classification_output& out, const std::string&) {
            LOG(INFO) << "classify id: " << out.class_id;
            auto max_score = std::max_element(out.scores.begin(), out.scores.end());
            LOG(INFO) << "max classify socre: " << *max_score;
            LOG(INFO) << "max classify id: "
                      << static_cast<int>(std::distance(out.scores.begin(), max_score));
        };
    return jinq::apps::run_benchmark(argc, argv, spec);
}
