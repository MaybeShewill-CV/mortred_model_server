/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: densenet_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// densenet benchmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/classification_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::classification::std_classification_output;

namespace {

void log_classification(const mat_input&, const std_classification_output& out, const std::string&) {
    LOG(INFO) << "classify id: " << out.class_id;
    auto max_score = std::max_element(out.scores.begin(), out.scores.end());
    LOG(INFO) << "max classify socre: " << *max_score;
    LOG(INFO) << "max classify id: "
              << static_cast<int>(std::distance(out.scores.begin(), max_score));
}

}  // namespace

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_classification_output> spec;
    spec.model_name = "densenet";
    spec.display_name = "densenet classifier";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 1000;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(argc, argv, "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(argc, argv, "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    };
    spec.make_model = [](const std::string& model_name) {
        return jinq::factory::classification::create_densenet_classifier<mat_input, std_classification_output>(model_name);
    };
    spec.handle_output = log_classification;
    return jinq::apps::run_benchmark(argc, argv, spec);
}
