/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: dinov2_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// dinov2 benchmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/feature_embedding_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::feature_embedding::std_feature_embedding_output;

namespace {

void log_embedding(const mat_input&, const std_feature_embedding_output& out, const std::string&) {
    LOG(INFO) << "dinov2 image embedding features dims: " << out.embedding.size();
    LOG(INFO) << "dinov2 image embedding features:"
              << " " << out.embedding[0] << " " << out.embedding[1] << " "
              << out.embedding[2] << " " << out.embedding[3] << " " << out.embedding[4];
}

}  // namespace

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_feature_embedding_output> spec;
    spec.model_name = "dinov2";
    spec.display_name = "dinov2 feature extractor";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 100;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(argc, argv, "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(argc, argv, "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    };
    spec.make_model = [](const std::string& model_name) {
        return jinq::factory::feature_embedding::create_dinov2_feature_extractor<mat_input, std_feature_embedding_output>(model_name);
    };
    spec.handle_output = log_embedding;
    return jinq::apps::run_benchmark(argc, argv, spec);
}
