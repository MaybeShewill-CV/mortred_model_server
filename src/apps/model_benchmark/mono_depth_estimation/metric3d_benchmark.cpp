/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: metric3d_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// metric3d benchmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/mono_depth_estimate_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::mono_depth_estimation::std_mde_output;

namespace {

void save_depth(const mat_input&, const std_mde_output& out, const std::string& image_path) {
    std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) +
                       "_metric3d_colorized_depth_result.png";
    std::string output_path = jinq::common::FilePathUtil::concat_path(
        "../demo_data/model_test_input/mono_depth_estimation", output_file_name);
    cv::imwrite(output_path, out.colorized_depth_map);
    LOG(INFO) << "prediction colorized depth image has been written into: " << output_path;

    output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) +
                       "_metric3d_depth_result.yaml";
    output_path = jinq::common::FilePathUtil::concat_path("../demo_data/model_test_input/mono_depth_estimation", output_file_name);
    cv::FileStorage out_depth_map;
    out_depth_map.open(output_path, cv::FileStorage::WRITE);
    out_depth_map.write("depth_map", out.depth_map);
    LOG(INFO) << "prediction depth map has been written into: " << output_path;

    output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) +
                       "_metric3d_conf_result.yaml";
    output_path = jinq::common::FilePathUtil::concat_path("../demo_data/model_test_input/mono_depth_estimation", output_file_name);
    cv::FileStorage out_conf_map;
    out_conf_map.open(output_path, cv::FileStorage::WRITE);
    out_conf_map.write("confidence_map", out.confidence_map);
    LOG(INFO) << "prediction confidence map has been written into: " << output_path;
}

}  // namespace

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_mde_output> spec;
    spec.model_name = "metric3d";
    spec.display_name = "metric3d estimator";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 10;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(argc, argv, "../demo_data/model_test_input/mono_depth_estimation/0000000005.png");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(argc, argv, "../demo_data/model_test_input/mono_depth_estimation/0000000005.png");
    };
    spec.make_model = [](const std::string& model_name) {
        return jinq::factory::mono_depth_estimation::create_metric3d_estimator<mat_input, std_mde_output>(model_name);
    };
    spec.handle_output = save_depth;
    return jinq::apps::run_benchmark(argc, argv, spec);
}
