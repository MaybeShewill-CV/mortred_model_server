/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: modnet_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// modnet benchmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/matting_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::matting::std_matting_output;

namespace {

void save_matting(const mat_input&, const std_matting_output& out, const std::string& image_path) {
    std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_modnet_result.png";
    std::string output_path = jinq::common::FilePathUtil::concat_path(
        "../demo_data/model_test_input/matting", output_file_name);
    cv::imwrite(output_path, out.matting_result);
    LOG(INFO) << "segmentation result image has been written into: " << output_path;
}

}  // namespace

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_matting_output> spec;
    spec.model_name = "modnet";
    spec.display_name = "modnet segmentor";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 100;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(argc, argv, "../demo_data/model_test_input/matting/matting_test.jpg");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(argc, argv, "../demo_data/model_test_input/matting/matting_test.jpg");
    };
    spec.make_model = [](const std::string& model_name) {
        return jinq::factory::matting::create_modnet_segmentor<mat_input, std_matting_output>(model_name);
    };
    spec.handle_output = save_matting;
    return jinq::apps::run_benchmark(argc, argv, spec);
}
