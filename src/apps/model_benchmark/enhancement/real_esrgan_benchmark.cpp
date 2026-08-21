/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: real_esrgan_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// real_esrgan benchmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/enhancement_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::enhancement::std_enhancement_output;

namespace {

void save_enhanced(const mat_input&, const std_enhancement_output& out, const std::string& image_path) {
    std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_real-esrgan_result.png";
    std::string output_path = jinq::common::FilePathUtil::concat_path(
        "../demo_data/model_test_input/enhancement/real_esr", output_file_name);
    cv::imwrite(output_path, out.enhancement_result);
    LOG(INFO) << "enhancement result image has been written into: " << output_path;
}

}  // namespace

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_enhancement_output> spec;
    spec.model_name = "real-esrgan";
    spec.display_name = "real-esrgan enhancementor";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 100;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(argc, argv, "../demo_data/model_test_input/enhancement/real_esr/test.jpg");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(argc, argv, "../demo_data/model_test_input/enhancement/real_esr/test.jpg");
    };
    spec.make_model = [](const std::string& model_name) {
        return jinq::factory::enhancement::create_realesrgan_enhancementor<mat_input, std_enhancement_output>(model_name);
    };
    spec.handle_output = save_enhanced;
    return jinq::apps::run_benchmark(argc, argv, spec);
}
