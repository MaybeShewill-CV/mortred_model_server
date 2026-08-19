/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: dbnet_benchmark.cpp
* Date: 2026-08-19
************************************************/

// dbnet benchmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/ocr_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::ocr::std_text_regions_output;

namespace {

void vis_and_save(const mat_input& in, const std_text_regions_output& out, const std::string& image_path) {
    cv::Mat vis_image = in.input_image.clone();
    jinq::common::CvUtils::vis_text_detection(vis_image, out);
    std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_dbnet_result.png";
    std::string output_path = jinq::common::FilePathUtil::concat_path(
        "../demo_data/model_test_input/ocr", output_file_name);
    cv::imwrite(output_path, vis_image);
    LOG(INFO) << "detection result image has been written into: " << output_path;
}

}  // namespace

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_text_regions_output> spec;
    spec.model_name = "dbnet";
    spec.display_name = "dbnet detector";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 100;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(argc, argv, "../demo_data/model_test_input/ocr/railway_ticket.png");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(argc, argv, "../demo_data/model_test_input/ocr/railway_ticket.png");
    };
    spec.make_model = [](const std::string& model_name) {
        return jinq::factory::ocr::create_dbtext_detector<mat_input, std_text_regions_output>(model_name);
    };
    spec.handle_output = vis_and_save;
    return jinq::apps::run_benchmark(argc, argv, spec);
}
