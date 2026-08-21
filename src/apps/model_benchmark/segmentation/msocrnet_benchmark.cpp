/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: msocrnet_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// msocrnet benchmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/scene_segmentation_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;

namespace {

void save_colorized(const mat_input&, const std_scene_segmentation_output& out, const std::string& image_path) {
    cv::Mat color_seg_result;
    jinq::common::CvUtils::colorize_segmentation_mask(
        out.segmentation_result, color_seg_result, 80);
    std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_msocrnet_result.png";
    std::string output_path = jinq::common::FilePathUtil::concat_path(
        "../demo_data/model_test_input/scene_segmentation", output_file_name);
    cv::imwrite(output_path, color_seg_result);
    LOG(INFO) << "segmentation result image has been written into: " << output_path;
}

}  // namespace

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_scene_segmentation_output> spec;
    spec.model_name = "msocrnet";
    spec.display_name = "msocrnet segmentor";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 100;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(argc, argv, "../demo_data/model_test_input/scene_segmentation/cityscapes_test.png");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(argc, argv, "../demo_data/model_test_input/scene_segmentation/cityscapes_test.png");
    };
    spec.make_model = [](const std::string& model_name) {
        return jinq::factory::scene_segmentation::create_msocrnet_segmentor<mat_input, std_scene_segmentation_output>(model_name);
    };
    spec.handle_output = save_colorized;
    return jinq::apps::run_benchmark(argc, argv, spec);
}
