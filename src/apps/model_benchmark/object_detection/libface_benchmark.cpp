/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: libface_benchmark.cpp
* Date: 2026-08-19
************************************************/

// libface benchmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/obj_detection_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::object_detection::std_face_detection_output;

namespace {

void vis_and_save(const mat_input& in, const std_face_detection_output& out, const std::string& image_path) {
    cv::Mat vis_image = in.input_image.clone();
    jinq::common::CvUtils::vis_object_detection(vis_image, out, 80);
    std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_libface_result.png";
    std::string output_path = jinq::common::FilePathUtil::concat_path(
        "../demo_data/model_test_input/object_detection", output_file_name);
    cv::imwrite(output_path, vis_image);
    LOG(INFO) << "detection result image has been written into: " << output_path;
}

}  // namespace

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_face_detection_output> spec;
    spec.model_name = "libface";
    spec.display_name = "libface detector";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 100;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(argc, argv, "../demo_data/model_test_input/object_detection/face_wo_mask.jpg");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(argc, argv, "../demo_data/model_test_input/object_detection/face_wo_mask.jpg");
    };
    spec.make_model = [](const std::string& model_name) {
        return jinq::factory::object_detection::create_libface_detector<mat_input, std_face_detection_output>(model_name);
    };
    spec.handle_output = vis_and_save;
    return jinq::apps::run_benchmark(argc, argv, spec);
}
