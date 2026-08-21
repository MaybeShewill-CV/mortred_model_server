/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: lightglue_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// lightglue benckmark tool

#include "apps/common/benchmark_runner.h"
#include "models/feature_point/lightglue.h"

using jinq::models::io_define::common_io::pair_mat_input;
using jinq::models::io_define::feature_point::std_feature_point_match_output;

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<pair_mat_input, std_feature_point_match_output> spec;
    spec.model_name = "lightglue";
    spec.display_name = "lightglue matcher";
    spec.usage = "exe config_file_path [src_image_path dst_image_path]";
    spec.loops = 100;
    spec.args_ok = [](int argc) { return argc == 2 || argc == 4; };
    spec.input_ok = [](const pair_mat_input& in) {
        return !in.src_input_image.empty() && !in.dst_input_image.empty();
    };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        const char* src = "../demo_data/model_test_input/feature_point/match_test_01.jpg";
        const char* dst = "../demo_data/model_test_input/feature_point/match_test_02.jpg";
        if (argc == 4) {
            src = argv[2];
            dst = argv[3];
        }
        pair_mat_input in;
        in.src_input_image = cv::imread(src, cv::IMREAD_COLOR);
        in.dst_input_image = cv::imread(dst, cv::IMREAD_COLOR);
        return in;
    };
    spec.image_path_of = [](int argc, char** argv) {
        return std::string(argc == 4 ? argv[2]
                                     : "../demo_data/model_test_input/feature_point/match_test_01.jpg");
    };
    spec.make_model = [](const std::string&) {
        return std::unique_ptr<
            jinq::models::BaseAiModel<pair_mat_input, std_feature_point_match_output>>(
            new jinq::models::feature_point::LightGlue<pair_mat_input,
                                                       std_feature_point_match_output>());
    };
    spec.handle_output = [](const pair_mat_input& in,
                            const std_feature_point_match_output& out,
                            const std::string& image_path) {
        cv::Mat vis_result;
        jinq::common::CvUtils::visualize_fp_match_result(
            in.src_input_image, in.dst_input_image, out, vis_result);
        std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
        output_file_name =
            output_file_name.substr(0, output_file_name.find_last_of('.')) + "_lightglue_result.png";
        std::string output_path = jinq::common::FilePathUtil::concat_path(
            "../demo_data/model_test_input/feature_point", output_file_name);
        cv::imwrite(output_path, vis_result);
        LOG(INFO) << "feature point match result image has been written into: " << output_path;
    };
    return jinq::apps::run_benchmark(argc, argv, spec);
}
