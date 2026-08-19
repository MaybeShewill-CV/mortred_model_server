/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * File: ddim_sampler_benchmark.cpp
 * Date: 2026-08-19
 ************************************************/

// ddim-sampler benchmark tool

#include "apps/common/benchmark_runner.h"
#include "models/diffusion/ddim_sampler.h"

using jinq::models::io_define::diffusion::std_ddim_input;
using jinq::models::io_define::diffusion::std_ddim_output;

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<std_ddim_input, std_ddim_output> spec;
    spec.model_name = "ddim";
    spec.display_name = "ddim sampler";
    spec.usage =
        "exe config_file_path [sample_size(default: 256)] [sample_steps(default: 10)] "
        "[save_all_mid_results(default: true)]";
    spec.loops = 1;
    spec.warmup = false;
    spec.args_ok = [](int argc) { return argc >= 2 && argc <= 5; };
    spec.input_ok = [](const std_ddim_input&) { return true; };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        int sample_size = 256;
        if (argc >= 3) {
            sample_size = std::stoi(argv[2]);
        }
        int sample_steps = 10;
        if (argc >= 4) {
            sample_steps = std::stoi(argv[3]);
        }
        bool save_all_mid_results = true;
        if (argc >= 5) {
            save_all_mid_results = std::stoi(argv[4]) == 1;
        }
        std_ddim_input in;
        in.sample_size = cv::Size(sample_size, sample_size);
        in.total_steps = 1000;
        in.sample_steps = sample_steps;
        in.channels = 3;
        in.save_all_mid_results = save_all_mid_results;
        in.eta = 1.0f;
        return in;
    };
    spec.image_path_of = [](int, char**) {
        return std::string("../demo_data/model_test_input/diffusion/ddim");
    };
    spec.make_model = [](const std::string&) {
        return std::unique_ptr<jinq::models::BaseAiModel<std_ddim_input, std_ddim_output>>(
            new jinq::models::diffusion::DDIMSampler<std_ddim_input, std_ddim_output>());
    };
    spec.handle_output = [](const std_ddim_input& in, const std_ddim_output& out,
                            const std::string& save_dir) {
        if (in.save_all_mid_results) {
            auto stacked_sampled_image =
                jinq::common::CvUtils::stack_multiple_ddpm_images(out.sampled_images);
            auto stacked_predict_x0_image =
                jinq::common::CvUtils::stack_multiple_ddpm_images(out.predicted_x0);
            cv::imwrite(jinq::common::FilePathUtil::concat_path(
                            save_dir, "stacked_sampled_image.png"),
                        stacked_sampled_image);
            cv::imwrite(jinq::common::FilePathUtil::concat_path(
                            save_dir, "stacked_predict_x0_image.png"),
                        stacked_predict_x0_image);
        } else {
            cv::imwrite(jinq::common::FilePathUtil::concat_path(save_dir, "sample-step-0.png"),
                        out.sampled_images[0]);
            cv::imwrite(jinq::common::FilePathUtil::concat_path(
                            save_dir, "predict_x0-step-0.png"),
                        out.predicted_x0[0]);
        }
        LOG(INFO) << "sampled result image has been written into: " << save_dir;
    };
    return jinq::apps::run_benchmark(argc, argv, spec);
}