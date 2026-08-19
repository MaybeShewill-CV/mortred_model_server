/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * File: ddpm_sampler_benchmark.cpp
 * Date: 2026-08-19
 ************************************************/

// ddpm-sampler benchmark tool

#include "apps/common/benchmark_runner.h"
#include "models/diffusion/ddpm_sampler.h"

using jinq::models::io_define::diffusion::std_ddpm_input;
using jinq::models::io_define::diffusion::std_ddpm_output;

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<std_ddpm_input, std_ddpm_output> spec;
    spec.model_name = "ddpm";
    spec.display_name = "ddpm sampler";
    spec.usage =
        "exe config_file_path [save_dir] [save_all_mid_results(default: false)] "
        "[use_fixed_noise(default: false)]";
    spec.loops = 1;
    spec.warmup = false;
    spec.args_ok = [](int argc) { return argc >= 2 && argc <= 5; };
    spec.input_ok = [](const std_ddpm_input&) { return true; };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        bool save_all_mid_results = false;
        if (argc >= 4) {
            save_all_mid_results = std::stoi(argv[3]) == 1;
        }
        bool use_fixed_noise = false;
        if (argc >= 5) {
            use_fixed_noise = std::stoi(argv[4]) == 1;
        }
        std_ddpm_input in;
        in.sample_size = cv::Size(128, 128);
        in.timestep = 1000;
        in.channels = 3;
        in.save_all_mid_results = save_all_mid_results;
        in.use_fixed_noise_for_psample = use_fixed_noise;
        return in;
    };
    spec.image_path_of = [](int argc, char** argv) {
        return std::string(argc >= 3 ? argv[2]
                                     : "../demo_data/model_test_input/diffusion/ddpm");
    };
    spec.make_model = [](const std::string&) {
        return std::unique_ptr<jinq::models::BaseAiModel<std_ddpm_input, std_ddpm_output>>(
            new jinq::models::diffusion::DDPMSampler<std_ddpm_input, std_ddpm_output>());
    };
    spec.handle_output = [](const std_ddpm_input& in, const std_ddpm_output& out,
                            const std::string& save_dir) {
        if (in.save_all_mid_results) {
            for (auto idx = 0; idx < in.timestep; ++idx) {
                auto image = out.out_images[idx];
                std::string save_name =
                    "sample-step-" + std::to_string(in.timestep - 1 - idx) + ".png";
                cv::imwrite(jinq::common::FilePathUtil::concat_path(save_dir, save_name), image);
            }
        } else {
            cv::imwrite(jinq::common::FilePathUtil::concat_path(save_dir, "sample-step-0.png"),
                        out.out_images[0]);
        }
        LOG(INFO) << "sampled result image has been written into: " << save_dir;
    };
    return jinq::apps::run_benchmark(argc, argv, spec);
}
