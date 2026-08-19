/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * File: ldm_sampler_benchmark.cpp
 * Date: 2026-08-19
 ************************************************/

// ldm-sampler benchmark tool

#include "apps/common/benchmark_runner.h"
#include "models/diffusion/ldm_sampler.h"

using jinq::models::io_define::diffusion::DDPMSampler_Type;
using jinq::models::io_define::diffusion::std_ldm_input;
using jinq::models::io_define::diffusion::std_ldm_output;

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<std_ldm_input, std_ldm_output> spec;
    spec.model_name = "ldm";
    spec.display_name = "ldm sampler";
    spec.usage =
        "exe config_file_path [sample_size(default: 256)] [sample_steps(default: 100)] "
        "[sampler_type(default: ddim)]";
    spec.loops = 1;
    spec.warmup = false;
    spec.args_ok = [](int argc) { return argc >= 2 && argc <= 5; };
    spec.input_ok = [](const std_ldm_input&) { return true; };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        int sample_size = 256;
        if (argc >= 3) {
            sample_size = std::stoi(argv[2]);
        }
        int sample_steps = 100;
        if (argc >= 4) {
            sample_steps = std::stoi(argv[3]);
        }
        auto sampler_type = DDPMSampler_Type::DDIM;
        if (argc >= 5) {
            sampler_type = static_cast<DDPMSampler_Type>(std::stoi(argv[4]));
        }
        std_ldm_input in;
        in.sample_size = cv::Size(sample_size, sample_size);
        in.step_size = sample_steps;
        if (sampler_type == DDPMSampler_Type::DDPM) {
            in.step_size = 1000;
        }
        in.downscale = 8;
        in.latent_dims = 4;
        in.latent_scale = 0.18215f;
        in.sampler_type = sampler_type;
        return in;
    };
    spec.image_path_of = [](int, char**) {
        return std::string("../demo_data/model_test_input/diffusion/ldm");
    };
    spec.make_model = [](const std::string&) {
        return std::unique_ptr<jinq::models::BaseAiModel<std_ldm_input, std_ldm_output>>(
            new jinq::models::diffusion::LDMSampler<std_ldm_input, std_ldm_output>());
    };
    spec.handle_output = [](const std_ldm_input&, const std_ldm_output& out,
                            const std::string& save_dir) {
        cv::imwrite(jinq::common::FilePathUtil::concat_path(save_dir, "ldm_sample_output.png"),
                    out.sampled_image);
        LOG(INFO) << "sampled result image has been written into: " << save_dir;
    };
    return jinq::apps::run_benchmark(argc, argv, spec);
}
