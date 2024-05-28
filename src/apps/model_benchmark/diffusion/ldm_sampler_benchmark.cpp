/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ldm_sampler_benchmark.cpp
 * Date: 24-5-24
 ************************************************/

// ldm-sampler benchmark tool

#include <random>

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "models/diffussion/ldm_sampler.h"

using jinq::common::CvUtils;
using jinq::common::Timestamp;
using jinq::common::FilePathUtil;
using jinq::models::diffusion::LDMSampler;
using jinq::models::diffusion::AutoEncoderKL;
using jinq::models::io_define::diffusion::std_vae_decode_input;
using jinq::models::io_define::diffusion::std_vae_decode_output;
using jinq::models::io_define::diffusion::std_ldm_input;
using jinq::models::io_define::diffusion::std_ldm_output;
using jinq::models::io_define::diffusion::DDPMSampler_Type;

int main(int argc, char** argv) {

    if (argc != 2 && argc != 3 && argc != 4 && argc != 5) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path [sample_size(default: 256)] [sample_steps(default: 100)] "
                     "[sampler_type(default: ddim)]";
        return -1;
    }

    // parse input params
    std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }
    auto cfg = toml::parse(cfg_file_path);

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

    // construct model input
    std_ldm_output model_output;
    std_ldm_input model_input;
    model_input.sample_size = cv::Size(sample_size, sample_size);
    model_input.step_size = sample_steps;
    if (sampler_type == DDPMSampler_Type::DDPM) {
        model_input.step_size = 1000;
    }
    model_input.downscale = 8;
    model_input.latent_dims = 4;
    model_input.latent_scale = 0.18215f;
    model_input.sampler_type = sampler_type;

    // construct ddpm unet
    auto sampler = std::make_unique<LDMSampler<std_ldm_input, std_ldm_output> >();
    sampler->init(cfg);
    if (!sampler->is_successfully_initialized()) {
        LOG(INFO) << "ldm sampler model init failed";
        return -1;
    }

    // run benchmark
    int loop_times = 1;
    LOG(INFO) << "ldm sampler run loop times: " << loop_times;
    LOG(INFO) << "start ldm sampler benchmark at: " << Timestamp::now().to_format_str();
    auto ts = Timestamp::now();
    for (int i = 0; i < loop_times; ++i) {
        sampler->run(model_input, model_output);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;

    // save sampled images
    std::string save_dir = "../demo_data/model_test_input/diffusion/ldm";
    auto image = model_output.sampled_image;
    std::string save_name = "ldm_sample_output.png";
    std::string save_path = FilePathUtil::concat_path(save_dir, save_name);
    cv::imwrite(save_path, image);
    LOG(INFO) << "sampled result image has been written into: " << save_path;

    return 0;
}