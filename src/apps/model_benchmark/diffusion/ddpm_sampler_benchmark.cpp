/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ddpm_sampler_benchmark.cpp
 * Date: 24-4-24
 ************************************************/

// ddpm-sampler benchmark tool

#include <random>

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "models/diffussion/ddpm_sampler.h"

using jinq::common::CvUtils;
using jinq::common::Timestamp;
using jinq::common::FilePathUtil;
using jinq::models::io_define::diffusion::std_ddpm_input;
using jinq::models::io_define::diffusion::std_ddpm_output;
using jinq::models::diffusion::DDPMSampler;

int main(int argc, char** argv) {

    if (argc != 2 && argc != 3 && argc != 4 && argc != 5) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path [save_dir] [save_all_mid_results(default: true)] [use_fixed_noise(default: false)]";
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

    std::string save_dir = "../demo_data/model_test_input/diffusion/ddpm";
    if (argc == 3) {
        save_dir = argv[2];
    }

    bool save_all_mid_results = false;
    if (argc >= 4) {
        save_all_mid_results = std::stoi(argv[3]) == 1;
    }

    bool use_fixed_noise = false;
    if (argc >= 5) {
        use_fixed_noise = std::stoi(argv[4]) == 1;
    }

    // construct model input
    std_ddpm_input model_input;
    model_input.sample_size = cv::Size(128, 128);
    model_input.timestep = 1000;
    model_input.channels = 3;
    model_input.save_all_mid_results = save_all_mid_results;
    model_input.use_fixed_noise_for_psample = use_fixed_noise;
    std_ddpm_output model_output;

    // construct ddpm unet
    auto unet = std::make_unique<DDPMSampler<std_ddpm_input, std_ddpm_output > >();
    unet->init(cfg);
    if (!unet->is_successfully_initialized()) {
        LOG(INFO) << "ddpm sampler model init failed";
        return -1;
    }

    // run benchmark
    int loop_times = 1;
    LOG(INFO) << "ddpm unet run loop times: " << loop_times;
    LOG(INFO) << "start ddpm sampler benchmark at: " << Timestamp::now().to_format_str();
    auto ts = Timestamp::now();
    for (int i = 0; i < loop_times; ++i) {
        unet->run(model_input, model_output);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;

    // save sampled images
    if (save_all_mid_results) {
        for (auto idx = 0; idx < model_input.timestep; ++idx) {
            auto image = model_output.out_images[idx];
            std::string save_name = "sample-step-" + std::to_string(model_input.timestep - 1 - idx) + ".png";
            std::string save_path = FilePathUtil::concat_path(save_dir, save_name);
            cv::imwrite(save_path, image);
        }
    } else {
        auto image = model_output.out_images[0];
        std::string save_name = "sample-step-0.png";
        std::string save_path = FilePathUtil::concat_path(save_dir, save_name);
        cv::imwrite(save_path, image);
    }
    LOG(INFO) << "sampled result image has been written into: " << save_dir;

    return 0;
}