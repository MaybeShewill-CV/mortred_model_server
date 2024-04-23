/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ddpm_unet_benchmark.cpp
 * Date: 24-4-23
 ************************************************/

// ddpm-unet benckmark tool

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "models/diffussion/ddpm_unet.h"

using jinq::common::CvUtils;
using jinq::common::Timestamp;
using jinq::common::FilePathUtil;
using jinq::models::io_define::diffusion::std_ddpm_unet_input;
using jinq::models::io_define::diffusion::std_ddpm_unet_output;
using jinq::models::diffusion::DDPMUNet;

int main(int argc, char** argv) {

    if (argc != 2 && argc != 3) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path [test_image_path]";
        return -1;
    }

    std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }
    auto cfg = toml::parse(cfg_file_path);

    // construct model input
    cv::Mat input_image(cv::Size(128, 128), CV_32FC3);
    cv::randn(input_image, 0.0, 1.0);
    std_ddpm_unet_input model_input;
    model_input.xt = input_image;
    model_input.timestep = 1000;
    std_ddpm_unet_output model_output;

    // construct enhancementor
    auto unet = std::make_unique<DDPMUNet<std_ddpm_unet_input, std_ddpm_unet_output > >();
    unet->init(cfg);
    if (!unet->is_successfully_initialized()) {
        LOG(INFO) << "ddpm unet model init failed";
        return -1;
    }

    // run benchmark
    int loop_times = 100;
    LOG(INFO) << "input test image size: " << input_image.size();
    LOG(INFO) << "ddpm unet run loop times: " << loop_times;
    LOG(INFO) << "start ddpm unet benchmark at: " << Timestamp::now().to_format_str();
    auto ts = Timestamp::now();
    for (int i = 0; i < loop_times; ++i) {
        unet->run(model_input, model_output);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;
    LOG(INFO) << model_output.predict_noise.size();
    LOG(INFO) << model_output.predict_noise.channels();

    return 0;
}
