/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: diffusion.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_DIFFUSION_H
#define MORTRED_MODELS_IO_DIFFUSION_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace io_define {
namespace diffusion {

// diffusion

enum DDPMSampler_Type {
    DDPM = 0,
    DDIM = 1,
};

struct ddpm_unet_input {
    std::vector<float> xt;
    int64_t timestep;
};
struct ddpm_unet_output {
    std::vector<float> predict_noise;
};
using std_ddpm_unet_input = ddpm_unet_input;
using std_ddpm_unet_output = ddpm_unet_output;

struct cls_cond_ddpm_unet_input {
    std::vector<float> xt;
    int64_t timestep;
    int cls_id = 0;
};
struct cls_cond_ddpm_unet_output {
    std::vector<float> predict_noise;
};
using std_cls_cond_ddpm_unet_input = cls_cond_ddpm_unet_input;
using std_cls_cond_ddpm_unet_output = cls_cond_ddpm_unet_output;

struct ddpm_sample_input {
    cv::Size sample_size;
    int timestep;
    int channels = 3;
    bool save_all_mid_results = true;
    bool use_fixed_noise_for_psample = false;
    bool save_raw_output = false;
};
struct ddpm_sample_output {
    std::vector<cv::Mat> out_images;
    std::vector<std::vector<float>> out_raw_predictions;
};
using std_ddpm_input = ddpm_sample_input;
using std_ddpm_output = ddpm_sample_output;

struct ddim_sample_input {
    cv::Size sample_size;
    int total_steps;
    int sample_steps;
    int channels = 3;
    bool save_all_mid_results = true;
    float *xt_data = nullptr;
    float eta = 1.0f;
    bool save_raw_output = false;
};
struct ddim_sample_output {
    std::vector<cv::Mat> sampled_images;
    std::vector<cv::Mat> predicted_x0;
    std::vector<std::vector<float>> raw_sampled_images;
    std::vector<std::vector<float>> raw_predicted_x0;
};
using std_ddim_input = ddim_sample_input;
using std_ddim_output = ddim_sample_output;

struct cls_cond_ddim_sample_input {
    cv::Size sample_size;
    int total_steps;
    int sample_steps;
    int cls_id = 0;
    int channels = 3;
    bool save_all_mid_results = true;
    float *xt_data = nullptr;
    float eta = 1.0f;
};
struct cls_cond_ddim_sample_output {
    std::vector<cv::Mat> sampled_images;
    std::vector<cv::Mat> predicted_x0;
};
using std_cls_cond_ddim_input = cls_cond_ddim_sample_input;
using std_cls_cond_ddim_output = cls_cond_ddim_sample_output;

struct autoencoder_kl_input {
    std::vector<float> decode_data;
};
struct autoencoder_kl_output {
    cv::Mat decode_output;
};
using std_vae_decode_input = autoencoder_kl_input;
using std_vae_decode_output = autoencoder_kl_output;

struct ldm_sample_input {
    cv::Size sample_size;
    int step_size;
    int downscale = 8;
    int latent_dims = 4;
    float latent_scale = 0.18215f;
    DDPMSampler_Type sampler_type = DDPMSampler_Type::DDIM;
};
struct ldm_sample_output {
    cv::Mat sampled_image;
};
using std_ldm_input = ldm_sample_input;
using std_ldm_output = ldm_sample_output;

} // namespace diffusion
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_DIFFUSION_H
