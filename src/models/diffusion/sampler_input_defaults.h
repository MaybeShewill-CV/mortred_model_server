/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sampler_input_defaults.h
 * Date: 26-9-7
 ************************************************/

// Seeds the HTTP adapter's per-request sampler INPUT template from the
// sampler TOML section. sample_size is a weight contract ([height, width]),
// not a request-overridable ParamSpec. Missing or non-positive size is an
// init failure so a ready process never serves 0x0 noise.

#ifndef MORTRED_MODELS_DIFFUSION_SAMPLER_INPUT_DEFAULTS_H
#define MORTRED_MODELS_DIFFUSION_SAMPLER_INPUT_DEFAULTS_H

#include <cstdint>
#include <string>
#include <type_traits>

#include "glog/logging.h"
#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/io/diffusion.h"

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace diffusion {

using jinq::common::StatusCode;
using jinq::models::io_define::diffusion::std_cls_cond_ddim_input;
using jinq::models::io_define::diffusion::std_ddim_input;
using jinq::models::io_define::diffusion::std_ddpm_input;
using jinq::models::io_define::diffusion::std_ldm_input;

inline StatusCode parse_sample_size(const toml::table &section, const char *section_name, cv::Size *out) {
    if (!section.contains("sample_size")) {
        LOG(ERROR) << section_name << " missing sample_size = [height, width]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::array *size = section["sample_size"].as_array();
    if (size == nullptr || size->size() != 2) {
        LOG(ERROR) << section_name << " sample_size must be [height, width]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    const int64_t height = (*size)[0].value_or<int64_t>(-1);
    const int64_t width = (*size)[1].value_or<int64_t>(-1);
    if (height <= 0 || width <= 0) {
        LOG(ERROR) << section_name << " sample_size must contain positive integers";
        return StatusCode::MODEL_INIT_FAILED;
    }
    *out = cv::Size(static_cast<int>(width), static_cast<int>(height));
    return StatusCode::OK;
}

inline int parse_positive_i32(const toml::table &section, const char *key, int fallback) {
    const int64_t value = section[key].value_or<int64_t>(fallback);
    return static_cast<int>(value);
}

inline StatusCode require_positive_i32(const toml::table &section, const char *section_name, const char *key, int *out) {
    if (!section.contains(key)) {
        LOG(ERROR) << section_name << " missing " << key;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const int64_t value = section[key].value_or<int64_t>(0);
    if (value <= 0) {
        LOG(ERROR) << section_name << " " << key << " must be > 0";
        return StatusCode::MODEL_INIT_FAILED;
    }
    *out = static_cast<int>(value);
    return StatusCode::OK;
}

inline StatusCode seed_ddpm_input(const toml::table &cfg, std_ddpm_input *input) {
    const toml::table *section = cfg["DDPM_SAMPLER"].as_table();
    if (section == nullptr) {
        LOG(ERROR) << "missing [DDPM_SAMPLER]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    StatusCode status = parse_sample_size(*section, "[DDPM_SAMPLER]", &input->sample_size);
    if (status != StatusCode::OK) {
        return status;
    }
    status = require_positive_i32(*section, "[DDPM_SAMPLER]", "timesteps", &input->timestep);
    if (status != StatusCode::OK) {
        return status;
    }
    const int channels = parse_positive_i32(*section, "channels", 3);
    if (channels <= 0) {
        LOG(ERROR) << "[DDPM_SAMPLER] channels must be > 0";
        return StatusCode::MODEL_INIT_FAILED;
    }
    input->channels = channels;
    input->use_fixed_noise_for_psample = (*section)["use_fixed_noise_for_psample"].value_or<bool>(false);
    input->save_raw_output = (*section)["save_raw_output"].value_or<bool>(false);
    input->save_all_mid_results = false;
    return StatusCode::OK;
}

inline StatusCode seed_ddim_common(const toml::table &section, const char *section_name, int *total_steps, int *sample_steps,
                                   int *channels, float *eta, cv::Size *sample_size) {
    StatusCode status = parse_sample_size(section, section_name, sample_size);
    if (status != StatusCode::OK) {
        return status;
    }
    status = require_positive_i32(section, section_name, "total_timesteps", total_steps);
    if (status != StatusCode::OK) {
        return status;
    }
    status = require_positive_i32(section, section_name, "sample_steps", sample_steps);
    if (status != StatusCode::OK) {
        return status;
    }
    const int ch = parse_positive_i32(section, "channels", 3);
    if (ch <= 0) {
        LOG(ERROR) << section_name << " channels must be > 0";
        return StatusCode::MODEL_INIT_FAILED;
    }
    *channels = ch;
    *eta = static_cast<float>(section["eta"].value_or<double>(1.0));
    return StatusCode::OK;
}

inline StatusCode seed_ddim_input(const toml::table &cfg, std_ddim_input *input) {
    const toml::table *section = cfg["DDIM_SAMPLER"].as_table();
    if (section == nullptr) {
        LOG(ERROR) << "missing [DDIM_SAMPLER]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    const auto status =
        seed_ddim_common(*section, "[DDIM_SAMPLER]", &input->total_steps, &input->sample_steps, &input->channels, &input->eta,
                         &input->sample_size);
    if (status != StatusCode::OK) {
        return status;
    }
    input->save_raw_output = (*section)["save_raw_output"].value_or<bool>(false);
    input->save_all_mid_results = false;
    return StatusCode::OK;
}

inline StatusCode seed_cls_cond_ddim_input(const toml::table &cfg, std_cls_cond_ddim_input *input) {
    const toml::table *section = cfg["DDIM_SAMPLER"].as_table();
    if (section == nullptr) {
        LOG(ERROR) << "missing [DDIM_SAMPLER]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    const auto status =
        seed_ddim_common(*section, "[DDIM_SAMPLER]", &input->total_steps, &input->sample_steps, &input->channels, &input->eta,
                         &input->sample_size);
    if (status != StatusCode::OK) {
        return status;
    }
    const int64_t cls_id = (*section)["cls_id"].value_or<int64_t>(0);
    if (cls_id < 0) {
        LOG(ERROR) << "[DDIM_SAMPLER] cls_id must be >= 0";
        return StatusCode::MODEL_INIT_FAILED;
    }
    input->cls_id = static_cast<int>(cls_id);
    input->save_all_mid_results = false;
    return StatusCode::OK;
}

inline StatusCode seed_ldm_input(const toml::table &cfg, std_ldm_input *input) {
    const toml::table *section = cfg["LDM_SAMPLER"].as_table();
    if (section == nullptr) {
        LOG(ERROR) << "missing [LDM_SAMPLER]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    StatusCode status = parse_sample_size(*section, "[LDM_SAMPLER]", &input->sample_size);
    if (status != StatusCode::OK) {
        return status;
    }
    status = require_positive_i32(*section, "[LDM_SAMPLER]", "step_size", &input->step_size);
    if (status != StatusCode::OK) {
        return status;
    }
    input->downscale = parse_positive_i32(*section, "downscale", input->downscale);
    input->latent_dims = parse_positive_i32(*section, "latent_dims", input->latent_dims);
    input->latent_scale = static_cast<float>((*section)["latent_scale"].value_or<double>(input->latent_scale));
    const std::string sampler_name = (*section)["sampler_type"].value_or<std::string>(std::string("ddim"));
    if (sampler_name == "ddpm") {
        input->sampler_type = jinq::models::io_define::diffusion::DDPMSampler_Type::DDPM;
    } else if (sampler_name == "ddim") {
        input->sampler_type = jinq::models::io_define::diffusion::DDPMSampler_Type::DDIM;
    } else {
        LOG(ERROR) << "[LDM_SAMPLER] sampler_type must be ddim or ddpm";
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename SAMPLER_INPUT>
StatusCode seed_sampler_input(const toml::table &cfg, SAMPLER_INPUT *input) {
    if constexpr (std::is_same_v<SAMPLER_INPUT, std_ddpm_input>) {
        return seed_ddpm_input(cfg, input);
    } else if constexpr (std::is_same_v<SAMPLER_INPUT, std_ddim_input>) {
        return seed_ddim_input(cfg, input);
    } else if constexpr (std::is_same_v<SAMPLER_INPUT, std_cls_cond_ddim_input>) {
        return seed_cls_cond_ddim_input(cfg, input);
    } else if constexpr (std::is_same_v<SAMPLER_INPUT, std_ldm_input>) {
        return seed_ldm_input(cfg, input);
    } else {
        LOG(ERROR) << "seed_sampler_input: unsupported sampler input type";
        return StatusCode::MODEL_INIT_FAILED;
    }
}

} // namespace diffusion
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_DIFFUSION_SAMPLER_INPUT_DEFAULTS_H
