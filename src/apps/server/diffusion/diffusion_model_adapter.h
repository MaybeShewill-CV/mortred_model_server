/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: diffusion_model_adapter.h
* Date: 26-8-23
************************************************/

// Adapter that wraps a diffusion sampler (custom INPUT/OUTPUT types) into
// the base64_input/base64_output contract expected by the server framework.
// The base64 payload is ignored; diffusion parameters come from the TOML
// config. The generated image is returned as a base64-encoded PNG.

#ifndef MORTRED_APPS_DIFFUSION_MODEL_ADAPTER_H
#define MORTRED_APPS_DIFFUSION_MODEL_ADAPTER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "common/base64.h"
#include "common/cv_utils.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "server/generic_cv_server.h"

namespace jinq {
namespace apps {
namespace diffusion {

using Base64Input = jinq::models::io_define::common_io::base64_input;
using Base64Output = jinq::models::io_define::common_io::base64_input;

/***
 * Wraps any diffusion sampler into the base64_input contract. The inner
 * sampler's INPUT is constructed from the TOML config at init time; the
 * base64 payload from the HTTP request is accepted but ignored (the server
 * framework requires it to be non-empty). The first generated image is
 * encoded back to base64 as the response.
 */
template<typename SAMPLER, typename SAMPLER_INPUT, typename SAMPLER_OUTPUT>
class DiffusionModelAdapter : public jinq::models::BaseAiModel<Base64Input, Base64Output> {
  public:
    explicit DiffusionModelAdapter(const std::string& name) : _m_name(name) {}

    StatusCode init(const toml::table& cfg) override {
        _m_sampler = std::make_unique<SAMPLER>();
        const auto status = _m_sampler->init(cfg);
        _m_initialized = status == jinq::common::StatusCode::OK;
        return status;
    }

  protected:
    StatusCode run_impl(const Base64Input& in, Base64Output& out) override {
        (void)in;  // diffusion parameters come from config, not from the payload
        SAMPLER_OUTPUT sampler_output;
        const auto status = _m_sampler->run(_m_input, sampler_output);
        if (status != jinq::common::StatusCode::OK) {
            return status;
        }
        // encode the first generated image as base64
        const cv::Mat& image = first_image(sampler_output);
        if (image.empty()) {
            return jinq::common::StatusCode::MODEL_EMPTY_OUTPUT;
        }
        std::vector<uchar> buffer;
        cv::imencode(".png", image, buffer);
        out.input_image_content = jinq::common::base64::encode(
            buffer.data(), buffer.size());
        return jinq::common::StatusCode::OK;
    }

    bool is_successfully_initialized() const override {
        return _m_initialized;
    }

    SAMPLER_INPUT& mutable_input() {
        return _m_input;
    }

  private:
    // extract the first image from various sampler output types
    const cv::Mat& first_image(const SAMPLER_OUTPUT& output) const {
        if constexpr (std::is_same_v<SAMPLER_OUTPUT,
                                     jinq::models::io_define::diffusion::std_ddpm_output>) {
            static const cv::Mat empty;
            return output.out_images.empty() ? empty : output.out_images.front();
        } else if constexpr (std::is_same_v<SAMPLER_OUTPUT,
                                            jinq::models::io_define::diffusion::std_ddim_output>) {
            static const cv::Mat empty;
            return output.sampled_images.empty() ? empty : output.sampled_images.front();
        } else if constexpr (std::is_same_v<SAMPLER_OUTPUT,
                                            jinq::models::io_define::diffusion::std_ldm_output>) {
            return output.sampled_image;
        } else {
            static const cv::Mat empty;
            return empty;
        }
    }

    std::string _m_name;
    std::unique_ptr<SAMPLER> _m_sampler;
    SAMPLER_INPUT _m_input{};
    bool _m_initialized = false;
};

}  // namespace diffusion
}  // namespace apps
}  // namespace jinq

#endif  // MORTRED_APPS_DIFFUSION_MODEL_ADAPTER_H
