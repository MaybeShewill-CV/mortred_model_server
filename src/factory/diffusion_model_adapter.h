#ifndef MORTRED_FACTORY_DIFFUSION_MODEL_ADAPTER_H
#define MORTRED_FACTORY_DIFFUSION_MODEL_ADAPTER_H

#include <opencv2/opencv.hpp>
#include <type_traits>
#include <vector>

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/status_code.h"
#include "models/base_model.h"
#include "models/backend/param_spec.h"
#include "models/model_io_define.h"

namespace jinq {
namespace factory {
namespace diffusion {

using ImageInput = jinq::models::io_define::common_io::image_input;
using Base64Output = jinq::models::io_define::common_io::base64_input;

/***
 * Wraps any diffusion sampler into the unified image_input contract. The
 * inner sampler's INPUT is a template assembled from the TOML config at init
 * time; each request copies the template and overlays the whitelisted knobs
 * from the request params (timesteps / steps / eta / cls_id - family
 * dependent). The base64 payload itself is still ignored (generative model);
 * the first generated image is encoded back to base64 PNG as the response.
 */
template <typename SAMPLER, typename SAMPLER_INPUT, typename SAMPLER_OUTPUT>
class DiffusionModelAdapter : public jinq::models::BaseAiModel<ImageInput, Base64Output> {
  public:
    jinq::common::StatusCode init(const toml::table &cfg) override {
        _m_sampler = std::make_unique<SAMPLER>();
        const auto status = _m_sampler->init(cfg);
        _m_initialized = status == jinq::common::StatusCode::OK;
        return status;
    }

  protected:
    jinq::common::StatusCode run_impl(const ImageInput &in, Base64Output &out) override {
        // copy the config template and overlay request-level knobs: the
        // worker stays reusable across concurrent requests
        SAMPLER_INPUT request_input = _m_input;
        if (in.params != nullptr) {
            apply_request_overrides(*in.params, &request_input);
        }
        SAMPLER_OUTPUT sampler_output;
        const auto status = _m_sampler->run(request_input, sampler_output);
        if (status != jinq::common::StatusCode::OK) {
            return status;
        }
        const cv::Mat &image = first_image(sampler_output);
        if (image.empty()) {
            return jinq::common::StatusCode::MODEL_EMPTY_OUTPUT;
        }
        std::vector<uchar> buffer;
        cv::imencode(".png", image, buffer);
        out.input_image_content = jinq::common::base64::encode(buffer.data(), buffer.size());
        return jinq::common::StatusCode::OK;
    }

    bool is_successfully_initialized() const override { return _m_initialized; }

    SAMPLER_INPUT &mutable_input() { return _m_input; }

    /*** test seam: inspect the fake sampler without engines */
    SAMPLER &sampler() { return *_m_sampler; }

  private:
    /*** request-overridable knobs per sampler family; the config template
     * value is the fallback for every key (the same resolution rule as the
     * detection/classification families) */
    void apply_request_overrides(const jinq::models::backend::ParamSet &params, SAMPLER_INPUT *input) const {
        if constexpr (std::is_same_v<SAMPLER_INPUT, jinq::models::io_define::diffusion::std_ddpm_input>) {
            input->timestep = params.get_i32("timesteps", input->timestep);
        } else if constexpr (std::is_same_v<SAMPLER_INPUT, jinq::models::io_define::diffusion::std_ddim_input>) {
            input->sample_steps = params.get_i32("sample_steps", input->sample_steps);
            input->eta = params.get_f32("eta", input->eta);
        } else if constexpr (std::is_same_v<SAMPLER_INPUT, jinq::models::io_define::diffusion::std_cls_cond_ddim_input>) {
            input->sample_steps = params.get_i32("sample_steps", input->sample_steps);
            input->eta = params.get_f32("eta", input->eta);
            input->cls_id = params.get_i32("cls_id", input->cls_id);
        } else if constexpr (std::is_same_v<SAMPLER_INPUT, jinq::models::io_define::diffusion::std_ldm_input>) {
            input->step_size = params.get_i32("step_size", input->step_size);
        }
    }

    // extract the first image from the various sampler output contracts
    const cv::Mat &first_image(const SAMPLER_OUTPUT &output) const {
        if constexpr (std::is_same_v<SAMPLER_OUTPUT, jinq::models::io_define::diffusion::std_ddpm_output>) {
            static const cv::Mat empty;
            return output.out_images.empty() ? empty : output.out_images.front();
        } else if constexpr (std::is_same_v<SAMPLER_OUTPUT, jinq::models::io_define::diffusion::std_ddim_output>) {
            static const cv::Mat empty;
            return output.sampled_images.empty() ? empty : output.sampled_images.front();
        } else if constexpr (std::is_same_v<SAMPLER_OUTPUT, jinq::models::io_define::diffusion::std_cls_cond_ddim_output>) {
            static const cv::Mat empty;
            return output.sampled_images.empty() ? empty : output.sampled_images.front();
        } else if constexpr (std::is_same_v<SAMPLER_OUTPUT, jinq::models::io_define::diffusion::std_ldm_output>) {
            return output.sampled_image;
        } else {
            static const cv::Mat empty;
            return empty;
        }
    }

    std::unique_ptr<SAMPLER> _m_sampler;
    SAMPLER_INPUT _m_input{};
    bool _m_initialized = false;
};

} // namespace diffusion
} // namespace factory
} // namespace jinq

#endif // MORTRED_FACTORY_DIFFUSION_MODEL_ADAPTER_H
