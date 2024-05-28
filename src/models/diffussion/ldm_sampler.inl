/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ldm_sampler.cpp
 * Date: 24-5-24
 ************************************************/

#include "ldm_sampler.h"

#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/diffussion/ddim_sampler.h"
#include "models/diffussion/ddpm_sampler.h"
#include "models/diffussion/autoencoder_kl.h"

namespace jinq {
namespace models {

using jinq::common::Base64;
using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;

namespace diffusion {

using jinq::models::io_define::diffusion::DDPMSampler_Type;
using jinq::models::io_define::diffusion::std_ddpm_input;
using jinq::models::io_define::diffusion::std_ddpm_output;
using jinq::models::io_define::diffusion::std_ddim_input;
using jinq::models::io_define::diffusion::std_ddim_output;
using jinq::models::io_define::diffusion::std_vae_decode_input;
using jinq::models::io_define::diffusion::std_vae_decode_output;
using jinq::models::io_define::diffusion::std_ldm_input;
using jinq::models::io_define::diffusion::std_ldm_output;
using DDPMSamplerPtr = jinq::models::diffusion::DDPMSampler<std_ddpm_input, std_ddpm_output>;
using DDIMSamplerPtr = jinq::models::diffusion::DDIMSampler<std_ddim_input, std_ddim_output>;
using VAEDecoderPtr = jinq::models::diffusion::AutoEncoderKL<std_vae_decode_input, std_vae_decode_output>;

namespace ldm_sampler_impl {

using internal_input = std_ldm_input;
using internal_output = std_ldm_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<std_ldm_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    return in;
}

/***
* transform different type of internal output into external output
* @tparam EXTERNAL_OUTPUT
* @tparam dummy
* @param in
* @return
 */
template <typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_ldm_output >::type>::value, std_ldm_output>::type
transform_output(const ldm_sampler_impl::internal_output& internal_out) {
    return internal_out;
}

} // namespace ddim_sampler_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class LDMSampler<INPUT, OUTPUT>::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

    /***
     *
     * @param transformer
     */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const decltype(toml::parse("")) &config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  private:
    // latent sampler
    std::unique_ptr<DDIMSamplerPtr> _m_ddim_sampler;
    std::unique_ptr<DDPMSamplerPtr> _m_ddpm_sampler;

    // vae decoder
    std::unique_ptr<VAEDecoderPtr> _m_vae_decoder;

    // ldm params
    DDPMSampler_Type _m_latent_sampler_type = DDPMSampler_Type::DDIM;
    int _m_schedule_total_steps = 1000;
    int _m_latent_dims = 4;
    int _m_downscale = 8;
    float _m_latent_scale = 0.18215f;

    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @param sample_size
     * @param latent_dims
     * @param step_size
     * @param latent_sample
     * @return
     */
    StatusCode generate_latent_samples(
        const cv::Size& sample_size, int latent_dims, int step_size, std::vector<float>& latent_sample);
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode LDMSampler<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // init diffusion sampler
    auto sampler_cfg_path = config.at("LDM_SAMPLER").at("latent_diffusion_cfg").as_string();
    if (!FilePathUtil::is_file_exist(sampler_cfg_path)) {
        LOG(ERROR) << "diffusion sampler config file: " << sampler_cfg_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto sampler_cfg = toml::parse(sampler_cfg_path);
    _m_ddim_sampler = std::make_unique<DDIMSamplerPtr>();
    auto status = _m_ddim_sampler->init(sampler_cfg);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "init latent ddim sampler failed, status code: " << status;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_ddpm_sampler = std::make_unique<DDPMSamplerPtr>();
    status = _m_ddpm_sampler->init(sampler_cfg);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "init latent ddpm sampler failed, status code: " << status;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init vae decoder
    auto vae_cfg_path = config.at("LDM_SAMPLER").at("vae_decoder_cfg").as_string();
    if (!FilePathUtil::is_file_exist(sampler_cfg_path)) {
        LOG(ERROR) << "vae decoder model config file: " << sampler_cfg_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto vae_decoder_cfg = toml::parse(vae_cfg_path);
    _m_vae_decoder = std::make_unique<VAEDecoderPtr>();
    status = _m_vae_decoder->init(vae_decoder_cfg);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "init vae decoder failed, status code: " << status;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init other ldm params
    _m_latent_dims = static_cast<int>(config.at("LDM_SAMPLER").at("latent_dims").as_integer());
    _m_latent_scale = static_cast<float>(config.at("LDM_SAMPLER").at("latent_scale").as_floating());
    _m_downscale = static_cast<int>(config.at("LDM_SAMPLER").at("downscale").as_integer());

    _m_successfully_initialized = true;

    return StatusCode::OK;
}

/***
*
* @param in
* @param out
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode LDMSampler<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // parse input params
    auto transformed_in = ldm_sampler_impl::transform_input(in);
    auto& sample_size = transformed_in.sample_size;
    auto& step_size = transformed_in.step_size;
    _m_downscale = transformed_in.downscale;
    _m_latent_dims = transformed_in.latent_dims;
    _m_latent_scale =  transformed_in.latent_scale;
    _m_latent_sampler_type = transformed_in.sampler_type;

    // generate latent space samples
    std::vector<float> latent_sample;
    cv::Size latent_size;
    latent_size.width = static_cast<int>(sample_size.width / _m_downscale);
    latent_size.height = static_cast<int>(sample_size.height / _m_downscale);
    auto status = generate_latent_samples(latent_size, _m_latent_dims, step_size, latent_sample);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "run ldm latent sample session failed, status code: " << status;
        return status;
    }
    for (auto& val : latent_sample) {
        val /= _m_latent_scale;
    }

    // vae decode output images
    std_vae_decode_input decoder_input;
    decoder_input.decode_data = latent_sample;
    std_vae_decode_output decoder_output;
    status = _m_vae_decoder->run(decoder_input, decoder_output);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "run ldm vae decode session failed, status code: " << status;
        return status;
    }

    // transform output
    ldm_sampler_impl::internal_output internal_out;
    decoder_output.decode_output.copyTo(internal_out.sampled_image);
    out = ldm_sampler_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param sample_size
 * @param latent_dims
 * @param step_size
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode LDMSampler<INPUT, OUTPUT>::Impl::generate_latent_samples(
    const cv::Size &sample_size, const int latent_dims, const int step_size, std::vector<float>& latent_sample) {
    StatusCode status;
    if (_m_latent_sampler_type == DDPMSampler_Type::DDPM) {
        std_ddpm_input input;
        input.sample_size = sample_size;
        input.save_all_mid_results = false;
        input.timestep = step_size;
        input.use_fixed_noise_for_psample = false;
        input.channels = latent_dims;
        input.save_raw_output = true;

        std_ddpm_output output;
        status = _m_ddpm_sampler->run(input, output);
        if (status != StatusCode::OK) {
            LOG(ERROR) << "run latent ddpm sampler failed, status code: " << status;
            latent_sample = std::vector<float>(sample_size.area() * latent_dims, 0.0);
            status = StatusCode::MODEL_RUN_SESSION_FAILED;
        } else {
            latent_sample = output.out_raw_predictions[0];
        }

    } else if (_m_latent_sampler_type == DDPMSampler_Type::DDIM) {
        std_ddim_input input;
        input.sample_size = sample_size;
        input.total_steps = _m_schedule_total_steps;
        input.sample_steps = step_size;
        input.channels = latent_dims;
        input.save_all_mid_results = false;
        input.xt_data = nullptr;
        input.eta = 1.0f;
        input.save_raw_output = true;

        std_ddim_output output;
        status = _m_ddim_sampler->run(input, output);
        if (status != StatusCode::OK) {
            LOG(ERROR) << "run latent ddim sampler failed, status code: " << status;
            latent_sample = std::vector<float>(sample_size.area() * latent_dims, 0.0);
            status = StatusCode::MODEL_RUN_SESSION_FAILED;
        } else {
            latent_sample = output.raw_predicted_x0[0];
        }

    } else {
        LOG(ERROR) << "not support latent sampler type: " << _m_latent_sampler_type;
        status = StatusCode::MODEL_RUN_SESSION_FAILED;
        latent_sample = std::vector<float>(sample_size.area() * latent_dims, 0.0);
    }

    return status;
}

/************* Export Function Sets *************/

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
LDMSampler<INPUT, OUTPUT>::LDMSampler() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
LDMSampler<INPUT, OUTPUT>::~LDMSampler() = default;

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param cfg
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode LDMSampler<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @return
 */
template <typename INPUT, typename OUTPUT>
bool LDMSampler<INPUT, OUTPUT>::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param input
* @param output
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode LDMSampler<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace diffusion
} // namespace models
} // namespace jinq
