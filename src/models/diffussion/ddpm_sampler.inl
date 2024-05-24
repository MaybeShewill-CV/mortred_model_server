/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: DDPMSampler.cpp
 * Date: 24-4-23
 ************************************************/

#include "ddpm_sampler.h"

#include <random>
#include <functional>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "indicators/indicators.hpp"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/diffussion/ddpm_unet.h"

namespace jinq {
namespace models {

using jinq::common::Base64;
using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;

namespace diffusion {

using jinq::models::io_define::diffusion::std_ddpm_input;
using jinq::models::io_define::diffusion::std_ddpm_output;
using jinq::models::io_define::diffusion::std_ddpm_unet_input;
using jinq::models::io_define::diffusion::std_ddpm_unet_output;
using DenoiseModelPtr = jinq::models::diffusion::DDPMUNet<std_ddpm_unet_input, std_ddpm_unet_output>;

namespace ddpm_sampler_impl {

using internal_input = std_ddpm_input;
using internal_output = std_ddpm_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<std_ddpm_input>::type>::value, internal_input>::type
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_ddpm_output >::type>::value, std_ddpm_output>::type
transform_output(const ddpm_sampler_impl::internal_output& internal_out) {
    return internal_out;
}

/***
 *
 * @param start
 * @param end
 * @param num
 * @return
 */
std::vector<double> linspace(const double start, const double end, const int num) {
    std::vector<double> result(num);
    if (num == 1) {
        result[0] = start;
        return result;
    }
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }

    return result;
}

/***
     *
     * @param vec_size
     * @return
 */
std::vector<float> generate_random_norm_vector(const size_t vec_size, float mean=0.0, float stddev=1.0) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::normal_distribution<float> distribution(mean, stddev);
    std::vector<float> result(vec_size, 0.0);
    std::generate(result.begin(), result.end(), [&]() { return distribution(gen); });
    return result;
}

} // namespace ddpm_sampler_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class DDPMSampler<INPUT, OUTPUT>::Impl {
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
    // beta schedule
    enum beta_schedule_type {
        linear = 0,
        cosine = 1,
        sigmoid = 2,
    };
    std::map<std::string, beta_schedule_type> _m_beta_schedule_type_map = {
        {"linear", linear}, {"cosine", cosine}, {"sigmoid", sigmoid}
    };
    beta_schedule_type _m_beta_schedule;

    // pre-compute coefficient
    std::vector<double> _m_betas;
    std::vector<double> _m_alpha_cumprod;
    int64_t _m_timesteps = 1000;
    double _m_beta_start = 0.0;
    double _m_beta_end = 0.0;
    std::unique_ptr<indicators::BlockProgressBar> _m_p_sample_bar;
    std::vector<float> _m_fixed_noise_for_psample;

    // denoise schedule
    std::unique_ptr<DenoiseModelPtr> _m_denoise_net;

    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @param timesteps
     * @param beta_start
     * @param beta_end
     * @return
     */
    std::vector<double> linear_beta_schedule(const int64_t timesteps, const double beta_start=0.0001, const double beta_end=0.02) {
        double scale = 1000.0 / static_cast<double>(timesteps);
        double real_beta_start = scale * beta_start;
        double real_beta_end = scale * beta_end;
        return ddpm_sampler_impl::linspace(real_beta_start, real_beta_end, static_cast<int>(timesteps));
    }

    /***
     *
     * @param timesteps
     * @param s
     * @return
     */
    std::vector<double> cos_beta_schedule(const int64_t timesteps, const double s=0.008) {
        auto steps = timesteps + 1;
        auto t = ddpm_sampler_impl::linspace(0, static_cast<double>(timesteps), static_cast<int>(steps));
        for (auto& val : t) {
            val /= static_cast<double>(timesteps);
            val = (val + s) / (1.0 + s) * M_PI * 0.5;
            val = std::cos(val);
            val = std::pow(val, 2);
        }
        std::transform(t.begin(), t.end(), t.begin(), [&](double x) { return x / t[0]; });
        std::vector<double> betas;
        for (int idx = 1; idx < t.size(); ++idx) {
            auto val = t[idx] / t[idx - 1];
            val = 1.0 - val;
            betas.push_back(val);
        }
        std::transform(betas.begin(), betas.end(), betas.begin(), [](double x) { return std::clamp(x, 0.0, 0.999); });
        return betas;
    }

    /***
     *
     * @param timesteps
     * @param start
     * @param end
     * @param tau
     * @return
     */
    std::vector<double> sigmoid_beta_schedule(const int64_t timesteps, const int64_t start=-3, const int64_t end=3, const int64_t tau=1) {
        auto steps = timesteps + 1;
        auto t = ddpm_sampler_impl::linspace(0, static_cast<double>(timesteps), static_cast<int>(steps));
        auto v_start = 1.0 / (1 + std::exp(-start / tau));
        auto v_end = 1.0 / (1 + std::exp(-end / tau));
        for (auto& val : t) {
            val = (val * static_cast<double>(end - start) + static_cast<double>(start)) / static_cast<double>(tau);
            val = -1.0 / (1 + std::exp(-val));
            val = (val + v_end) / (v_end - v_start);
        }
        std::transform(t.begin(), t.end(), t.begin(), [&](double x) { return x / t[0]; });
        std::vector<double> betas;
        for (int idx = 1; idx < t.size(); ++idx) {
            auto val = t[idx] / t[idx - 1];
            val = 1.0 - val;
            betas.push_back(val);
        }
        std::transform(betas.begin(), betas.end(), betas.begin(), [](double x) { return std::clamp(x, 0.0, 0.999); });
        return betas;
    }

    /***
     *
     * @param betas
     * @return
     */
    std::vector<double> precompute_alpha_cumprod(const std::vector<double>& betas) {
        std::vector<double> alphas;
        for (auto& val : betas) {
            alphas.push_back(1.0 - val);
        }
        std::vector<double> alpha_cumprod;
        for (auto idx = 0; idx < betas.size(); ++idx) {
            auto val = std::accumulate(alphas.begin(), alphas.begin() + idx + 1, 1.0, std::multiplies<>());
            alpha_cumprod.push_back(val);
        }
        return alpha_cumprod;
    }

    /***
     *
     * @param xt
     * @param predict_noise
     * @param timestep
     * @return
     */
    std::vector<float> compute_predict_mean(
        const std::vector<float>& xt, const std::vector<float>& predict_noise, int64_t timestep);

    /***
     *
     * @param xt
     * @param t
     * @param is_last_step
     * @param use_fixed_noise
     * @return
     */
    std::vector<float> p_sample_once(const std::vector<float>& xt, int64_t t, bool is_last_step=false, bool use_fixed_noise=false);

    /***
     *
     * @param size
     * @param timesteps
     * @param channels
     * @param save_all_mid_results
     * @param use_fixed_noise
     * @return
     */
    std::vector<std::vector<float> > p_sample(
        const cv::Size& size, int timesteps, int channels=3,
        bool save_all_mid_results=false, bool use_fixed_noise=false);
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DDPMSampler<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose beta schedule type
    auto ddpm_sampler_cfg = config.at("DDPM_SAMPLER");
    _m_timesteps = ddpm_sampler_cfg["timesteps"].as_integer();
    _m_beta_start = ddpm_sampler_cfg["beta_start"].as_floating();
    _m_beta_end = ddpm_sampler_cfg["beta_end"].as_floating();
    _m_beta_schedule = _m_beta_schedule_type_map[ddpm_sampler_cfg["beta_schedule"].as_string()];

    // precompute beta and alpha_cumprod
    if (_m_beta_schedule == beta_schedule_type::linear) {
        _m_betas = linear_beta_schedule(_m_timesteps, _m_beta_start, _m_beta_end);
    } else if (_m_beta_schedule == beta_schedule_type::cosine) {
        _m_betas = cos_beta_schedule(_m_timesteps, 0.008);
    } else if (_m_beta_schedule == beta_schedule_type::sigmoid) {
        _m_betas = sigmoid_beta_schedule(_m_timesteps);
    } else {
        LOG(ERROR) << "not support beta schedule type: " << _m_beta_schedule;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_alpha_cumprod = precompute_alpha_cumprod(_m_betas);

    // init denoise model
    _m_denoise_net = std::make_unique<DenoiseModelPtr>();
    auto init_status = _m_denoise_net->init(config);
    if (!_m_denoise_net->is_successfully_initialized()) {
        LOG(INFO) << "init denoise net failed, status code: " << init_status;
        return init_status;
    }

    // init p-sample progress bar
    _m_p_sample_bar = std::make_unique<indicators::BlockProgressBar>();
    _m_p_sample_bar->set_option(indicators::option::BarWidth{80});
    _m_p_sample_bar->set_option(indicators::option::Start{"["});
    _m_p_sample_bar->set_option(indicators::option::End{"]"});
    _m_p_sample_bar->set_option(indicators::option::ForegroundColor{indicators::Color::white});
    _m_p_sample_bar->set_option(indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
    _m_p_sample_bar->set_option(indicators::option::ShowElapsedTime{true});
    _m_p_sample_bar->set_option(indicators::option::ShowPercentage{true});
    _m_p_sample_bar->set_option(indicators::option::ShowRemainingTime(true));

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << "Successfully load ddpm sampler";
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load ddpm sampler";
    }

    return init_status;
}

/***
*
* @param in
* @param out
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DDPMSampler<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // prepare input params
    auto transformed_input = ddpm_sampler_impl::transform_input(in);
    auto sample_size = transformed_input.sample_size;
    auto sample_timestep = transformed_input.timestep;
    auto sample_channels = transformed_input.channels;
    auto save_all_mid_results = transformed_input.save_all_mid_results;
    auto use_fixed_noise_for_psample = transformed_input.use_fixed_noise_for_psample;
    auto save_raw_output = transformed_input.save_raw_output;

    // p-sample loop
    auto mid_sample_results = p_sample(sample_size, sample_timestep, sample_channels, save_all_mid_results, use_fixed_noise_for_psample);

    // transform sampled results into cv::Mat
    ddpm_sampler_impl::internal_output internal_out;
    StatusCode sample_status = StatusCode::OK;
    for (auto& img_data : mid_sample_results) {
        if (save_raw_output) {
            internal_out.out_raw_predictions.push_back(img_data);
        }
        // rescale image data to [0, 255]
        std::transform(img_data.begin(), img_data.end(), img_data.begin(), [](float x) { return std::clamp(x, -1.0f, 1.0f); });
        std::transform(img_data.begin(), img_data.end(), img_data.begin(), [](float x) { return (x + 1.0f) * 0.5f * 255.0f + 0.5; });
        std::transform(img_data.begin(), img_data.end(), img_data.begin(), [](float x) { return std::clamp(x, 0.0f, 255.0f); });
        auto hwc_data = CvUtils::convert_to_hwc_vec<float>(img_data, sample_channels, sample_size.height, sample_size.width);
        cv::Mat mid_image;
        if (sample_channels == 1) {
            mid_image = cv::Mat(sample_size, CV_32FC1, hwc_data.data());
        } else if (sample_channels == 3) {
            mid_image = cv::Mat(sample_size, CV_32FC3, hwc_data.data());
        } else {
            LOG(ERROR) << "not support image channels: " << sample_channels;
            continue;
        }
        mid_image.convertTo(mid_image, CV_8UC3);
        cv::cvtColor(mid_image, mid_image, cv::COLOR_RGB2BGR);
        internal_out.out_images.push_back(mid_image);
    }

    // transform output
    out = ddpm_sampler_impl::transform_output<OUTPUT>(internal_out);
    return sample_status;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param xt
 * @param predict_noise
 * @param timestep
 * @return
 */
template <typename INPUT, typename OUTPUT>
std::vector<float> DDPMSampler<INPUT, OUTPUT>::Impl::compute_predict_mean(
    const std::vector<float>& xt, const std::vector<float>& predict_noise, const int64_t timestep) {
    auto beta_t = _m_betas[timestep];
    auto alpha_t = 1.0 - beta_t;
    auto recipe_sqrt_alpha_t = 1.0 / std::sqrt((alpha_t));
    auto alpha_t_cumprod = _m_alpha_cumprod[timestep];
    auto recip_sqrt_one_minus_cumprod_alpha_t = 1.0 / std::sqrt(1.0 - alpha_t_cumprod);

    std::vector<float> result(xt.size());
    auto scale = beta_t * recip_sqrt_one_minus_cumprod_alpha_t;
    std::transform(predict_noise.begin(), predict_noise.end(), result.begin(), [scale](double x) { return x * scale; });
    std::transform(xt.begin(), xt.end(), result.begin(), result.begin(), std::minus<>());
    std::transform(result.begin(), result.end(), result.begin(), [recipe_sqrt_alpha_t](double x) { return x * recipe_sqrt_alpha_t;});

    return result;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param xt
 * @param t
 * @param is_last_step
 * @param use_fixed_noise
 * @return
 */
template <typename INPUT, typename OUTPUT>
std::vector<float> DDPMSampler<INPUT, OUTPUT>::Impl::p_sample_once(
    const std::vector<float>& xt, int64_t t, bool is_last_step, bool use_fixed_noise) {
    // compute predict noise
    std_ddpm_unet_input denoise_in;
    denoise_in.xt = xt;
    denoise_in.timestep = t;
    std_ddpm_unet_output denoise_out;
    auto status = _m_denoise_net->run(denoise_in, denoise_out);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "denoise model run session failed, status code: " << status;
        std::vector<float> result(xt.size());
        return result;
    }
    // compute predict mean
    std::vector<float> predict_mean = compute_predict_mean(xt, denoise_out.predict_noise, t);
    // compute sample result
    if (is_last_step) {
        return predict_mean;
    } else {
        // compute posterior_variance
        auto beta_t = _m_betas[t];
        auto alpha_t_cumprod = _m_alpha_cumprod[t];
        auto alpha_t_cumprod_pre = 1.0;
        if (t != 0) {
            alpha_t_cumprod_pre = _m_alpha_cumprod[t - 1];
        }
        auto posterior_variance = beta_t * (1.0 - alpha_t_cumprod_pre) / (1.0 - alpha_t_cumprod);
        // compute sample result
        std::vector<float> result(xt.size());
        auto scale = std::sqrt(posterior_variance);
        if (use_fixed_noise) {
            std::transform(_m_fixed_noise_for_psample.begin(), _m_fixed_noise_for_psample.end(), result.begin(),
                           [scale](double x) { return x * scale; });
        } else {
            // generate random noise
            std::vector<float> noise = ddpm_sampler_impl::generate_random_norm_vector(predict_mean.size());
            std::transform(noise.begin(), noise.end(), result.begin(), [scale](double x) { return x * scale; });
        }
        std::transform(predict_mean.begin(), predict_mean.end(), result.begin(), result.begin(), std::plus<>());
        return result;
    }
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param size
 * @param timesteps
 * @param channels
 * @param save_all_mid_results
 * @param use_fixed_noise_for_psample
 * @return
 */
template <typename INPUT, typename OUTPUT>
std::vector<std::vector<float> > DDPMSampler<INPUT, OUTPUT>::Impl::p_sample(
    const cv::Size &size, int timesteps, int channels, bool save_all_mid_results, bool use_fixed_noise_for_psample) {
    // construct xt random noise
    std::vector<float> xt = ddpm_sampler_impl::generate_random_norm_vector(size.area() * channels);
    std::vector<std::vector<float> > mid_sample_results;
    if (use_fixed_noise_for_psample) {
        _m_fixed_noise_for_psample = ddpm_sampler_impl::generate_random_norm_vector(xt.size());
    }

    // loop sample
    std::vector<double> steps = ddpm_sampler_impl::linspace(0.0, static_cast<double>(timesteps) - 1.0, static_cast<int>(timesteps));
    std::reverse(steps.begin(), steps.end());
    for (auto idx = 0; idx < steps.size(); ++idx) {
        auto t_step = static_cast<int64_t >(steps[idx]);
        bool is_last = t_step == 0;
        xt = p_sample_once(xt, t_step, is_last, use_fixed_noise_for_psample);
        if (save_all_mid_results) {
            mid_sample_results.push_back(xt);
        } else {
            if (is_last) {
                mid_sample_results.push_back(xt);
            }
        }
        _m_p_sample_bar->set_progress((static_cast<float>(idx + 1) / static_cast<float>(steps.size())) * 100.0f);
    }
    _m_p_sample_bar->mark_as_completed();

    return mid_sample_results;
}

/************* Export Function Sets *************/

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
DDPMSampler<INPUT, OUTPUT>::DDPMSampler() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
DDPMSampler<INPUT, OUTPUT>::~DDPMSampler() = default;

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param cfg
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DDPMSampler<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @return
 */
template <typename INPUT, typename OUTPUT>
bool DDPMSampler<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode DDPMSampler<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace diffusion
} // namespace models
} // namespace jinq