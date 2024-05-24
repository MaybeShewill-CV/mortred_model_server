/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: DDIMSampler.cpp
 * Date: 24-4-28
 ************************************************/

#include "ddim_sampler.h"

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

using jinq::models::io_define::diffusion::std_ddim_input;
using jinq::models::io_define::diffusion::std_ddim_output;
using jinq::models::io_define::diffusion::std_ddpm_unet_input;
using jinq::models::io_define::diffusion::std_ddpm_unet_output;
using DenoiseModelPtr = jinq::models::diffusion::DDPMUNet<std_ddpm_unet_input, std_ddpm_unet_output>;

namespace ddim_sampler_impl {

using internal_input = std_ddim_input;
using internal_output = std_ddim_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<std_ddim_input>::type>::value, internal_input>::type
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_ddim_output >::type>::value, std_ddim_output>::type
transform_output(const ddim_sampler_impl::internal_output& internal_out) {
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

} // namespace ddim_sampler_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class DDIMSampler<INPUT, OUTPUT>::Impl {
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
    int _m_timesteps = 1000;
    double _m_beta_start = 0.0;
    double _m_beta_end = 0.0;
    std::unique_ptr<indicators::BlockProgressBar> _m_p_sample_bar;

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
    std::vector<double> linear_beta_schedule(const int timesteps, const double beta_start=0.0001, const double beta_end=0.02) {
        double scale = 1000.0 / static_cast<double>(timesteps);
        double real_beta_start = scale * beta_start;
        double real_beta_end = scale * beta_end;
        return ddim_sampler_impl::linspace(real_beta_start, real_beta_end, static_cast<int>(timesteps));
    }

    /***
     *
     * @param timesteps
     * @param s
     * @return
     */
    std::vector<double> cos_beta_schedule(const int timesteps, const double s=0.008) {
        auto steps = timesteps + 1;
        auto t = ddim_sampler_impl::linspace(0, static_cast<double>(timesteps), static_cast<int>(steps));
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
    std::vector<double> sigmoid_beta_schedule(const int timesteps, const int start=-3, const int end=3, const int tau=1) {
        auto steps = timesteps + 1;
        auto t = ddim_sampler_impl::linspace(0, static_cast<double>(timesteps), static_cast<int>(steps));
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
     * @param total_steps
     * @param sample_steps
     * @param save_all_mid_results
     * @return
     */
    std::vector<std::tuple<std::vector<float>, std::vector<float> > > p_sample(
        std::vector<float>& xt, int total_steps, int sample_steps, float eta=0.0f, bool save_all_mid_results=false);

    /***
     *
     * @param xt
     * @param t
     * @param t_next
     * @return
     */
    std::tuple<std::vector<float>, std::vector<float> > p_sample_once(std::vector<float>& xt, int t, int t_next, float eta=0.0f);
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DDIMSampler<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose beta schedule type
    auto ddim_sampler_cfg = config.at("DDIM_SAMPLER");
    _m_timesteps = static_cast<int>(ddim_sampler_cfg["total_timesteps"].as_integer());
    _m_beta_start = ddim_sampler_cfg["beta_start"].as_floating();
    _m_beta_end = ddim_sampler_cfg["beta_end"].as_floating();
    _m_beta_schedule = _m_beta_schedule_type_map[ddim_sampler_cfg["beta_schedule"].as_string()];

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
        LOG(INFO) << "Successfully load ddim sampler";
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load ddim sampler";
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
StatusCode DDIMSampler<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // prepare input params
    auto transformed_input = ddim_sampler_impl::transform_input(in);
    auto sample_size = transformed_input.sample_size;
    auto total_steps = transformed_input.total_steps;
    auto sample_channels = transformed_input.channels;
    auto sample_steps = transformed_input.sample_steps;
    auto save_all_mid_results = transformed_input.save_all_mid_results;
    auto xt_data = transformed_input.xt_data;
    auto eta = transformed_input.eta;
    auto save_raw_output = transformed_input.save_raw_output;

    // p-sample loop
    std::vector<float> xt;
    if (xt_data == nullptr) {
        xt = ddim_sampler_impl::generate_random_norm_vector(sample_size.area() * sample_channels, 0.0, 1.0);
    } else {
        xt = std::vector<float>(xt_data, xt_data + sample_size.area() * sample_channels);
    }
    auto mid_sample_results = p_sample(xt, total_steps, sample_steps, eta, save_all_mid_results);

    // transform sampled results into cv::Mat
    ddim_sampler_impl::internal_output internal_out;
    StatusCode sample_status = StatusCode::OK;
    for (auto& img_tuple : mid_sample_results) {
        auto predict_x0 = std::get<0>(img_tuple);
        auto predict_xt = std::get<1>(img_tuple);
        if (save_raw_output) {
            internal_out.raw_predicted_x0.push_back(predict_x0);
            internal_out.raw_sampled_images.push_back(predict_xt);
        }
        // rescale image data to [0, 255]
        for (auto idx = 0; idx < predict_x0.size(); ++idx) {
            predict_x0[idx] = std::clamp(predict_x0[idx], -1.0f, 1.0f);
            predict_x0[idx] = (predict_x0[idx] + 1.0f) * 0.5f * 255.0f + 0.5f;
            predict_x0[idx] = std::clamp(predict_x0[idx], 0.0f, 255.0f);

            predict_xt[idx] = std::clamp(predict_xt[idx], -1.0f, 1.0f);
            predict_xt[idx] = (predict_xt[idx] + 1.0f) * 0.5f * 255.0f + 0.5f;
            predict_xt[idx] = std::clamp(predict_xt[idx], 0.0f, 255.0f);
        }
        // assign output predict x0 images
        auto hwc_data = CvUtils::convert_to_hwc_vec<float>(predict_x0, sample_channels, sample_size.height, sample_size.width);
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
        internal_out.predicted_x0.push_back(mid_image);
        // assign output predict samples images
        hwc_data = CvUtils::convert_to_hwc_vec<float>(predict_xt, sample_channels, sample_size.height, sample_size.width);
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
        internal_out.sampled_images.push_back(mid_image);
    }

    // transform output
    out = ddim_sampler_impl::transform_output<OUTPUT>(internal_out);
    return sample_status;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param xt
 * @param total_steps
 * @param sample_steps
 * @param eta
 * @param save_all_mid_results
 * @return
 */
template <typename INPUT, typename OUTPUT>
std::vector<std::tuple<std::vector<float>, std::vector<float> > > DDIMSampler<INPUT, OUTPUT>::Impl::p_sample(
    std::vector<float>& xt, int total_steps, int sample_steps, float eta, bool save_all_mid_results) {
    // loop sample
    std::vector<std::tuple<std::vector<float>, std::vector<float> > > mid_sample_results;
    std::vector<double> steps = ddim_sampler_impl::linspace(0.0, static_cast<double>(total_steps) - 1.0, sample_steps);
    std::reverse(steps.begin(), steps.end());
    for (auto idx = 0; idx < steps.size(); ++idx) {
        auto t = steps[idx];
        bool is_last = t == 0;
        auto t_next = is_last ? t : steps[idx + 1];
        auto sample_result = p_sample_once(xt, t, t_next, eta);
        xt = std::get<1>(sample_result);
        if (save_all_mid_results) {
            mid_sample_results.push_back(sample_result);
        } else {
            if (is_last) {
                mid_sample_results.push_back(sample_result);
            }
        }
        _m_p_sample_bar->set_progress((static_cast<float>(idx + 1) / static_cast<float>(steps.size())) * 100.0f);
    }
    _m_p_sample_bar->mark_as_completed();

    return mid_sample_results;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param xt
 * @param t
 * @param t_next
 * @return
 */
template <typename INPUT, typename OUTPUT>
std::tuple<std::vector<float>, std::vector<float> > DDIMSampler<INPUT, OUTPUT>::Impl::p_sample_once(
    std::vector<float>& xt, int t, int t_next, float eta) {
    // compute predict noise
    std_ddpm_unet_input denoise_in;
    denoise_in.xt = xt;
    denoise_in.timestep = t;
    std_ddpm_unet_output denoise_out;
    auto status = _m_denoise_net->run(denoise_in, denoise_out);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "denoise model run session failed, status code: " << status;
        auto result = std::make_tuple(std::vector<float>(xt.size()), std::vector<float>(xt.size()));
        return result;
    }
    auto& predict_noise = denoise_out.predict_noise;

    // compute equation(12) part 1 "predicted x0" in origin paper
    auto alpha_t_cumprod = _m_alpha_cumprod[t];
    auto alpha_t_next_cumprod = _m_alpha_cumprod[t_next];
    std::vector<float> x0_t(xt.size());
    for (auto idx = 0; idx < xt.size(); ++idx) {
        auto sqrt_alpha_t_cumprod = static_cast<float>(std::sqrt(alpha_t_cumprod));
        auto val = static_cast<float>(std::sqrt(1 - alpha_t_cumprod));
        val = predict_noise[idx] * val;
        val = xt[idx] - val;
        val = val / (sqrt_alpha_t_cumprod + 0.0000001f);
        x0_t[idx] = val;
    }

    // compute sigma_t
    auto sigma_t = (1 - alpha_t_next_cumprod) / (1 - alpha_t_cumprod) * (1 - alpha_t_cumprod / alpha_t_next_cumprod);
    sigma_t =  eta * std::sqrt(sigma_t);

    // compute equation(12) part 2 "direction pointing to x t" in origin paper
    std::vector<float> direction_to_xt(xt.size());
    for (auto idx = 0; idx < xt.size(); ++idx) {
        auto noise = predict_noise[idx];
        auto val = static_cast<float>(std::sqrt(1.0f - alpha_t_next_cumprod - std::pow(sigma_t, 2)));
        val *= noise;
        direction_to_xt[idx] = val;
    }

    // compute equation(12) part 3 "random noise" in origin paper
    auto random_noise = ddim_sampler_impl::generate_random_norm_vector(xt.size());
    for (auto& val : random_noise) {
        val *= static_cast<float>(sigma_t);
    }

    // compute xt-1
    std::vector<float> xt_next(xt.size());
    for (auto idx = 0; idx < xt.size(); ++idx) {
        auto x0_t_v = static_cast<float>(std::sqrt(alpha_t_next_cumprod)) * x0_t[idx];
        auto direct_to_xt_v = direction_to_xt[idx];
        auto random_noise_v = random_noise[idx];
        xt_next[idx] = x0_t_v + direct_to_xt_v + random_noise_v;
    }

    return std::make_tuple(x0_t, xt_next);
}

/************* Export Function Sets *************/

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
DDIMSampler<INPUT, OUTPUT>::DDIMSampler() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
DDIMSampler<INPUT, OUTPUT>::~DDIMSampler() = default;

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param cfg
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DDIMSampler<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @return
 */
template <typename INPUT, typename OUTPUT>
bool DDIMSampler<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode DDIMSampler<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace diffusion
} // namespace models
} // namespace jinq
