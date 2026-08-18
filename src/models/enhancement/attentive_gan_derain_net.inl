/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: attentive_gan_derain_net.inl
* Date: 22-6-14
************************************************/

#include "attentive_gan_derain_net.h"
#include "models/cv_image_input.h"

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::mat_input;

namespace enhancement {

using jinq::models::io_define::enhancement::std_enhancement_output;

namespace attentiveganderain_impl {

using internal_input = mat_input;
using internal_output = std_enhancement_output;

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template<typename INPUT>
internal_input transform_input(const INPUT& in) {
    internal_input result{};
    result.input_image = jinq::models::cv_input::load_image(in);
    return result;
}

/***
 * transform different type of internal output into external output
 * @tparam EXTERNAL_OUTPUT
 * @tparam dummy
 * @param in
 * @return
 */
template <typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_enhancement_output>::type>::value, std_enhancement_output>::type
transform_output(const attentiveganderain_impl::internal_output &internal_out) {
    return internal_out;
}

} // namespace attentiveganderain_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT> class AttentiveGanDerain<INPUT, OUTPUT>::Impl {
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
    Impl(const Impl &transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl &operator=(const Impl &transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const toml::table &config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT &in, OUTPUT &out);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized; 
    };

  private:
    jinq::models::MnnNet _m_net;
    // input image size
    cv::Size _m_input_size_user = cv::Size();
    // input node size
    cv::Size _m_input_size_host = cv::Size();
    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @param input_image
     * @return
     */
    cv::Mat preprocess_image(const cv::Mat &input_image) const;

    /***
     *
     * @return
     */
    cv::Mat postprocess() const;
};

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode AttentiveGanDerain<INPUT, OUTPUT>::Impl::init(const toml::table &config) {
    if (!config.contains("ATTENTIVEGANDERAIN")) {
        LOG(ERROR) << "Config missing ATTENTIVEGANDERAIN section please check config file";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* cfg_content_ptr = config["ATTENTIVEGANDERAIN"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section ATTENTIVEGANDERAIN missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    auto init_status = _m_net.init(cfg_content, {"input_tensor"}, {"final_output"});
    if (init_status != StatusCode::OK) {
        _m_successfully_initialized = false;
        return init_status;
    }
    _m_input_size_host.width = _m_net.input("input_tensor")->width();
    _m_input_size_host.height = _m_net.input("input_tensor")->height();

    if (!cfg_content.contains("model_input_image_size")) {
        _m_input_size_user.width = 320;
        _m_input_size_user.height = 240;
    } else {
        _m_input_size_user.width = static_cast<int>(cfg_content["model_input_image_size"][1].value_or<int64_t>(0));
        _m_input_size_user.height = static_cast<int>(cfg_content["model_input_image_size"][0].value_or<int64_t>(0));
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "Attentive gan derain model initialization complete!!!";
    return StatusCode::OK;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode AttentiveGanDerain<INPUT, OUTPUT>::Impl::run(const INPUT &in, OUTPUT &out) {
    // transform external input into internal input
    auto internal_in = attentiveganderain_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    auto preprocessed_image = preprocess_image(internal_in.input_image);
    // run session
    MNN::Tensor input_tensor_user(_m_net.input("input_tensor"), MNN::Tensor::DimensionType::TENSORFLOW);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!CvUtils::copy_image_to_tensor(input_tensor_data, preprocessed_image, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("input_tensor")->copyFromHostTensor(&input_tensor_user);
    _m_net.run_session();
    // postprocess
    cv::Mat output_image = postprocess();
    if (output_image.size() != _m_input_size_user) {
        cv::resize(output_image, output_image, _m_input_size_user);
    }
    attentiveganderain_impl::internal_output internal_out;
    output_image.copyTo(internal_out.enhancement_result);
    // transform output
    out = attentiveganderain_impl::transform_output<OUTPUT>(internal_out);
    return StatusCode::OK;
}

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT>
cv::Mat AttentiveGanDerain<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat &input_image) const {
    // resize image
    cv::Mat tmp;
    if (input_image.size() != _m_input_size_host) {
        cv::resize(input_image, tmp, _m_input_size_host);
    } else {
        tmp = input_image;
    }

    // normalize
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    tmp /= 127.5;
    cv::subtract(tmp, cv::Scalar(1.0, 1.0, 1.0), tmp);
    return tmp;
}

/***
 * @return
 */
template <typename INPUT, typename OUTPUT> cv::Mat AttentiveGanDerain<INPUT, OUTPUT>::Impl::postprocess() const {
    // convert tensor format
    MNN::Tensor output_tensor_user(_m_net.output("final_output"), _m_net.output("final_output")->getDimensionType());
    _m_net.output("final_output")->copyToHostTensor(&output_tensor_user);
    auto host_data = output_tensor_user.host<float>();

    // construct result image
    cv::Mat output_feats(_m_input_size_host, CV_32FC3, host_data);
    std::vector<cv::Mat> output_feats_split;
    cv::split(output_feats, output_feats_split);
    auto b_max_value = *std::max_element(output_feats_split[0].begin<float>(), output_feats_split[0].end<float>());
    auto b_min_value = *std::min_element(output_feats_split[0].begin<float>(), output_feats_split[0].end<float>());
    auto g_max_value = *std::max_element(output_feats_split[1].begin<float>(), output_feats_split[1].end<float>());
    auto g_min_value = *std::min_element(output_feats_split[1].begin<float>(), output_feats_split[1].end<float>());
    auto r_max_value = *std::max_element(output_feats_split[2].begin<float>(), output_feats_split[2].end<float>());
    auto r_min_value = *std::min_element(output_feats_split[2].begin<float>(), output_feats_split[2].end<float>());
    cv::Mat output_image(_m_input_size_host, CV_8UC3);
    for (auto row = 0; row < output_image.size().height; ++row) {
        for (auto col = 0; col < output_image.size().width; ++col) {
            float b_feats_val = output_feats.at<cv::Vec3f>(row, col)[0];
            float g_feats_val = output_feats.at<cv::Vec3f>(row, col)[1];
            float r_feats_val = output_feats.at<cv::Vec3f>(row, col)[2];

            auto b_scale_val = static_cast<float>((b_feats_val - b_min_value) * 255.0 / (b_max_value - b_min_value));
            auto g_scale_val = static_cast<float>((g_feats_val - g_min_value) * 255.0 / (g_max_value - g_min_value));
            auto r_scale_val = static_cast<float>((r_feats_val - r_min_value) * 255.0 / (r_max_value - r_min_value));

            output_image.at<cv::Vec3b>(row, col)[0] = static_cast<uint8_t>(b_scale_val);
            output_image.at<cv::Vec3b>(row, col)[1] = static_cast<uint8_t>(g_scale_val);
            output_image.at<cv::Vec3b>(row, col)[2] = static_cast<uint8_t>(r_scale_val);
        }
    }
    return output_image;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> AttentiveGanDerain<INPUT, OUTPUT>::AttentiveGanDerain() { 
    _m_pimpl = std::make_unique<Impl>(); 
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> AttentiveGanDerain<INPUT, OUTPUT>::~AttentiveGanDerain() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT> StatusCode AttentiveGanDerain<INPUT, OUTPUT>::init(const toml::table &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT> bool AttentiveGanDerain<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode AttentiveGanDerain<INPUT, OUTPUT>::run(const INPUT &input, OUTPUT &output) {
    return _m_pimpl->run(input, output);
}

} // namespace enhancement
} // namespace models
} // namespace jinq
