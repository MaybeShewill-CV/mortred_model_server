/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: enlightengan.inl
* Date: 22-6-13
************************************************/

#include "enlightengan.h"
#include "models/cv_image_input.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::cv_utils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::base64;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;

namespace enhancement {

using jinq::models::io_define::enhancement::std_enhancement_output;

namespace enlightengan_impl {

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
template<typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_enhancement_output>::type>::value, std_enhancement_output>::type
transform_output(const enlightengan_impl::internal_output& internal_out) {
    return internal_out;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class EnlightenGan<INPUT, OUTPUT>::Impl {
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
    StatusCode init(const toml::table& config);

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
    jinq::models::MnnNet _m_net;
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
    void preprocess_image(const cv::Mat& input_image, cv::Mat& output_src, cv::Mat& output_gray) const;
};


/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode EnlightenGan<INPUT, OUTPUT>::Impl::init(const toml::table& config) {
    if (!config.contains("ENLIGHTENGAN")) {
        LOG(ERROR) << "Config file missing ENLIGHTENGAN section please check";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* cfg_content_ptr = config["ENLIGHTENGAN"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section ENLIGHTENGAN missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    auto init_status = _m_net.tomlt(cfg_content, {"input_src", "input_gray"}, {"output"});
    if (init_status != StatusCode::OK) {
        _m_successfully_initialized = false;
        return init_status;
    }
    _m_input_size_host.width = _m_net.input("input_src")->width();
    _m_input_size_host.height = _m_net.input("input_src")->height();
    _m_successfully_initialized = true;

    LOG(INFO) << "Enlighten-gan enhancement model initialization complete!!!";
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode EnlightenGan<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = enlightengan_impl::transform_input(in);

    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    if (!internal_in.input_image.data || internal_in.input_image.empty() ||
            internal_in.input_image.size().height < 10 || internal_in.input_image.size().width < 10) {
        LOG(ERROR) << "invalid image data or empty image";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    if (internal_in.input_image.channels() != 3 && internal_in.input_image.channels() != 4) {
        LOG(ERROR) << "input image should have 3 or 4 channels, but got: "
                    << internal_in.input_image.channels() << " instead";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    if (internal_in.input_image.size() != _m_input_size_host) {
        _m_input_size_host.height = static_cast<int>(std::ceil(internal_in.input_image.size().height / 16) * 16);
        _m_input_size_host.width = static_cast<int>(std::ceil(internal_in.input_image.size().width / 16) * 16);
        _m_net.resize_tensor(_m_net.input("input_src"), 1, 3, _m_input_size_host.height, _m_input_size_host.width);
        _m_net.resize_tensor(_m_net.input("input_gray"), 1, 1, _m_input_size_host.height, _m_input_size_host.width);
    }
    cv::Mat input_src;
    cv::Mat input_gray;
    preprocess_image(internal_in.input_image, input_src, input_gray);
    auto input_src_chw_data = cv_utils::convert_to_chw_vec(input_src);

    // run session
    MNN::Tensor input_tensor_user_src(_m_net.input("input_src"), MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user_src.host<float>();
    auto input_tensor_size = input_tensor_user_src.size();
    if (!cv_utils::copy_image_to_tensor(input_tensor_data, input_src_chw_data, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("input_src")->copyFromHostTensor(&input_tensor_user_src);

    MNN::Tensor input_tensor_user_gray(_m_net.input("input_gray"), MNN::Tensor::DimensionType::CAFFE);
    input_tensor_data = input_tensor_user_gray.host<float>();
    input_tensor_size = input_tensor_user_gray.size();
    if (!cv_utils::copy_image_to_tensor(input_tensor_data, input_gray, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("input_gray")->copyFromHostTensor(&input_tensor_user_gray);
    _m_net.run_session();

    // decode output tensor
    MNN::Tensor output_tensor_user(_m_net.output("output"), MNN::Tensor::DimensionType::CAFFE);
    _m_net.output("output")->copyToHostTensor(&output_tensor_user);
    auto host_data = output_tensor_user.host<float>();
    auto element_size = output_tensor_user.elementSize();
    std::vector<uchar> output_img_data;
    output_img_data.resize(element_size);

    for (auto row = 0; row < _m_input_size_host.height; ++row) {
        for (auto col = 0; col < _m_input_size_host.width; ++col) {
            for (auto c = 0; c < 3; ++c) {
                auto hwc_idx = row * _m_input_size_host.width * 3 + col * 3 + c;
                auto chw_idx = c * _m_input_size_host.height * _m_input_size_host.width + row * _m_input_size_host.width + col;
                auto pix_val_f = (host_data[chw_idx] + 1.0) * 255.0 / 2.0;
                if (pix_val_f < 0.0) {
                    pix_val_f = 0.0;
                }
                if (pix_val_f >= 255) {
                    pix_val_f = 255.0;
                }
                auto pix_val = static_cast<uchar>(pix_val_f);
                output_img_data[hwc_idx] = pix_val;
            }
        }
    }
    
    enlightengan_impl::internal_output internal_out;
    cv::Mat result_image(_m_input_size_host, CV_8UC3, output_img_data.data());
    cv::cvtColor(result_image, internal_out.enhancement_result, cv::COLOR_RGB2BGR);
    if (internal_out.enhancement_result.size() != internal_in.input_image.size()) {
        cv::resize(internal_out.enhancement_result, internal_out.enhancement_result, internal_in.input_image.size());
    }

    // refine output image
    if (internal_in.input_image.channels() == 4) {
        std::vector<cv::Mat> input_image_split;
        cv::split(internal_in.input_image, input_image_split);

        std::vector<cv::Mat> output_image_split;
        cv::split(internal_out.enhancement_result, output_image_split);
        output_image_split.push_back(input_image_split[3]);
        cv::merge(output_image_split, internal_out.enhancement_result);
    }

    // transform internal output into external output
    out = enlightengan_impl::transform_output<OUTPUT>(internal_out);
    return StatusCode::OK;
}


/***
*
* @param input_image
* @param output_src
* @param output_gray
*/
template<typename INPUT, typename OUTPUT>
void EnlightenGan<INPUT, OUTPUT>::Impl::preprocess_image(
    const cv::Mat& input_image, cv::Mat& output_src,
    cv::Mat& output_gray) const {
    input_image.copyTo(output_src);

    // resize image
    if (output_src.channels() == 4) {
        cv::cvtColor(output_src, output_src, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(output_src, output_src, cv::COLOR_BGR2RGB);
    }
    if (input_image.size() != _m_input_size_host) {
        cv::resize(input_image, output_src, _m_input_size_host);
    }

    // normalize
    if (output_src.type() != CV_32FC3) {
        output_src.convertTo(output_src, CV_32FC3);
    }

    output_src /= 255.0;
    cv::subtract(output_src, cv::Scalar(0.5, 0.5, 0.5), output_src);
    cv::divide(output_src, cv::Scalar(0.5, 0.5, 0.5), output_src);

    // make gray output
    std::vector<cv::Mat> src_split;
    cv::split(output_src, src_split);
    output_gray = 1.0 - (0.299 * (src_split[0] + 1.0) + 0.587 * (src_split[1] + 1.0) + 0.114 * (src_split[2] + 1.0)) * 0.5;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
EnlightenGan<INPUT, OUTPUT>::EnlightenGan() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
EnlightenGan<INPUT, OUTPUT>::~EnlightenGan() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode EnlightenGan<INPUT, OUTPUT>::init(const toml::table& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool EnlightenGan<INPUT, OUTPUT>::is_successfully_initialized() const {
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
template<typename INPUT, typename OUTPUT>
StatusCode EnlightenGan<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}
