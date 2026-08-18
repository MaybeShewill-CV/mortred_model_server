/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: pp_matthing.inl
* Date: 22-7-19
************************************************/

#include "pp_matting.h"
#include "models/cv_image_input.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;

namespace matting {
using jinq::models::io_define::matting::std_matting_output;

namespace ppmat_impl {

using internal_input = mat_input;
using internal_output = std_matting_output;

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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_matting_output>::type>::value, std_matting_output>::type
transform_output(const ppmat_impl::internal_output& internal_out) {
    return internal_out;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class PPMatting<INPUT, OUTPUT>::Impl {
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
    // input image size
    cv::Size _m_input_size_user = cv::Size();
    // input node size
    cv::Size _m_input_size_host = cv::Size();
    // flag
    bool _m_successfully_initialized = false;

private:

    /***
     *
     * @param input_image
     * @return
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;
};

/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode PPMatting<INPUT, OUTPUT>::Impl::init(const toml::table& config) {
    if (!config.contains("PP_MATTING")) {
        LOG(ERROR) << "Config file missing PP_MATTING section, please check";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* cfg_content_ptr = config["PP_MATTING"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section PP_MATTING missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    auto init_status = _m_net.init(cfg_content, {"img"}, {});
    if (init_status != StatusCode::OK) {
        _m_successfully_initialized = false;
        return init_status;
    }
    if (_m_net.first_output() == nullptr) {
        LOG(ERROR) << "Fetch PPMatting Output Node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.width = _m_net.input("img")->width();
    _m_input_size_host.height = _m_net.input("img")->height();

    _m_successfully_initialized = true;
    LOG(INFO) << "PPMatting matting model initialization complete!!!";
    return StatusCode::OK;
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param input_image
* @return
*/
template<typename INPUT, typename OUTPUT>
cv::Mat PPMatting<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
    cv::Mat tmp;
    // swap channles
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);

    // resize image
    if (tmp.size() != _m_input_size_host) {
        cv::resize(tmp, tmp, _m_input_size_host);
    }

    // convert image data type
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    // normalize image
    tmp /= 255.0;
    cv::subtract(tmp, cv::Scalar(0.5, 0.5, 0.5), tmp);
    cv::divide(tmp, cv::Scalar(0.5, 0.5, 0.5), tmp);

    return tmp;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode PPMatting<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = ppmat_impl::transform_input(in);

    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run session
    MNN::Tensor input_tensor_user(_m_net.input("img"), MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!CvUtils::copy_image_to_tensor(input_tensor_data, input_chw_image_data, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("img")->copyFromHostTensor(&input_tensor_user);
    _m_net.run_session();

    // fetch net output
    MNN::Tensor output_tensor_user(_m_net.first_output(), MNN::Tensor::DimensionType::CAFFE);
    _m_net.first_output()->copyToHostTensor(&output_tensor_user);
    auto host_data = output_tensor_user.host<float>();
    cv::Mat result_image(_m_input_size_host, CV_32FC1, host_data);
    cv::resize(result_image, result_image, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);
    result_image *= 255.0;
    result_image.convertTo(result_image, CV_32SC1);

    // transform internal output into external output
    ppmat_impl::internal_output internal_out;
    internal_out.matting_result = result_image;
    out = ppmat_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
PPMatting<INPUT, OUTPUT>::PPMatting() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
PPMatting<INPUT, OUTPUT>::~PPMatting() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode PPMatting<INPUT, OUTPUT>::init(const toml::table& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool PPMatting<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode PPMatting<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}