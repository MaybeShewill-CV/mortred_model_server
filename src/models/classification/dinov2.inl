/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dinov2.inl
* Date: 23-6-12
************************************************/

#include "dinov2.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::Base64;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;

namespace classification {

using jinq::models::io_define::classification::std_classification_output;

namespace dinov2_impl {

using internal_input = mat_input;
using internal_output = std_classification_output;

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template<typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<file_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    internal_input result{};
    if (!FilePathUtil::is_file_exist(in.input_image_path)) {
        DLOG(WARNING) << "input image: " << in.input_image_path << " not exist";
        return result;
    }
    result.input_image = cv::imread(in.input_image_path, cv::IMREAD_UNCHANGED);
    return result;
}

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template<typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<mat_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    return in;
}

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template<typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<base64_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    internal_input result{};
    auto image = CvUtils::decode_base64_str_into_cvmat(in.input_image_content);
    if (!image.data || image.empty()) {
        DLOG(WARNING) << "image data empty";
        return result;
    } else {
        result.input_image = image;
        return result;
    }
}

/***
* transform different type of internal output into external output
* @tparam EXTERNAL_OUTPUT
* @tparam dummy
* @param in
* @return
*/
template<typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_classification_output>::type>::value, std_classification_output>::type
transform_output(const dinov2_impl::internal_output& internal_out) {
    return internal_out;
}


}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class Dinov2<INPUT, OUTPUT>::Impl {
public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        if (_m_net != nullptr && _m_session != nullptr) {
            _m_net->releaseModel();
            _m_net->releaseSession(_m_session);
        }
    }

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
    StatusCode init(const decltype(toml::parse(""))& config);

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
    std::string _m_model_file_path;
    // mnn interpreter
    MNN::Interpreter* _m_net = nullptr;
    // mnn session
    MNN::Session* _m_session = nullptr;
    // mnn session config
    MNN::ScheduleConfig _m_session_config;
    // mnn input tensor
    MNN::Tensor* _m_input_tensor = nullptr;
    // mnn output tensor
    MNN::Tensor* _m_output_tensor = nullptr;
    // mnn backend threads nums
    int _m_threads_nums = 4;
    // mnn input tensor size
    cv::Size _m_input_tensor_size = cv::Size(224, 224);
    // flag
    bool _m_successfully_initialized = false;

private:
    /***
     * preprocess image
     * @param input_image : input image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;
};


/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode Dinov2<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("DINOV2")) {
        LOG(ERROR) << "Config file does not contain DINOV2 section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("DINOV2");

    // init Interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "Dinov2 classification model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = MNN::Interpreter::createFromFile(_m_model_file_path.c_str());
    if (_m_net == nullptr) {
        LOG(ERROR) << "Create Interpreter failed, model file path: " << _m_model_file_path;
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << R"(Config file parse error, doesn't not have field "model_threads_nums", use default value 4)";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = static_cast<int>(cfg_content.at("model_threads_num").as_integer());
    }

    // init session
    MNN::ScheduleConfig mnn_config;
    if (!cfg_content.contains("compute_backend")) {
        LOG(WARNING) << "Config doesn\'t have compute_backend field default cpu";
        mnn_config.type = MNN_FORWARD_CPU;
    } else {
        std::string compute_backend = cfg_content.at("compute_backend").as_string();

        if (std::strcmp(compute_backend.c_str(), "cuda") == 0) {
            mnn_config.type = MNN_FORWARD_CUDA;
        } else if (std::strcmp(compute_backend.c_str(), "cpu") == 0) {
            mnn_config.type = MNN_FORWARD_CPU;
        } else {
            LOG(WARNING) << "not supported compute backend use default cpu instead";
            mnn_config.type = MNN_FORWARD_CPU;
        }
    }
    mnn_config.numThread = _m_threads_nums;
    MNN::BackendConfig backend_config;
    if (!cfg_content.contains("backend_precision_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_precision_mode field default Precision_Normal";
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(cfg_content.at("backend_precision_mode").as_integer());
    }
    if (!cfg_content.contains("backend_power_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_power_mode field default Power_Normal";
        backend_config.power = MNN::BackendConfig::Power_Normal;
    } else {
        backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(cfg_content.at("backend_power_mode").as_integer());
    }
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);
    if (_m_session == nullptr) {
        LOG(ERROR) << "Create Session failed, model file path: " << _m_model_file_path;
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init input/output tensor
    _m_input_tensor = _m_net->getSessionInput(_m_session, "input_images");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "cls_tokens");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch Dinov2 classification model input node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch Dinov2 classification model output node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size.height = _m_input_tensor->shape()[2];
    _m_input_tensor_size.width = _m_input_tensor->shape()[3];

    _m_successfully_initialized = true;
    LOG(INFO) << "Dinov2 classification model initialization complete !!!";
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
StatusCode Dinov2<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = dinov2_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    auto preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run session
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, input_chw_image_data.data(), input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);

    // decode output tensor
    MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    _m_output_tensor->copyToHostTensor(&output_tensor_user);
    auto* host_data = output_tensor_user.host<float>();

    // transform output
    dinov2_impl::internal_output internal_out;
    for (auto index = 0; index < output_tensor_user.elementSize(); ++index) {
        internal_out.scores.push_back(host_data[index]);
    }
    auto max_score = std::max_element(host_data, host_data + output_tensor_user.elementSize());
    auto cls_id = static_cast<int>(std::distance(host_data, max_score));
    internal_out.class_id = cls_id;
    out = dinov2_impl::transform_output<OUTPUT>(internal_out);

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
cv::Mat Dinov2<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
    cv::Mat result;
    cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);
    cv::resize(result, result, _m_input_tensor_size);
    result.convertTo(result, CV_32FC3);

    cv::divide(result, 255.0, result);
    cv::subtract(result, cv::Scalar(0.48145466, 0.4578275, 0.40821073), result);
    cv::divide(result, cv::Scalar(0.26862954, 0.26130258, 0.27577711), result);

    return result;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
Dinov2<INPUT, OUTPUT>::Dinov2() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
Dinov2<INPUT, OUTPUT>::~Dinov2() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode Dinov2<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool Dinov2<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode Dinov2<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}