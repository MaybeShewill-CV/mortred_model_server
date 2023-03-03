/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: modnet_matthing.inl
* Date: 22-7-19
************************************************/

#include "modnet_matting.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "common/base64.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::Base64;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;
using jinq::common::Timestamp;

namespace matting {
using jinq::models::io_define::matting::std_matting_output;

namespace modnet_impl {

struct internal_input {
    cv::Mat input_image;
};

using internal_output = std_matting_output;

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
    internal_input result{};
    result.input_image = in.input_image;
    return result;
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
    auto image_decode_string = jinq::common::Base64::base64_decode(in.input_image_content);
    std::vector<uchar> image_vec_data(image_decode_string.begin(), image_decode_string.end());

    if (image_vec_data.empty()) {
        DLOG(WARNING) << "image data empty";
        return result;
    } else {
        cv::Mat ret;
        cv::imdecode(image_vec_data, cv::IMREAD_UNCHANGED).copyTo(result.input_image);
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_matting_output>::type>::value, std_matting_output>::type
transform_output(const modnet_impl::internal_output& internal_out) {
    std_matting_output result;
    internal_out.matting_result.copyTo(result.matting_result);
    return result;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class ModNetMatting<INPUT, OUTPUT>::Impl {
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
    // MNN Interpreter
    std::unique_ptr<MNN::Interpreter> _m_net = nullptr;
    // MNN Session
    MNN::Session* _m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor* _m_input_tensor = nullptr;
    // MNN Loc Output tensor node
    MNN::Tensor* _m_output_tensor = nullptr;
    // MNN backend thread nums
    uint _m_threads_nums = 4;
    // input tensor size
    cv::Size _m_input_size_user = cv::Size();
    // input node size
    cv::Size _m_input_size_host = cv::Size();
    // init flag
    bool _m_successfully_initialized = false;

private:
    /***
     *
     * @param input_image :
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;
};

/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode ModNetMatting<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("MODNET_MATTING")) {
        LOG(ERROR) << "Config file missing MODNET_MATTING section, please check";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("MODNET_MATTING");

    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config file doesn\'t contain model_threads_num field, using default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = cfg_content.at("model_threads_num").as_integer();
    }

    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config file doesn\'t contain model_file_path field, please check again";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "ModNet model file path: " << _m_model_file_path << ", not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));

    if (nullptr == _m_net) {
        LOG(ERROR) << "Create ModNet Interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    MNN::ScheduleConfig mnn_config;

    if (!cfg_content.contains("compute_backend")) {
        LOG(WARNING) << "Config doesn\'t contain compute_backend field, using default backend cpu";
        LOG(INFO) << "Using CPU compute backend...";
        mnn_config.type = MNN_FORWARD_CPU;
    } else {
        std::string compute_backend = cfg_content.at("compute_backend").as_string();
        if (std::strcmp(compute_backend.c_str(), "cuda") == 0) {
            mnn_config.type = MNN_FORWARD_CUDA;
        } else if (std::strcmp(compute_backend.c_str(), "cpu") == 0) {
            mnn_config.type = MNN_FORWARD_CPU;
        } else {
            LOG(WARNING) << "Compute backend not support, using default backend cpu";
            LOG(INFO) << "Using CPU compute backend...";
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

    if (nullptr == _m_session) {
        LOG(ERROR) << "Create modnet Session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_tensor = _m_net->getSessionInput(_m_session, "img");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "sigmoid_2.tmp_0");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch modnet Input Node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch modnet Output Node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

    _m_successfully_initialized = true;
    LOG(INFO) << "Modnet matting model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
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
cv::Mat ModNetMatting<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
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
StatusCode ModNetMatting<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = modnet_impl::transform_input(in);

    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run session
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, input_chw_image_data.data(), input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);

    // fetch net output
    MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    _m_output_tensor->copyToHostTensor(&output_tensor_user);
    auto host_data = output_tensor_user.host<float>();
    cv::Mat result_image(_m_input_size_host, CV_32FC1, host_data);
    cv::resize(result_image, result_image, _m_input_size_user, 0.0, 0.0, cv::INTER_LINEAR);
    result_image *= 255.0;
    result_image.convertTo(result_image, CV_8UC1);

    // transform internal output into external output
    modnet_impl::internal_output internal_out;
    internal_out.matting_result = result_image;
    out = modnet_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
ModNetMatting<INPUT, OUTPUT>::ModNetMatting() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
ModNetMatting<INPUT, OUTPUT>::~ModNetMatting() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode ModNetMatting<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool ModNetMatting<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode ModNetMatting<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}