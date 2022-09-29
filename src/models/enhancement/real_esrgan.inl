/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: RealEsrGan.cpp
* Date: 22-9-29
************************************************/

#include "real_esrgan.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/cv_utils.h"
#include "common/base64.h"
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

namespace enhancement {

using jinq::models::io_define::enhancement::std_enhancement_output;

namespace real_esrgan_impl {

struct internal_input {
    cv::Mat input_image;
};

using internal_output = std_enhancement_output;

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
    auto image = CvUtils::decode_base64_str_into_cvmat(in.input_image_content);

    if (!image.data || image.empty()) {
        DLOG(WARNING) << "image data empty";
        return result;
    } else {
        image.copyTo(result.input_image);
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_enhancement_output>::type>::value, std_enhancement_output>::type
transform_output(const real_esrgan_impl::internal_output& internal_out) {
    std_enhancement_output result;
    internal_out.enhancement_result.copyTo(result.enhancement_result);
    return result;
}


}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class RealEsrGan<INPUT, OUTPUT>::Impl {
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
    // 模型文件存储路径
    std::string _m_model_file_path;
    // MNN Interpreter
    std::unique_ptr<MNN::Interpreter> _m_net = nullptr;
    // MNN Session
    MNN::Session* _m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor* _m_input_tensor = nullptr;
    // MNN Loc Output tensor node
    MNN::Tensor* _m_output_tensor = nullptr;
    // MNN后端使用线程数
    int _m_threads_nums = 4;
    //　计算图定义的输入node尺寸
    cv::Size _m_input_size_host = cv::Size();
    // 是否成功初始化标志位
    bool _m_successfully_initialized = false;

private:
    /***
    *
    * @param input_image
    * @return
    */
    static cv::Mat preprocess_image(const cv::Mat& input_image);
};


/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode RealEsrGan<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("RealEsrGan")) {
        LOG(ERROR) << "Config file missing RealEsrGan section please check";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("RealEsrGan");

    // init threads
    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config doesn\'t have model_threads_num field default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = static_cast<int>(cfg_content.at("model_threads_num").as_integer());
    }

    // init Interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "RealEsrGan model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(
                 MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));

    if (nullptr == _m_net) {
        LOG(ERROR) << "Create real-esrgan enhancement model interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init Session
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
    backend_config.precision = MNN::BackendConfig::Precision_High;
    backend_config.power = MNN::BackendConfig::Power_High;
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);

    if (nullptr == _m_session) {
        LOG(ERROR) << "Create real-esrgan enhancement model session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_tensor = _m_net->getSessionInput(_m_session, "input");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "output");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch real-esrgan enhancement model input src node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch real-esrgan enhancement model output node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();
    _m_successfully_initialized = true;

    LOG(INFO) << "Real-esrgan enhancement model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
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
StatusCode RealEsrGan<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = real_esrgan_impl::transform_input(in);

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
        _m_net->resizeTensor(_m_input_tensor, 1, 3, internal_in.input_image.size().height,
                             internal_in.input_image.size().width);
        _m_net->resizeSession(_m_session);
        _m_output_tensor = _m_net->getSessionOutput(_m_session, "output");
    }

    cv::Mat input_src = preprocess_image(internal_in.input_image);

    // run session
    MNN::Tensor input_tensor_user_src(_m_input_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    auto input_tensor_data = input_tensor_user_src.host<float>();
    auto input_tensor_size = input_tensor_user_src.size();
    ::memcpy(input_tensor_data, input_src.data, input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user_src);
    _m_net->runSession(_m_session);

    // decode output tensor
    real_esrgan_impl::internal_output internal_out;
    MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    _m_output_tensor->copyToHostTensor(&output_tensor_user);
    auto host_data = output_tensor_user.host<float>();
    auto element_size = output_tensor_user.elementSize();

    std::vector<uchar> output_img_data;
    output_img_data.resize(element_size);

    for (auto index = 0; index < element_size; ++index) {
        auto pix_val_f = host_data[index] * 255.0;

        if (pix_val_f < 0.0) {
            pix_val_f = 0.0;
        }

        if (pix_val_f >= 255) {
            pix_val_f = 255.0;
        }

        auto pix_val = static_cast<uchar>(pix_val_f);
        output_img_data[index] = pix_val;
    }

    auto output_w = output_tensor_user.width();
    auto output_h = output_tensor_user.height();
    cv::Mat result_image(cv::Size(output_w, output_h), CV_8UC3, output_img_data.data());
    cv::cvtColor(result_image, internal_out.enhancement_result, cv::COLOR_RGB2BGR);

    // transform internal output into external output
    out = real_esrgan_impl::transform_output<OUTPUT>(internal_out);
    return StatusCode::OK;
}


/***
*
* @param input_image
* @param output_src
* @param output_gray
*/
template<typename INPUT, typename OUTPUT>
cv::Mat RealEsrGan<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) {
    cv::Mat output_src;
    input_image.copyTo(output_src);

    if (output_src.channels() == 4) {
        cv::cvtColor(output_src, output_src, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(output_src, output_src, cv::COLOR_BGR2RGB);
    }

    // normalize
    if (output_src.type() != CV_32FC3) {
        output_src.convertTo(output_src, CV_32FC3);
    }

    output_src /= 255.0;

    return output_src;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
RealEsrGan<INPUT, OUTPUT>::RealEsrGan() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
RealEsrGan<INPUT, OUTPUT>::~RealEsrGan() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode RealEsrGan<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool RealEsrGan<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode RealEsrGan<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}