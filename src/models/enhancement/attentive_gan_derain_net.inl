/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: attentive_gan_derain_net.inl
* Date: 22-6-14
************************************************/

#include "attentive_gan_derain_net.h"

#include "MNN/Interpreter.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "common/base64.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::Base64;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::mat_input;

namespace enhancement {

using jinq::models::io_define::enhancement::std_enhancement_output;

namespace attentiveganderain_impl {

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
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<file_input>::type>::value, internal_input>::type transform_input(const INPUT &in) {
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
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<mat_input>::type>::value, internal_input>::type transform_input(const INPUT &in) {
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
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<base64_input>::type>::value, internal_input>::type transform_input(const INPUT &in) {
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
template <typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_enhancement_output>::type>::value, std_enhancement_output>::type
transform_output(const attentiveganderain_impl::internal_output &internal_out) {
    std_enhancement_output result;
    internal_out.enhancement_result.copyTo(result.enhancement_result);
    return result;
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
    StatusCode init(const decltype(toml::parse("")) &config);

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
    // 模型文件存储路径
    std::string _m_model_file_path;
    // MNN Interpreter
    std::unique_ptr<MNN::Interpreter> _m_net = nullptr;
    // MNN Session
    MNN::Session *_m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor *_m_input_tensor = nullptr;
    // MNN Loc Output tensor node
    MNN::Tensor *_m_output_tensor = nullptr;
    // MNN后端使用线程数
    int _m_threads_nums = 4;
    // 用户输入网络的图像尺寸
    cv::Size _m_input_size_user = cv::Size();
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
StatusCode AttentiveGanDerain<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    if (!config.contains("ATTENTIVEGANDERAIN")) {
        LOG(ERROR) << "Config missing ATTENTIVEGANDERAIN section please check config file";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("ATTENTIVEGANDERAIN");

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
        LOG(ERROR) << "AttentiveGanDerain model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));

    if (nullptr == _m_net) {
        LOG(ERROR) << "Create attentive gan derain model interpreter failed";
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
        LOG(ERROR) << "Create attentive gan derain model session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_tensor = _m_net->getSessionInput(_m_session, "input_tensor");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "final_output");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch attentive gan derain model input src node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch enlighten-gan enhancement model output node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

    if (!cfg_content.contains("model_input_image_size")) {
        _m_input_size_user.width = 320;
        _m_input_size_user.height = 240;
    } else {
        _m_input_size_user.width = static_cast<int>(cfg_content.at("model_input_image_size").as_array()[1].as_integer());
        _m_input_size_user.height = static_cast<int>(cfg_content.at("model_input_image_size").as_array()[0].as_integer());
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "Attentive gan derain model: " << FilePathUtil::get_file_name(_m_model_file_path) << " initialization complete!!!";
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
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, preprocessed_image.data, input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);
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
    MNN::Tensor output_tensor_user(_m_output_tensor, _m_output_tensor->getDimensionType());
    _m_output_tensor->copyToHostTensor(&output_tensor_user);
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
template <typename INPUT, typename OUTPUT> StatusCode AttentiveGanDerain<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
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