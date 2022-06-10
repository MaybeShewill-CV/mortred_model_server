/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: BiseNetV2.cpp
* Date: 22-6-9
************************************************/

#include "bisenetv2.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/file_path_util.h"
#include "common/base64.h"

namespace morted {
namespace models {

using morted::common::FilePathUtil;
using morted::common::StatusCode;
using morted::common::Base64;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::common_io::file_input;
using morted::models::io_define::common_io::base64_input;

namespace scene_segmentation {
using morted::models::io_define::scene_segmentation::common_out;

namespace bisenetv2_impl {

struct internal_input {
    cv::Mat input_image;
};

struct internal_output {
    cv::Mat segmentation_result;
};

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
        LOG(WARNING) << "input image: " << in.input_image_path << " not exist";
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
    auto image_decode_string = morted::common::Base64::base64_decode(in.input_image_content);
    std::vector<uchar> image_vec_data(image_decode_string.begin(), image_decode_string.end());

    if (image_vec_data.empty()) {
        LOG(WARNING) << "image data empty";
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<common_out>::type>::value, common_out>::type
transform_output(const bisenetv2_impl::internal_output& internal_out) {
    common_out result;
    internal_out.segmentation_result.copyTo(result.segmentation_result);
    return result;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class BiseNetV2<INPUT, OUTPUT>::Impl {
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
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT& in, std::vector<OUTPUT>& out) {
        // transform external input into internal input
        auto internal_in = bisenetv2_impl::transform_input(in);

        if (!internal_in.input_image.data || internal_in.input_image.empty()) {
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }

        // preprocess image
        cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
        // run session
        MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
        auto input_tensor_data = input_tensor_user.host<float>();
        auto input_tensor_size = input_tensor_user.size();
        ::memcpy(input_tensor_data, preprocessed_image.data, input_tensor_size);
        _m_input_tensor->copyFromHostTensor(&input_tensor_user);
        _m_net->runSession(_m_session);
        // fetch net output
        auto* output_tensor_user = new MNN::Tensor(_m_output_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
        _m_output_tensor->copyToHostTensor(output_tensor_user);
        auto host_data = output_tensor_user->host<float>();

        int output_tensor_height = output_tensor_user->shape()[0];
        int output_tensor_width = output_tensor_user->shape()[1];
        const int output_tensor_channels = output_tensor_user->shape()[2];
        cv::Mat softmax_score_mat(
            cv::Size(output_tensor_width, output_tensor_height),
            CV_32FC(output_tensor_channels),
            host_data);
        cv::Mat result_image(softmax_score_mat.size(), CV_32SC1);

        for (auto row = 0; row < softmax_score_mat.rows; ++row) {
            for (auto col = 0; col < softmax_score_mat.cols; ++col) {
                auto* scores = softmax_score_mat.ptr<float>(row, col);
                auto max_score_iter = std::max_element(scores, scores + output_tensor_channels);
                int cls_id = static_cast<int>(std::distance(scores, max_score_iter));
                result_image.at<int>(row, col) = cls_id;
            }
        }

        cv::resize(result_image, result_image, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);

        // transform internal output into external output
        bisenetv2_impl::internal_output internal_out;
        internal_out.segmentation_result = result_image;
        out.push_back(bisenetv2_impl::transform_output<OUTPUT>(internal_out));

        return StatusCode::OK;
    }

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
    uint _m_threads_nums = 4;
    // 用户输入网络的图像尺寸
    cv::Size _m_input_size_user = cv::Size();
    //　计算图定义的输入node尺寸
    cv::Size _m_input_size_host = cv::Size();
    // 是否成功初始化标志位
    bool _m_successfully_initialized = false;

private:
    /***
     * 图像预处理, 转换图像为CV_32FC3, 通过dst = src / 127.5 - 1.0来归一化图像到[-1.0, 1.0]
     * @param input_image : 输入图像
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;
};

/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode BiseNetV2<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("BISENETV2")) {
        LOG(ERROR) << "Config文件没有BISENETV2相关配置, 请重新检查配置文件";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("BISENETV2");

    // 初始化线程模型计算使用的线程数
    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config file doesn\'t contain model_threads_num field, using default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = cfg_content.at("model_threads_num").as_integer();
    }

    // 初始化MNN Interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config file doesn\'t contain model_file_path field, please check again";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "BiseNetv2 model file path: " << _m_model_file_path << ", not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));

    if (nullptr == _m_net) {
        LOG(ERROR) << "Create BiseNetv2 Interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // 初始化MNN Session
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
    backend_config.precision = MNN::BackendConfig::Precision_High;
    backend_config.power = MNN::BackendConfig::Power_High;
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);

    if (nullptr == _m_session) {
        LOG(ERROR) << "Create BiseNetv2 Session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // 初始化graph input node和output node
    _m_input_tensor = _m_net->getSessionInput(_m_session, "input_tensor");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "final_output");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch BiseNetv2 Input Node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch BiseNetv2 Output Node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

    // 初始化用户输入的图像归一化尺寸
    if (!cfg_content.contains("model_input_image_size")) {
        LOG(WARNING) << "Config doesn\'t contain model_input_image_size filed, using default value [1024, 512]";
        _m_input_size_user.width = 1024;
        _m_input_size_user.height = 512;
    } else {
        _m_input_size_user.width = static_cast<int>(
                                       cfg_content.at("model_input_image_size").as_array()[1].as_integer());
        _m_input_size_user.height = static_cast<int>(
                                        cfg_content.at("model_input_image_size").as_array()[0].as_integer());
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "BiseNetv2 detection model: " << FilePathUtil::get_file_name(_m_model_file_path)
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
cv::Mat BiseNetV2<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
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

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
BiseNetV2<INPUT, OUTPUT>::BiseNetV2() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
BiseNetV2<INPUT, OUTPUT>::~BiseNetV2() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode BiseNetV2<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool BiseNetV2<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode BiseNetV2<INPUT, OUTPUT>::run(const INPUT& input, std::vector<OUTPUT>& output) {
    return _m_pimpl->run(input, output);
}

}
}
}