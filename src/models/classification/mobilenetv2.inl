/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: MobileNetv2.cpp
* Date: 22-6-13
************************************************/

#include "mobilenetv2.h"

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

namespace classification {

using morted::models::io_define::classification::common_out;

namespace mobilenetv2_impl {

struct internal_input {
    cv::Mat input_image;
};

struct internal_output {
    int class_id;
    std::vector<float> scores;
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
    std::vector <uchar> image_vec_data(image_decode_string.begin(), image_decode_string.end());

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
transform_output(const mobilenetv2_impl::internal_output& internal_out) {
    common_out result;
    result.class_id = internal_out.class_id;
    result.scores = internal_out.scores;
    return result;
}


}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class MobileNetv2<INPUT, OUTPUT>::Impl {
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
    StatusCode init(const decltype(toml::parse(

                                       ""))& config);

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
        auto internal_in = mobilenetv2_impl::transform_input(in);
        if (!internal_in.input_image.data || internal_in.input_image.empty()) {
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }

        // preprocess image
        auto preprocessed_image = preprocess_image(internal_in.input_image);
        // run session
        MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
        auto input_tensor_data = input_tensor_user.host<float>();
        auto input_tensor_size = input_tensor_user.size();
        ::memcpy(input_tensor_data, preprocessed_image.data, input_tensor_size);
        _m_input_tensor->copyFromHostTensor(&input_tensor_user);
        _m_net->runSession(_m_session);
        // decode output tensor
        MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
        _m_output_tensor->copyToHostTensor(&output_tensor_user);
        // transform output
        out.clear();
        mobilenetv2_impl::internal_output internal_out;
        for (auto index = 0; index < output_tensor_user.elementSize(); ++index) {
            internal_out.scores.push_back(output_tensor_user.host<float>()[index]);
        }
        auto max_score = std::max_element(
                output_tensor_user.host<float>(),
                        output_tensor_user.host<float>() + output_tensor_user.elementSize());
        auto cls_id = static_cast<int>(std::distance(output_tensor_user.host<float>(), max_score));
        internal_out.class_id = cls_id;
        out.push_back(mobilenetv2_impl::transform_output<OUTPUT>(internal_out));

        return StatusCode::OK;
    }

private:
    std::string _m_model_file_path;
    // MNN Net即模型数据持有者
    std::unique_ptr <MNN::Interpreter> _m_net = nullptr;
    // MNN session即模型输入数据持有者
    MNN::Session* _m_session = nullptr;
    // MNN session配置
    MNN::ScheduleConfig _m_session_config;
    // MNN 输入tensor
    MNN::Tensor* _m_input_tensor = nullptr;
    // MNN score输出tensor
    MNN::Tensor* _m_output_tensor = nullptr;
    // MNN后端使用线程数
    int _m_threads_nums = 4;
    // MNN 模型输入tensor大小
    cv::Size _m_input_tensor_size = cv::Size(224, 224);
    // 模型是否成功初始化标志位
    bool _m_successfully_initialized = false;
    // 图像均值
    const cv::Scalar _m_mean_value = cv::Scalar(103.94f, 116.78f, 123.68f);

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
StatusCode MobileNetv2<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("MOBILENETV2")) {
        LOG(ERROR) << "Config文件没有MOBILENETV2相关配置, 请重新检查配置文件";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    toml::value cfg_content = config.at("MOBILENETV2");

    // init Interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "MobileNetv2 classification model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));

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

    // 根据Interpreter创建Session
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
    if (_m_session == nullptr) {
        LOG(ERROR) << "Create Session failed, model file path: " << _m_model_file_path;
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // 创建Tensor
    _m_input_tensor = _m_net->getSessionInput(_m_session, "input_tensor");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "output_tensor");
    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch mobilenetv2 classification model input node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch mobilenetv2 classification model output node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // 初始化input tensor size
    _m_input_tensor_size = cv::Size(_m_input_tensor->width(), _m_input_tensor->height());

    // 初始化类是否成功初始化标志位
    _m_successfully_initialized = true;
    LOG(INFO) << "MobileNetv2 classification model initialization complete !!!";
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
cv::Mat MobileNetv2<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
    // resize input image
    cv::Mat tmp;

    if (input_image.size() != _m_input_tensor_size) {
        cv::resize(input_image, tmp, _m_input_tensor_size);
    } else {
        tmp = input_image;
    }

    // normalize image
    tmp.convertTo(tmp, CV_32FC3);
    cv::subtract(tmp, _m_mean_value, tmp);

    return tmp;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
MobileNetv2<INPUT, OUTPUT>::MobileNetv2() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
MobileNetv2<INPUT, OUTPUT>::~MobileNetv2() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode MobileNetv2<INPUT, OUTPUT>::init(const decltype(toml::parse(

            ""))& cfg) {
    return _m_pimpl->
           init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool MobileNetv2<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode MobileNetv2<INPUT, OUTPUT>::run(const INPUT& input, std::vector <OUTPUT>& output) {
    return _m_pimpl->run(input, output);
}

}
}
}