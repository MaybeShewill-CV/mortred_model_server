/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamVitEncoder.cpp
 * Date: 23-6-7
 ************************************************/

#include "sam_vit_encoder.h"

#include "glog/logging.h"
#include "MNN/Interpreter.hpp"
#include "onnxruntime/onnxruntime_cxx_api.h"

#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "common/time_stamp.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace segment_anything {

class SamVitEncoder::Impl {
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
     * @param cfg
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    jinq::common::StatusCode encode(const cv::Mat& input_image, std::vector<float>& image_embeddings);

    /***
     *
     * @return
     */
    std::vector<int> get_encoder_input_shape() const {
        return _m_input_shape;
    }

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

  private:
    // model file path
    std::string _m_model_path;

    // model compute thread nums
    uint16_t _m_thread_nums = 1;

    // model backend device
    std::string _m_model_device;

    // model input/output names
    std::string _m_input_name;
    std::string _m_output_name;

    // mnn model session
    std::unique_ptr<MNN::Interpreter> _m_net;
    MNN::Session* _m_session = nullptr;
    MNN::Tensor* _m_input_tensor = nullptr;
    MNN::Tensor* _m_output_tensor = nullptr;

    // onnx model envs
    Ort::Env _m_env;
    Ort::MemoryInfo _m_memo_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    int _m_device_id = 0;

    // model session options
    Ort::SessionOptions _m_onnx_sess_options;

    // model session
    std::unique_ptr<Ort::Session> _m_onnx_sess;

    // model input/output shape info
    std::vector<int> _m_input_shape;
    std::vector<int> _m_output_shape;

    // init flag
    bool _m_successfully_init_model = false;

    // use onnx or mnn
    enum model_type {
        MNN = 0,
        ONNX = 1,
    };
    model_type _m_backend_type = MNN;

  private:
    /***
     *
     * @param input_image
     * @return
     */
    cv::Mat preprocess_image(const cv::Mat& input_image);

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init_mnn_model(const decltype(toml::parse("")) &cfg);

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init_onnx_model(const decltype(toml::parse("")) &cfg);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    StatusCode mnn_encode(const cv::Mat &input_image, std::vector<float> &image_embeddings);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    StatusCode onnx_encode(const cv::Mat &input_image, std::vector<float> &image_embeddings);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode SamVitEncoder::Impl::init(const decltype(toml::parse("")) &cfg) {
    // init sam encoder configs
    auto sam_encoder_cfg = cfg.at("SAM_VIT_ENCODER");
    _m_model_path = sam_encoder_cfg["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "sam encoder model file path: " << _m_model_path << " not exists";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto model_file_name = FilePathUtil::get_file_name(_m_model_path);
    if (model_file_name.find(".mnn") != std::string::npos) {
        _m_backend_type = MNN;
    } else if (model_file_name.find(".onnx") != std::string::npos) {
        _m_backend_type = ONNX;
    }

    StatusCode init_status;
    if (_m_backend_type == MNN) {
        init_status = init_mnn_model(cfg);
    } else {
        init_status = init_onnx_model(cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_init_model = true;
        LOG(INFO) << "Successfully load sam vit encoder";
    } else {
        _m_successfully_init_model = false;
        LOG(INFO) << "Failed load sam vit encoder";
    }

    return init_status;
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
jinq::common::StatusCode SamVitEncoder::Impl::encode(
    const cv::Mat &input_image,
    std::vector<float> &image_embeddings) {
    StatusCode infer_status;
    if (_m_backend_type == MNN) {
        infer_status = mnn_encode(input_image, image_embeddings);
    } else {
        infer_status = onnx_encode(input_image, image_embeddings);
    }
    return infer_status;
}

/***
 *
 * @param input_image
 * @return
 */
cv::Mat SamVitEncoder::Impl::preprocess_image(const cv::Mat &input_image) {

    auto input_node_h = static_cast<int>(_m_input_shape[2]);
    auto input_node_w = static_cast<int>(_m_input_shape[3]);
    auto ori_img_width = static_cast<float>(input_image.size().width);
    auto ori_img_height = static_cast<float>(input_image.size().height);
    auto long_side = std::max(input_image.size().height, input_image.size().width);
    float scale = static_cast<float>(input_node_h) / static_cast<float>(long_side);
    cv::Size target_size = cv::Size(
        static_cast<int>(scale * ori_img_width), static_cast<int>(scale * ori_img_height));

    cv::Mat result;
    cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);
    cv::resize(result, result,target_size);
    result.convertTo(result, CV_32FC3);

    cv::subtract(result, cv::Scalar(123.675, 116.28, 103.53), result);
    cv::divide(result, cv::Scalar(58.395, 57.12, 57.375), result);

    // pad image
    auto pad_h = input_node_h - target_size.height;
    auto pad_w = input_node_w - target_size.width;
    cv::copyMakeBorder(result, result, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, 0.0);

    return result;
}

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamVitEncoder::Impl::init_mnn_model(const decltype(toml::parse("")) &cfg) {
    auto sam_encoder_cfg = cfg.at("SAM_VIT_ENCODER");
    _m_model_path = sam_encoder_cfg["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "sam encoder model file path: " << _m_model_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_path.c_str()));
    _m_thread_nums = sam_encoder_cfg["model_threads_num"].as_integer();
    _m_model_device = sam_encoder_cfg["compute_backend"].as_string();
    MNN::ScheduleConfig mnn_config;
    mnn_config.numThread = _m_thread_nums;
    mnn_config.type = MNN_FORWARD_CPU;
    if (std::strcmp(_m_model_device.c_str(), "cuda") == 0) {
        mnn_config.type = MNN_FORWARD_CUDA;
    }
    MNN::BackendConfig backend_cfg;
    backend_cfg.precision = MNN::BackendConfig::Precision_Normal;
    backend_cfg.power = MNN::BackendConfig::Power_Normal;
    mnn_config.backendConfig = &backend_cfg;

    _m_session = _m_net->createSession(mnn_config);

    _m_input_name = "input_image";
    _m_output_name = "image_embeddings";

    _m_input_tensor = _m_net->getSessionInput(_m_session, _m_input_name.c_str());
    _m_output_tensor = _m_net->getSessionOutput(_m_session, _m_output_name.c_str());

    _m_input_shape = _m_input_tensor->shape();
    _m_output_shape = _m_output_tensor->shape();
    if (_m_input_shape.size() != 4 || _m_output_shape.size() != 4) {
        LOG(ERROR) << "invalid encoder input/output node shape";
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OJBK;
}

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamVitEncoder::Impl::init_onnx_model(const decltype(toml::parse("")) &cfg) {
    // ort env and memo info
    _m_env = {ORT_LOGGING_LEVEL_WARNING, ""};

    // init sam encoder configs
    auto sam_encoder_cfg = cfg.at("SAM_VIT_ENCODER");
    _m_model_path = sam_encoder_cfg["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "sam encoder model file path: " << _m_model_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_model_device = sam_encoder_cfg["compute_backend"].as_string();
    if (std::strcmp(_m_model_device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_device_id = static_cast<int>(sam_encoder_cfg["gpu_device_id"].as_integer());
    }
    _m_thread_nums = sam_encoder_cfg["model_threads_num"].as_integer();
    _m_onnx_sess_options.SetIntraOpNumThreads(_m_thread_nums);
    _m_onnx_sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    _m_onnx_sess_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_device_id;
        _m_onnx_sess_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    _m_onnx_sess = std::make_unique<Ort::Session>(_m_env, _m_model_path.c_str(), _m_onnx_sess_options);
    auto input_shape = _m_onnx_sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    for (auto& v : input_shape) {
        _m_input_shape.push_back(static_cast<int>(v));
    }
    auto output_shape = _m_onnx_sess->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    for (auto& v : output_shape) {
        _m_output_shape.push_back(static_cast<int>(v));
    }

    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode SamVitEncoder::Impl::mnn_encode(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    // preprocess image
    auto preprocessed_image = preprocess_image(input_image);
    auto input_tensor_values = CvUtils::convert_to_chw_vec(preprocessed_image);
    if (input_tensor_values.empty()) {
        LOG(ERROR) << "empty input data for sam vit encoder";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // run encoder
    auto input_tensor_user = MNN::Tensor(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, input_tensor_values.data(), input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);

    _m_net->runSession(_m_session);

    MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    _m_output_tensor->copyToHostTensor(&output_tensor_user);

    auto embeds_size = std::accumulate(
        std::begin(_m_output_shape), std::end(_m_output_shape), 1, std::multiplies());
    image_embeddings.resize(embeds_size);
    auto img_embeds_val = output_tensor_user.host<float>();
    for (auto idx = 0; idx < embeds_size; ++idx) {
        image_embeddings[idx] = img_embeds_val[idx];
    }

    return StatusCode::OJBK;
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode SamVitEncoder::Impl::onnx_encode(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    // init encoder inputs
    std::vector<Ort::Value> encoder_input_tensor;

    // preprocess image
    auto preprocessed_image = preprocess_image(input_image);
    auto input_tensor_values = CvUtils::convert_to_chw_vec(preprocessed_image);
    if (input_tensor_values.empty()) {
        LOG(ERROR) << "empty input data for sam vit encoder";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    auto input_image_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, (float*)input_tensor_values.data(), input_tensor_values.size(),
        reinterpret_cast<const int64_t *>(_m_input_shape.data()), _m_input_shape.size());
    encoder_input_tensor.push_back(std::move(input_image_tensor));

    // run session
    std::vector<const char*> input_names = { _m_input_name.c_str() };
    std::vector<const char*> output_names = { _m_output_name.c_str() };
    auto output_tensors = _m_onnx_sess->Run(
        Ort::RunOptions{nullptr}, input_names.data(), encoder_input_tensor.data(),
        encoder_input_tensor.size(), output_names.data(), output_names.size());
    auto output_preds_value = output_tensors[0].GetTensorMutableData<float>();

    // copy image embeddings
    auto embeds_size = std::accumulate(
        std::begin(_m_output_shape), std::end(_m_output_shape), 1, std::multiplies());
    image_embeddings.resize(embeds_size);
    for (auto idx = 0; idx < embeds_size; ++idx) {
        image_embeddings[idx] = output_preds_value[idx];
    }

    return StatusCode::OJBK;
}

/***
 *
 */
SamVitEncoder::SamVitEncoder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
SamVitEncoder::~SamVitEncoder() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode SamVitEncoder::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
jinq::common::StatusCode SamVitEncoder::encode(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    return _m_pimpl->encode(input_image, image_embeddings);
}

/***
 *
 * @return
 */
std::vector<int> SamVitEncoder::get_encoder_input_shape() const {
   return _m_pimpl->get_encoder_input_shape();
}

/***
 *
 * @return
 */
bool SamVitEncoder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}