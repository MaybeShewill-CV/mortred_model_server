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
#include "TensorRT-8.6.1.6/NvInferRuntime.h"
#include "toml/value.hpp"

#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "models/trt_helper/trt_helper.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace segment_anything {

using trt_helper::EngineBinding;
using trt_helper::DeviceMemory;
using trt_helper::TrtHelper;
using trt_helper::TrtLogger;

class SamVitEncoder::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        if (_m_backend_type == TRT) {
            auto status = cudaStreamDestroy(_m_cuda_stream);
            if (status != cudaSuccess) {
                LOG(ERROR) << "~Failed to free sam trt segment object. Destruct cuda stream "
                              "failed code str: " << cudaGetErrorString(status);
            }
        }
    }

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    StatusCode encode(const cv::Mat& input_image, std::vector<float>& image_embeddings);

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

    // onnx model session options
    Ort::SessionOptions _m_onnx_sess_options;

    // onnx model session
    std::unique_ptr<Ort::Session> _m_onnx_sess;

    // model input/output shape info
    std::vector<int> _m_input_shape;
    std::vector<int> _m_output_shape;

    // tensorrt engine
    std::unique_ptr<nvinfer1::IRuntime> _m_trt_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> _m_trt_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> _m_trt_execution_context;
    std::unique_ptr<TrtLogger> _m_trt_logger;

    // input/output tensor binding
    EngineBinding _m_input_binding;
    EngineBinding _m_output_binding;

    // trt device memory
    DeviceMemory _m_device_memory;
    cudaStream_t _m_cuda_stream = nullptr;

    // output data host
    std::vector<float> _m_output_host_memory;

    // input image size
    cv::Size _m_input_size_user = cv::Size();
    //　input tensor size
    cv::Size _m_input_size_host = cv::Size();

    // init flag
    bool _m_successfully_init_model = false;

    // use onnx mnn or trt
    enum model_type {
        TRT = 0,
        ONNX = 1,
        MNN = 2,
    };
    model_type _m_backend_type = TRT;

  private:
    /***
     *
     * @param input_image
     * @return
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init_mnn_model(const toml::value& cfg);

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init_onnx_model(const toml::value& cfg);

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init_trt_model(const toml::value& cfg);

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

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    StatusCode trt_encode(const cv::Mat &input_image, std::vector<float> &image_embeddings);

    /***
     *
     * @param input_file_path
     * @param file_content
     * @return
     */
    static bool read_model_file(const std::string& input_file_path, std::vector<unsigned char>& file_content) {
        // read file
        std::ifstream file(input_file_path, std::ios::binary);
        if (!file.is_open() || file.eof() || file.fail() || file.bad()) {
            LOG(ERROR) << "open input file: " << input_file_path << " failed, error: " << strerror(errno);
            return false;
        }
        file.unsetf(std::ios::skipws);
        std::streampos file_size;
        file.seekg(0, std::ios::end);
        file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        file_content.resize(file_size);
        file.read(reinterpret_cast<std::ifstream::char_type*>(&file_content.front()), file_size);
        file.close();
        return true;
    }
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamVitEncoder::Impl::init(const decltype(toml::parse("")) &cfg) {
    // choose backend type
    auto backend_dict = cfg.at("BACKEND_DICT");
    auto backend_name = cfg.at("SAM_ENCODER").at("backend_type").as_string();
    _m_backend_type = static_cast<model_type>(backend_dict[backend_name].as_integer());
    
    // init sam encoder configs
    toml::value sam_encoder_cfg;
    if (_m_backend_type == MNN) {
        sam_encoder_cfg = cfg.at("SAM_MNN_ENCODER");
    } else if (_m_backend_type == ONNX) {
        sam_encoder_cfg = cfg.at("SAM_ONNX_ENCODER");
    } else {
        sam_encoder_cfg = cfg.at("SAM_TRT_ENCODER");
    }
    auto model_file_name = FilePathUtil::get_file_name(sam_encoder_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == MNN) {
        init_status = init_mnn_model(sam_encoder_cfg);
    } else if (_m_backend_type == ONNX) {
        init_status = init_onnx_model(sam_encoder_cfg);
    } else {
        init_status = init_trt_model(sam_encoder_cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_init_model = true;
        LOG(INFO) << "Successfully load sam vit encoder from: " << model_file_name;
    } else {
        _m_successfully_init_model = false;
        LOG(INFO) << "Failed load sam vit encoder from: " << model_file_name;
    }

    return init_status;
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode SamVitEncoder::Impl::encode(
    const cv::Mat &input_image,
    std::vector<float> &image_embeddings) {
    StatusCode infer_status;
    if (_m_backend_type == MNN) {
        infer_status = mnn_encode(input_image, image_embeddings);
    } else if (_m_backend_type == ONNX) {
        infer_status = onnx_encode(input_image, image_embeddings);
    } else {
        infer_status = trt_encode(input_image, image_embeddings);
    }
    return infer_status;
}

/***
 *
 * @param input_image
 * @return
 */
cv::Mat SamVitEncoder::Impl::preprocess_image(const cv::Mat &input_image) const {
    // long-side resize image
    auto input_node_h = static_cast<int>(_m_input_size_host.height);
    auto input_node_w = static_cast<int>(_m_input_size_host.width);
    auto ori_img_width = static_cast<float>(input_image.size().width);
    auto ori_img_height = static_cast<float>(input_image.size().height);
    auto long_side = std::max(input_image.size().height, input_image.size().width);
    float scale = static_cast<float>(input_node_h) / static_cast<float>(long_side);
    auto target_width = static_cast<int>(std::round(scale * ori_img_width));
    auto target_height = static_cast<int>(std::round(scale * ori_img_height));
    cv::Size target_size = cv::Size(target_width, target_height);

    cv::Mat result;
    cv::resize(input_image, result,target_size);
    result.convertTo(result, CV_32FC3);

    // mean abstraction
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
StatusCode SamVitEncoder::Impl::init_mnn_model(const toml::value& cfg) {
    _m_model_path = cfg.at("model_file_path").as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "sam encoder model file path: " << _m_model_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_path.c_str()));
    _m_thread_nums = cfg.at("model_threads_num").as_integer();
    _m_model_device = cfg.at("compute_backend").as_string();
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
    _m_input_size_host.height = _m_input_shape[2];
    _m_input_size_host.width = _m_input_shape[3];
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
StatusCode SamVitEncoder::Impl::init_onnx_model(const toml::value& cfg) {
    // ort env and memo info
    _m_env = {ORT_LOGGING_LEVEL_ERROR, ""};

    // init sam encoder configs
    _m_model_path = cfg.at("model_file_path").as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "sam encoder model file path: " << _m_model_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_model_device = cfg.at("compute_backend").as_string();
    if (std::strcmp(_m_model_device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_device_id = static_cast<int>(cfg.at("gpu_device_id").as_integer());
    }
    _m_thread_nums = cfg.at("model_threads_num").as_integer();
    _m_onnx_sess_options.SetIntraOpNumThreads(_m_thread_nums);
    _m_onnx_sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    _m_onnx_sess_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_device_id;
        _m_onnx_sess_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    _m_input_name = "input_image";
    _m_output_name = "image_embeddings";

    _m_onnx_sess = std::make_unique<Ort::Session>(_m_env, _m_model_path.c_str(), _m_onnx_sess_options);
    auto input_shape = _m_onnx_sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    for (auto& v : input_shape) {
        _m_input_shape.push_back(static_cast<int>(v));
    }
    _m_input_size_host.height = _m_input_shape[2];
    _m_input_size_host.width = _m_input_shape[3];
    auto output_shape = _m_onnx_sess->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    for (auto& v : output_shape) {
        _m_output_shape.push_back(static_cast<int>(v));
    }

    return StatusCode::OK;
}

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamVitEncoder::Impl::init_trt_model(const toml::value& cfg) {
    // init trt runtime
    _m_trt_logger = std::make_unique<TrtLogger>();
    auto* trt_runtime = nvinfer1::createInferRuntime(*_m_trt_logger);
    if(trt_runtime == nullptr) {
        LOG(ERROR) << "Init TensorRT runtime failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_trt_runtime = std::unique_ptr<nvinfer1::IRuntime>(trt_runtime);

    // init trt engine
    if (!cfg.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_path = cfg.at("model_file_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "Sam trt segmentation model file: " << _m_model_path << " not exist";
        return StatusCode::MODEL_INIT_FAILED;
    }
    std::vector<unsigned char> model_file_content;
    if (!read_model_file(_m_model_path, model_file_content)) {
        LOG(ERROR) << "read model file: " << _m_model_path << " failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto model_content_length = sizeof(model_file_content[0]) * model_file_content.size();
    _m_trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        _m_trt_runtime->deserializeCudaEngine(model_file_content.data(), model_content_length));
    if (_m_trt_engine == nullptr) {
        LOG(ERROR) << "deserialize trt engine failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init trt execution context
    _m_trt_execution_context = std::unique_ptr<nvinfer1::IExecutionContext>(_m_trt_engine->createExecutionContext());
    if (_m_trt_execution_context == nullptr) {
        LOG(ERROR) << "create trt engine failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind input tensor
    std::string input_node_name = "images";
    auto successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, _m_input_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_input_binding.dims().nbDims != 4) {
        std::string input_shape_str = TrtHelper::dims_to_string(_m_input_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_input_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = _m_input_binding.dims().d[2];
    _m_input_size_host.width = _m_input_binding.dims().d[3];
    for (auto idx = 0; idx < _m_input_binding.dims().nbDims; idx++) {
        _m_input_shape.push_back(_m_input_binding.dims().d[idx]);
    }

    // bind output tensor
    std::string output_node_name = "image_embeddings";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, output_node_name, _m_output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_output_binding.dims().nbDims != 4) {
        std::string output_shape_str = TrtHelper::dims_to_string(_m_output_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_output_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic output tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }
    for (auto idx = 0; idx < _m_output_binding.dims().nbDims; idx++) {
        _m_output_shape.push_back(_m_output_binding.dims().d[idx]);
    }

    // setup device memory
    auto set_device_memo_status = TrtHelper::setup_device_memory(_m_trt_engine, _m_device_memory);
    if (set_device_memo_status != StatusCode::OK) {
        LOG(ERROR) << "setup device memory for model failed, status code: " << set_device_memo_status;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init cuda stream
    if (cudaStreamCreate(&_m_cuda_stream) != cudaSuccess) {
        LOG(ERROR) << "ERROR: cuda stream creation failed." << std::endl;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // allocate output host tensor memo
    _m_output_host_memory.resize(_m_output_binding.volume());

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
    std::vector<int64 > input_shape;
    for (auto& v : _m_input_shape) {
        input_shape.push_back(static_cast<long>(v));
    }
    auto input_image_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, (float*)input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());
    encoder_input_tensor.push_back(std::move(input_image_tensor));

    // run session
    std::vector<const char*> input_names = { _m_input_name.c_str() };
    std::vector<const char*> output_names = { _m_output_name.c_str() };
    auto output_tensors = _m_onnx_sess->Run(
        Ort::RunOptions{nullptr}, input_names.data(), encoder_input_tensor.data(),
        encoder_input_tensor.size(), output_names.data(), output_names.size());
    auto output_preds_value = output_tensors[0].GetTensorMutableData<float>();
    auto output_mask_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

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
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode SamVitEncoder::Impl::trt_encode(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    // preprocess input data
    auto preprocessed_image = preprocess_image(input_image);
    auto input_chw_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    auto* cuda_mem_input = (float*)_m_device_memory.at(_m_input_binding.index());
    int32_t input_mem_size = static_cast<int32_t >(preprocessed_image.channels() * preprocessed_image.size().area() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
        cuda_mem_input, (float*)input_chw_data.data(), input_mem_size, cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    _m_trt_execution_context->setTensorAddress("images", cuda_mem_input);
    _m_trt_execution_context->setTensorAddress("image_embeddings", _m_device_memory.at(_m_output_binding.index()));
    if (!_m_trt_execution_context->enqueueV3(_m_cuda_stream)) {
        LOG(ERROR) << "excute input data for inference failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cuda_status = cudaMemcpyAsync(_m_output_host_memory.data(),
                                  _m_device_memory.at(_m_output_binding.index()),
                                  (int)(_m_output_binding.volume() * sizeof(float)),
                                  cudaMemcpyDeviceToHost, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cudaStreamSynchronize(_m_cuda_stream);

    // fetch image embeddings
    image_embeddings.resize(0);
    for (auto& val : _m_output_host_memory) {
        image_embeddings.push_back(val);
    }

    return StatusCode::OK;
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
StatusCode SamVitEncoder::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode SamVitEncoder::encode(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
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