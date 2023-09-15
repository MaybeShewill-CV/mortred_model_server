/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamVitTrtEncoder.cpp
 * Date: 23-9-7
 ************************************************/

#include "sam_vit_trt_encoder.h"

#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "TensorRT-8.6.1.6/NvInferRuntime.h"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/trt_helper/trt_helper.h"

namespace jinq {
namespace models {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::Base64;
using jinq::common::CvUtils;

namespace segment_anything {

using trt_helper::EngineBinding;
using trt_helper::DeviceMemory;
using trt_helper::TrtHelper;
using trt_helper::TrtLogger;

/***************** Impl Function Sets ******************/

class SamVitTrtEncoder::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        auto status = cudaStreamDestroy(_m_cuda_stream);
        if (status != cudaSuccess) {
            LOG(ERROR) << "~Failed to free sam trt segment object. Destruct cuda stream "
                          "failed code str: " << cudaGetErrorString(status);
        }
        DLOG(INFO) << "~destruct sam trt segmentation object";
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
        std::vector<int> result;
        for (auto idx = 0; idx < _m_input_binding.dims().nbDims; idx++) {
            result.push_back(_m_input_binding.dims().d[idx]);
        }
        return result;
    }

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  public:
    // model file path
    std::string _m_model_file_path;

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

    // flag
    bool _m_successfully_initialized = false;

  public:
    /***
     *
     * @param input_file_path
     * @param file_content
     * @return
     */
    static bool read_model_file(const std::string& input_file_path, std::vector<unsigned char>& file_content);

    /***
     * image preprocess
     * @param input_image : 输入图像
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;
};

/***
*
* @param cfg_file_path
* @return
 */
StatusCode SamVitTrtEncoder::Impl::init(const decltype(toml::parse(""))& config) {
    // init sam vit trt config section
    if (!config.contains("SAM_VIT_TRT_ENCODER")) {
        LOG(ERROR) << "Config file does not contain SAM_VIT_TRT_ENCODER section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    toml::value cfg_content = config.at("SAM_VIT_TRT_ENCODER");

    // init trt runtime
    _m_trt_logger = std::make_unique<TrtLogger>();
    auto* trt_runtime = nvinfer1::createInferRuntime(*_m_trt_logger);
    if(trt_runtime == nullptr) {
        LOG(ERROR) << "Init TensorRT runtime failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_trt_runtime = std::unique_ptr<nvinfer1::IRuntime>(trt_runtime);

    // init trt engine
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "Sam trt segmentation model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    std::vector<unsigned char> model_file_content;
    if (!read_model_file(_m_model_file_path, model_file_content)) {
        LOG(ERROR) << "read model file: " << _m_model_file_path << " failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto model_content_length = sizeof(model_file_content[0]) * model_file_content.size();
    _m_trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        _m_trt_runtime->deserializeCudaEngine(model_file_content.data(), model_content_length));
    if (_m_trt_engine == nullptr) {
        LOG(ERROR) << "deserialize trt engine failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init trt execution context
    _m_trt_execution_context = std::unique_ptr<nvinfer1::IExecutionContext>(_m_trt_engine->createExecutionContext());
    if (_m_trt_execution_context == nullptr) {
        LOG(ERROR) << "create trt engine failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind input tensor
    std::string input_node_name = "images";
    auto successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, _m_input_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_input_binding.dims().nbDims != 4) {
        std::string input_shape_str;
        for (auto idx = 0; idx < _m_input_binding.dims().nbDims; ++idx) {
            input_shape_str += std::to_string(_m_input_binding.dims().d[idx]) + ",";
        }
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N, C, H, W]";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_input_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = _m_input_binding.dims().d[2];
    _m_input_size_host.width = _m_input_binding.dims().d[3];

    // bind output tensor
    std::string output_node_name = "image_embeddings";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, output_node_name, _m_output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_output_binding.dims().nbDims != 4) {
        std::string output_shape_str;
        for (auto idx = 0; idx < _m_output_binding.dims().nbDims; ++idx) {
            output_shape_str += std::to_string(_m_output_binding.dims().d[idx]) + ",";
        }
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C, H, W]";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_output_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic output tensors";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup device memory
    auto set_device_memo_status = TrtHelper::setup_device_memory(_m_trt_engine, _m_device_memory);
    if (set_device_memo_status != StatusCode::OK) {
        LOG(ERROR) << "setup device memory for model failed, status code: " << set_device_memo_status;
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init cuda stream
    if (cudaStreamCreate(&_m_cuda_stream) != cudaSuccess) {
        LOG(ERROR) << "ERROR: cuda stream creation failed." << std::endl;
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // allocate output host tensor memo
    _m_output_host_memory.resize(_m_output_binding.volume());

    _m_successfully_initialized = true;
    LOG(INFO) << "Sam trt segmentation model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param input_file_path
 * @param file_content
 * @return
 */
bool SamVitTrtEncoder::Impl::read_model_file(
    const std::string &input_file_path, std::vector<unsigned char>& file_content) {
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

/***
 *
 * @param input_image
 * @return
 */
cv::Mat SamVitTrtEncoder::Impl::preprocess_image(const cv::Mat& input_image) const {
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
* @param in
* @param out
* @return
 */
StatusCode SamVitTrtEncoder::Impl::encode(const cv::Mat &input_image, std::vector<float> &image_embeddings){
    // preprocess input data
    _m_input_size_user = input_image.size();

    auto t_start = std::chrono::high_resolution_clock::now();
    auto preprocessed_image = preprocess_image(input_image);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    DLOG(INFO) << "      ---- embedding preprocess cost time: " << t_cost << " ms";

    t_start = std::chrono::high_resolution_clock::now();
    auto input_chw_data = CvUtils::convert_to_chw_vec(preprocessed_image);
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    DLOG(INFO) << "      ---- embedding convert to chw cost time: " << t_cost << " ms";

    t_start = std::chrono::high_resolution_clock::now();
    auto* cuda_mem_input = (float*)_m_device_memory.at(_m_input_binding.index());
    int32_t input_mem_size = static_cast<int32_t >(preprocessed_image.channels() * preprocessed_image.size().area() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
        cuda_mem_input, (float*)input_chw_data.data(), input_mem_size, cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    DLOG(INFO) << "      ---- embedding memcpy mat data to gpu cost time: " << t_cost << " ms";

    // do inference
    t_start = std::chrono::high_resolution_clock::now();
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
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    DLOG(INFO) << "      ---- embedding inference cost time: " << t_cost << " ms";

    // fetch image embeddings
    image_embeddings.resize(0);
    for (auto& val : _m_output_host_memory) {
        image_embeddings.push_back(val);
    }

    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
SamVitTrtEncoder::SamVitTrtEncoder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
SamVitTrtEncoder::~SamVitTrtEncoder() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
StatusCode SamVitTrtEncoder::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
bool SamVitTrtEncoder::is_successfully_initialized() const {
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
StatusCode SamVitTrtEncoder::encode(const cv::Mat &input_image, std::vector<float> &image_embeddings){
    return _m_pimpl->encode(input_image, image_embeddings);
}

/***
 *
 * @return
 */
std::vector<int> SamVitTrtEncoder::get_encoder_input_shape() const {
    return _m_pimpl->get_encoder_input_shape();
}

}
}
}
