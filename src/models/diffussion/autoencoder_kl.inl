/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: autoencoder_kl.cpp
 * Date: 24-5-23
 ************************************************/

#include "autoencoder_kl.h"

#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "TensorRT-8.6.1.6/NvInferRuntime.h"
#include "TensorRT-8.6.1.6/NvInferPlugin.h"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/trt_helper/trt_helper.h"

namespace jinq {
namespace models {

using jinq::common::Base64;
using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::trt_helper::EngineBinding;
using jinq::models::trt_helper::DeviceMemory;
using jinq::models::trt_helper::TrtHelper;
using jinq::models::trt_helper::TrtLogger;

namespace diffusion {

using jinq::models::io_define::diffusion::std_vae_decode_input;
using jinq::models::io_define::diffusion::std_vae_decode_output;

namespace autoencoder_kl_impl {

using internal_input = std_vae_decode_input;
using internal_output = std_vae_decode_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<std_vae_decode_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    return in;
}

/***
* transform different type of internal output into external output
* @tparam EXTERNAL_OUTPUT
* @tparam dummy
* @param in
* @return
 */
template <typename OUTPUT>
typename std::enable_if<
    std::is_same<OUTPUT, std::decay<std_vae_decode_output >::type>::value, std_vae_decode_output>::type
transform_output(const autoencoder_kl_impl::internal_output& internal_out) {
    return internal_out;
}

} // namespace autoencoder_kl_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class AutoEncoderKL<INPUT, OUTPUT>::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        if (_m_backend_type == ONNX) {}

        if (_m_backend_type == TRT) {
            cudaFreeHost(_m_trt_params.output_host);
            cudaStreamDestroy(_m_trt_params.cuda_stream);
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
    StatusCode init(const decltype(toml::parse("")) &config);

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
    struct TRTParams {
        // model file path
        std::string model_file_path;
        // trt context
        TrtLogger logger;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        cudaStream_t cuda_stream = nullptr;
        // trt bindings
        EngineBinding input_binding;
        EngineBinding output_binding;
        // trt memory
        void* input_device = nullptr;
        void* output_device = nullptr;
        float* output_host = nullptr;
    };

    struct ONNXParams {
        std::string model_file_path;
        // ort env params
        int thread_nums = 1;
        std::string device = "cuda";
        int device_id = 0;
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::Session* session = nullptr;
        Ort::AllocatorWithDefaultOptions allocator;
        // input/output node info
        std::vector<const char*> input_node_names;
        std::vector<std::vector<int64_t>> input_node_shapes;
        std::vector<const char*> output_node_names;
        std::vector<std::vector<int64_t>> output_node_shapes;
    };

    enum BackendType {
        TRT = 0,
        ONNX = 1,
    };

  private:
    // model backend type
    BackendType _m_backend_type = TRT;

    // trt net params
    TRTParams _m_trt_params;

    // onnx net params
    ONNXParams _m_onnx_params;

    //　input node size
    cv::Size _m_input_size = cv::Size();
    int _m_input_channels = 0;

    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @param config
     * @return
     */
    StatusCode init_trt(const toml::value& cfg);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode trt_run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @return
     */
    autoencoder_kl_impl::internal_output trt_decode_output();

    /***
     *
     * @param config
     * @return
     */
    StatusCode init_onnx(const toml::value& cfg);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode onnx_run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @param predict_output
     * @return
     */
    autoencoder_kl_impl::internal_output onnx_decode_output(Ort::Value& predict_output);
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode AutoEncoderKL<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("AUTOENCODER_KL").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init autoencoder-kl configs
    toml::value model_cfg;
    if (_m_backend_type == TRT) {
        model_cfg = config.at("AUTOENCODER_KL_TRT");
    } else if (_m_backend_type == ONNX) {
        model_cfg = config.at("AUTOENCODER_KL_ONNX");
    } else {
        LOG(ERROR) << "not supported backend type: " << _m_backend_type;
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto model_file_name = FilePathUtil::get_file_name(model_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == TRT) {
        init_status = init_trt(model_cfg);
    } else if (_m_backend_type == ONNX){
        init_status = init_onnx(model_cfg);
    } else {
        LOG(ERROR) << "not supported backend type: " << _m_backend_type;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << "Successfully load autoencoder-kl model from: " << model_file_name;
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load autoencoder-kl model from: " << model_file_name;
    }

    return init_status;
}

/***
*
* @param in
* @param out
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode AutoEncoderKL<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    StatusCode infer_status;
    if (_m_backend_type == TRT) {
        infer_status = trt_run(in, out);
    } else if (_m_backend_type == ONNX) {
        infer_status = onnx_run(in, out);
    } else {
        LOG(ERROR) << "not supported backend type: " << _m_backend_type;
        infer_status = StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    return infer_status;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode AutoEncoderKL<INPUT, OUTPUT>::Impl::init_trt(const toml::value& cfg) {
    // init trt runtime
    _m_trt_params.logger = TrtLogger();
    _m_trt_params.runtime = nvinfer1::createInferRuntime(_m_trt_params.logger);
    if(nullptr == _m_trt_params.runtime) {
        LOG(ERROR) << "init tensorrt runtime failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init trt engine
    if (!cfg.contains("model_file_path")) {
        LOG(ERROR) << "config doesn\'t have model_file_path field";
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_trt_params.model_file_path = cfg.at("model_file_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_trt_params.model_file_path)) {
        LOG(ERROR) << "AutoEncoderKL trt model file: " << _m_trt_params.model_file_path << " not exist";
        return StatusCode::MODEL_INIT_FAILED;
    }
    std::ifstream fgie(_m_trt_params.model_file_path, std::ios_base::in | std::ios_base::binary);
    if (!fgie) {
        LOG(ERROR) << "read model file: " << _m_trt_params.model_file_path << " failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    std::stringstream buffer;
    buffer << fgie.rdbuf();
    std::string stream_model(buffer.str());
    if (!initLibNvInferPlugins(nullptr, "")) {
        LOG(ERROR) << "init nvinfer plugin failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_trt_params.engine = _m_trt_params.runtime->deserializeCudaEngine(stream_model.data(), stream_model.size());
    if (nullptr == _m_trt_params.engine) {
        LOG(ERROR) << "deserialize trt engine failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init trt execution context
    _m_trt_params.context = _m_trt_params.engine->createExecutionContext();
    if (nullptr == _m_trt_params.context) {
        LOG(ERROR) << "create trt engine failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind input tensor
    std::string input_node_name = "input";
    auto successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, input_node_name, _m_trt_params.input_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.input_binding.dims().nbDims != 4) {
        std::string input_shape_str = TrtHelper::dims_to_string(_m_trt_params.input_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.input_binding.is_dynamic()) {
        LOG(ERROR) << "not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_channels = _m_trt_params.input_binding.dims().d[1];
    _m_input_size.height = _m_trt_params.input_binding.dims().d[2];
    _m_input_size.width = _m_trt_params.input_binding.dims().d[3];

    // bind output tensor
    std::string output_node_name = "decode_image";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind decode image output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.output_binding.dims().nbDims != 4) {
        std::string output_shape_str = TrtHelper::dims_to_string(_m_trt_params.output_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.output_binding.is_dynamic()) {
        LOG(ERROR) << "not support dynamic output tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup input host/device memory
    auto memo_size = _m_trt_params.input_binding.volume() * sizeof(float);
    auto cuda_status = cudaMalloc(&_m_trt_params.input_device, memo_size);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "allocate device memory for input image failed, err str: " << cudaGetErrorString(cuda_status);
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup output host/device memory
    memo_size = _m_trt_params.output_binding.volume() * sizeof(float);
    cuda_status = cudaMallocHost(reinterpret_cast<void**>(&_m_trt_params.output_host), memo_size);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "allocate host memory for output node failed, err str: " << cudaGetErrorString(cuda_status);
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    cuda_status = cudaMalloc(&_m_trt_params.output_device, memo_size);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "allocate device memory for output node failed, err str: " << cudaGetErrorString(cuda_status);
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init cuda stream
    if (cudaStreamCreate(&_m_trt_params.cuda_stream) != cudaSuccess) {
        LOG(ERROR) << "ERROR: cuda stream creation failed." << std::endl;
        return StatusCode::MODEL_INIT_FAILED;
    }

    return StatusCode::OK;
}

/****
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param in
 * @param out
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode AutoEncoderKL<INPUT, OUTPUT>::Impl::trt_run(const INPUT& in, OUTPUT& out) {
    // init sess
    auto& context = _m_trt_params.context;
    auto& input_binding = _m_trt_params.input_binding;
    auto& output_binding = _m_trt_params.output_binding;
    auto& input_device = _m_trt_params.input_device;
    auto& output_device = _m_trt_params.output_device;
    auto& output_host = _m_trt_params.output_host;
    auto& cuda_stream = _m_trt_params.cuda_stream;

    // preprocess input data
    auto& decode_input_data = in.decode_data;
    auto input_ele_counts = _m_input_size.area() * _m_input_channels;
    if (decode_input_data.size() != input_ele_counts) {
        LOG(INFO) << "wrong input data size, expected: " << input_ele_counts
                  << ", got: " << decode_input_data.size()
                  << " instead";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // h2d data transfer
    auto input_mem_size = input_binding.volume() * sizeof(float);
    auto cuda_status = cudaMemcpyAsync(
        input_device, decode_input_data.data(), input_mem_size, cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    context->setTensorAddress("input", input_device);
    context->setTensorAddress("decode_image", output_device);
    if (!context->enqueueV3(cuda_stream)) {
        LOG(ERROR) << "execute input data for inference failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // d2h data transfer
    cuda_status = cudaMemcpyAsync(
        output_host, output_device, output_binding.volume() * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cudaStreamSynchronize(cuda_stream);

    // decode output
    auto sample_output = trt_decode_output();

    // transform internal output into external output
    out = autoencoder_kl_impl::transform_output<OUTPUT>(sample_output);
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
autoencoder_kl_impl::internal_output AutoEncoderKL<INPUT, OUTPUT>::Impl::trt_decode_output() {
    // fetch origin decoded output
    auto elem_counts = _m_trt_params.output_binding.volume();
    std::vector<uint8_t> decode_out(elem_counts, 0.0f);
    for (auto idx = 0; idx < elem_counts; ++idx) {
        auto pix_value = _m_trt_params.output_host[idx] / 2.0f + 0.5f;
        pix_value = pix_value < 0.0f ? 0.0f : pix_value;
        pix_value = pix_value > 1.0f ? 1.0f : pix_value;
        decode_out[idx] = static_cast<uint8_t>(pix_value * 255.0f);
    }
    auto c = _m_trt_params.output_binding.dims().d[1];
    auto h = _m_trt_params.output_binding.dims().d[2];
    auto w = _m_trt_params.output_binding.dims().d[3];
    auto hwc_data = CvUtils::convert_to_hwc_vec<uint8_t>(decode_out, c, h, w);

    std_vae_decode_output out;
    cv::Size decode_image_size(_m_trt_params.output_binding.dims().d[3], _m_trt_params.output_binding.dims().d[2]);
    int decode_image_channels = _m_trt_params.output_binding.dims().d[1];
    auto mat_dtype = CV_8UC3;
    if (decode_image_channels == 1) {
        mat_dtype = CV_8UC1;
    }
    if (decode_image_channels == 4) {
        mat_dtype = CV_8UC4;
    }
    auto decode_out_img = cv::Mat(decode_image_size, mat_dtype, hwc_data.data());
    cv::cvtColor(decode_out_img, out.decode_output, cv::COLOR_RGB2BGR);
    return out;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode AutoEncoderKL<INPUT, OUTPUT>::Impl::init_onnx(const toml::value &cfg) {
    // ort env and memo info
    _m_onnx_params.env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "");

    // init session
    _m_onnx_params.model_file_path = cfg.at("model_file_path").as_string();
    if (!FilePathUtil::is_file_exist(_m_onnx_params.model_file_path)) {
        LOG(ERROR) << "autoencoder-kl model file path: " << _m_onnx_params.model_file_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_onnx_params.device = cfg.at("compute_backend").as_string();
    if (std::strcmp(_m_onnx_params.device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_onnx_params.device_id = static_cast<int>(cfg.at("gpu_device_id").as_integer());
    }
    _m_onnx_params.thread_nums = cfg.at("model_threads_num").as_integer();
    _m_onnx_params.session_options = Ort::SessionOptions();
    _m_onnx_params.session_options.SetIntraOpNumThreads(_m_onnx_params.thread_nums);
    _m_onnx_params.session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    _m_onnx_params.session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_onnx_params.device_id;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        cuda_options.gpu_mem_limit = 0;
        cuda_options.arena_extend_strategy = 1;
        cuda_options.do_copy_in_default_stream = 1;
        cuda_options.has_user_compute_stream = 0;
        cuda_options.default_memory_arena_cfg = nullptr;
        _m_onnx_params.session_options.AppendExecutionProvider_CUDA(cuda_options);
        _m_onnx_params.session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }
    _m_onnx_params.session = new Ort::Session(_m_onnx_params.env, _m_onnx_params.model_file_path.c_str(), _m_onnx_params.session_options);

    // init input/output nodes info
    auto input_nodes_counts = _m_onnx_params.session->GetInputCount();
    for (size_t i = 0 ; i < input_nodes_counts ; i++) {
        auto input_node_name = strdup(_m_onnx_params.session->GetInputNameAllocated(i, _m_onnx_params.allocator).get());
        auto input_node_shape = _m_onnx_params.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        _m_onnx_params.input_node_names.push_back(std::move(input_node_name));
        _m_onnx_params.input_node_shapes.push_back(input_node_shape);
        if (std::strcmp(input_node_name, "input") == 0) {
            _m_input_channels = input_node_shape[1];
            _m_input_size.height = input_node_shape[2];
            _m_input_size.width = input_node_shape[3];
        }
    }

    auto output_nodes_counts = _m_onnx_params.session->GetOutputCount();
    for (size_t i = 0 ; i < output_nodes_counts ; i++) {
        auto output_node_name = strdup(_m_onnx_params.session->GetOutputNameAllocated(i, _m_onnx_params.allocator).get());
        auto output_node_shape = _m_onnx_params.session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        _m_onnx_params.output_node_names.push_back(std::move(output_node_name));
        _m_onnx_params.output_node_shapes.push_back(output_node_shape);
    }

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
template <typename INPUT, typename OUTPUT>
StatusCode AutoEncoderKL<INPUT, OUTPUT>::Impl::onnx_run(const INPUT &in, OUTPUT &out) {
    // init sess
    auto& sess = _m_onnx_params.session;
    auto& input_node_shapes = _m_onnx_params.input_node_shapes;
    auto& input_node_names = _m_onnx_params.input_node_names;
    auto& output_node_names = _m_onnx_params.output_node_names;

    // preprocess input data
    auto& input_data = in.decode_data;
    auto input_ele_counts = _m_input_size.area() * _m_input_channels;
    if (input_data.size() != input_ele_counts) {
        LOG(INFO) << "wrong input data size, expected: " << input_ele_counts << ", got: " << input_data.size() << " instead";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // prepare input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
    auto input_xt_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)input_data.data(), input_data.size(), input_node_shapes[0].data(), input_node_shapes[0].size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_xt_tensor));

    // run session
    auto output_tensors = sess->Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(),
        output_node_names.data() , output_node_names.size());

    // decode output
    auto sample_output = onnx_decode_output(output_tensors[0]);

    // transform output
    out = autoencoder_kl_impl::transform_output<OUTPUT>(sample_output);

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
autoencoder_kl_impl::internal_output AutoEncoderKL<INPUT, OUTPUT>::Impl::onnx_decode_output(Ort::Value& predict_output) {
    // copy output tensor values
    autoencoder_kl_impl::internal_output internal_out;
    int out_elem_size = 1;
    for (auto &val : predict_output.GetTensorTypeAndShapeInfo().GetShape()) {
        out_elem_size *= static_cast<int>(val);
    }

    std::vector<uint8_t> decode_out(out_elem_size, 0.0f);
    for (auto idx = 0; idx < out_elem_size; ++idx) {
        auto pix_value = predict_output.template GetTensorMutableData<float>()[idx] / 2.0f + 0.5f;
        pix_value = pix_value < 0.0f ? 0.0f : pix_value;
        pix_value = pix_value > 1.0f ? 1.0f : pix_value;
        decode_out[idx] = static_cast<uint8_t>(pix_value * 255.0f);
    }
    auto c = _m_onnx_params.output_node_shapes[0][1];
    auto h = _m_onnx_params.output_node_shapes[0][2];
    auto w = _m_onnx_params.output_node_shapes[0][3];
    auto hwc_data = CvUtils::convert_to_hwc_vec<uint8_t>(decode_out, c, h, w);

    // construct sampled image
    std_vae_decode_output out;
    auto& output_node_shape = _m_onnx_params.output_node_shapes[0];
    cv::Size decode_image_size(output_node_shape[3], output_node_shape[2]);
    int decode_image_channels = output_node_shape[1];
    auto mat_dtype = CV_8UC3;
    if (decode_image_channels == 1) {
        mat_dtype = CV_8UC1;
    }
    if (decode_image_channels == 4) {
        mat_dtype = CV_8UC4;
    }
    auto decode_out_img = cv::Mat(decode_image_size, mat_dtype, hwc_data.data());
    cv::cvtColor(decode_out_img, out.decode_output, cv::COLOR_RGB2BGR);
    return out;
}

/************* Export Function Sets *************/

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
AutoEncoderKL<INPUT, OUTPUT>::AutoEncoderKL() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
AutoEncoderKL<INPUT, OUTPUT>::~AutoEncoderKL() = default;

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param cfg
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode AutoEncoderKL<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @return
 */
template <typename INPUT, typename OUTPUT>
bool AutoEncoderKL<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode AutoEncoderKL<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace diffusion
} // namespace models
} // namespace jinq