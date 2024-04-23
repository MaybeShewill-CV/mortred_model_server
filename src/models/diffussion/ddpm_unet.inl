/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: DDPMUNet.cpp
 * Date: 24-4-23
 ************************************************/

#include "ddpm_unet.h"

#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
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

using jinq::models::io_define::diffusion::std_ddpm_unet_input;
using jinq::models::io_define::diffusion::std_ddpm_unet_output;

namespace ddpm_unet_impl {

using internal_input = std_ddpm_unet_input;
using internal_output = std_ddpm_unet_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<std_ddpm_unet_input>::type>::value, internal_input>::type
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_ddpm_unet_output >::type>::value, std_ddpm_unet_output>::type
transform_output(const ddpm_unet_impl::internal_output& internal_out) {
    return internal_out;
}

} // namespace ddpm_unet_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class DDPMUNet<INPUT, OUTPUT>::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        if (_m_backend_type == MNN) {
            // todo implement MNN infer
        }
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
        EngineBinding input_xt_binding;
        EngineBinding input_t_binding;
        EngineBinding output_binding;
        // trt memory
        void* input_xt_device = nullptr;
        void* input_t_device = nullptr;
        void* output_device = nullptr;
        float* output_host = nullptr;
    };

    enum BackendType {
        TRT = 0,
        MNN = 1,
    };

  private:
    // model backend type
    BackendType _m_backend_type = TRT;

    // trt net params
    TRTParams _m_trt_params;

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
    ddpm_unet_impl::internal_output trt_decode_output();
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DDPMUNet<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("DDPM_UNET").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init ddpm-unet configs
    toml::value model_cfg;
    if (_m_backend_type == MNN) {
        model_cfg = config.at("DDPM_UNET_MNN");
    } else {
        model_cfg = config.at("DDPM_UNET_TRT");
    }
    auto model_file_name = FilePathUtil::get_file_name(model_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == MNN) {
        // todo implement mnn inference
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        init_status = init_trt(model_cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << "Successfully load ddpm-unet model from: " << model_file_name;
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load ddpm-unet model from: " << model_file_name;
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
StatusCode DDPMUNet<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    StatusCode infer_status;
    if (_m_backend_type == MNN) {
        // todo implement mnn inference
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    } else {
        infer_status = trt_run(in, out);
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
StatusCode DDPMUNet<INPUT, OUTPUT>::Impl::init_trt(const toml::value& cfg) {
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
        LOG(ERROR) << "DDPMUNet trt estimation model file: " << _m_trt_params.model_file_path << " not exist";
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
    std::string input_node_name = "xt";
    auto successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, input_node_name, _m_trt_params.input_xt_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input xt tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.input_xt_binding.dims().nbDims != 4) {
        std::string input_shape_str = TrtHelper::dims_to_string(_m_trt_params.input_xt_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.input_xt_binding.is_dynamic()) {
        LOG(ERROR) << "not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_channels = _m_trt_params.input_xt_binding.dims().d[1];
    _m_input_size.height = _m_trt_params.input_xt_binding.dims().d[2];
    _m_input_size.width = _m_trt_params.input_xt_binding.dims().d[3];

    input_node_name = "t";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, input_node_name, _m_trt_params.input_t_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input t tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind output tensor
    std::string output_node_name = "predict_noise";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind predicted noise output tensor failed";
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
    auto memo_size = _m_trt_params.input_xt_binding.volume() * sizeof(float);
    auto cuda_status = cudaMalloc(&_m_trt_params.input_xt_device, memo_size);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "allocate device memory for input image failed, err str: " << cudaGetErrorString(cuda_status);
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    memo_size = _m_trt_params.input_t_binding.volume() * sizeof(int64_t );
    cuda_status = cudaMalloc(&_m_trt_params.input_t_device, memo_size);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "allocate device memory for input timesteps failed, err str: " << cudaGetErrorString(cuda_status);
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
StatusCode DDPMUNet<INPUT, OUTPUT>::Impl::trt_run(const INPUT& in, OUTPUT& out) {
    // init sess
    auto& context = _m_trt_params.context;
    auto& input_xt_binding = _m_trt_params.input_xt_binding;
    auto& input_t_binding = _m_trt_params.input_t_binding;
    auto& output_binding = _m_trt_params.output_binding;
    auto& input_xt_device = _m_trt_params.input_xt_device;
    auto& input_t_device = _m_trt_params.input_t_device;
    auto& output_device = _m_trt_params.output_device;
    auto& output_host = _m_trt_params.output_host;
    auto& cuda_stream = _m_trt_params.cuda_stream;

    // preprocess input data
    auto& xt = in.xt;
    auto& timestep = in.timestep;

    // h2d data transfer
    auto input_mem_size = input_xt_binding.volume() * sizeof(float);
    auto cuda_status = cudaMemcpyAsync(
        input_xt_device, (float*)xt.data(), input_mem_size, cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    input_mem_size = input_t_binding.volume() * sizeof(int64_t );
    cuda_status = cudaMemcpyAsync(
        input_t_device, &timestep, input_mem_size, cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    context->setTensorAddress("xt", input_xt_device);
    context->setTensorAddress("t", input_t_device);
    context->setTensorAddress("predict_noise", output_device);
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
    auto predict_noise = trt_decode_output();

    // transform internal output into external output
    out = ddpm_unet_impl::transform_output<OUTPUT>(predict_noise);
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
ddpm_unet_impl::internal_output DDPMUNet<INPUT, OUTPUT>::Impl::trt_decode_output() {
    // fetch origin predict noise
    auto elem_counts = _m_input_size.area() * _m_input_channels;
    std_ddpm_unet_output out;
    out.predict_noise.resize(elem_counts);
    for (auto idx = 0; idx < elem_counts; ++idx) {
        out.predict_noise[idx] = _m_trt_params.output_host[idx];
    }
    return out;
}

/************* Export Function Sets *************/

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
DDPMUNet<INPUT, OUTPUT>::DDPMUNet() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
DDPMUNet<INPUT, OUTPUT>::~DDPMUNet() = default;

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param cfg
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DDPMUNet<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @return
 */
template <typename INPUT, typename OUTPUT>
bool DDPMUNet<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode DDPMUNet<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace diffusion
} // namespace models
} // namespace jinq
