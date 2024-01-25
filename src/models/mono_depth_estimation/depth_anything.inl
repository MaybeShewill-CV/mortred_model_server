/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: depth_anything.inl
 * Date: 24-1-25
 ************************************************/

#include "depth_anything.h"

#include <random>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include "TensorRT-8.6.1.6/NvInferRuntime.h"

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
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::trt_helper::EngineBinding;
using jinq::models::trt_helper::DeviceMemory;
using jinq::models::trt_helper::TrtHelper;
using jinq::models::trt_helper::TrtLogger;

namespace mono_depth_estimation {

using jinq::models::io_define::mono_depth_estimation::std_mde_output;

namespace depth_anything_impl {

using internal_input = mat_input;
using internal_output = std_mde_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
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
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<mat_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    return in;
}

/***
*
* @tparam INPUT
* @param in
* @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<base64_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    internal_input result{};
    auto image_decode_string = jinq::common::Base64::base64_decode(in.input_image_content);
    std::vector<uchar> image_vec_data(image_decode_string.begin(), image_decode_string.end());

    if (image_vec_data.empty()) {
        LOG(WARNING) << "image data empty";
        return result;
    } else {
        cv::Mat ret;
        result.input_image = cv::imdecode(image_vec_data, cv::IMREAD_COLOR);
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_mde_output>::type>::value, std_mde_output>::type
transform_output(const depth_anything_impl::internal_output& internal_out) {
    return internal_out;
}

} // namespace depth_anything_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class DepthAnything<INPUT, OUTPUT>::Impl {
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
            auto status = cudaStreamDestroy(_m_trt_params.cuda_stream);
            if (status != cudaSuccess) {
                LOG(ERROR) << "failed to free DepthAnything trt object. destruct cuda stream "
                              "failed code str: " << cudaGetErrorString(status);
            }
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

    enum BackendType {
        TRT = 0,
        MNN = 1,
    };

  private:
    // model backend type
    BackendType _m_backend_type = TRT;

    // trt net params
    TRTParams _m_trt_params;

    // input image size
    cv::Size _m_input_size_user = cv::Size();
    //　input node size
    cv::Size _m_input_size_host = cv::Size();
    // focal length
    float _m_focal_length = 0.0f;
    // intrinsic params
    std::vector<float> _m_intrinsic_params = {0.0, 0.0, 0.0, 0.0};

    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     * preprocess
     * @param input_image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image);

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
    depth_anything_impl::internal_output trt_decode_output();
};

/***
*
* @param cfg_file_path
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DepthAnything<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("DEPTH_ANYTHING").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init depth anything configs
    toml::value model_cfg;
    if (_m_backend_type == MNN) {
        model_cfg = config.at("DEPTH_ANYTHING_MNN");
    } else {
        model_cfg = config.at("DEPTH_ANYTHING_TRT");
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
        LOG(INFO) << "Successfully load DepthAnything model from: " << model_file_name;
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load DepthAnything model from: " << model_file_name;
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
StatusCode DepthAnything<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
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
* @param input_image
* @return
 */
template <typename INPUT, typename OUTPUT>
cv::Mat DepthAnything<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) {
    // keep ratio rescale input image
    int rescale_w;
    int rescale_h;
    float aspect_ratio = (float)input_image.cols / (float)input_image.rows;
    if (aspect_ratio >= 1) {
        rescale_w = _m_input_size_host.width;
        rescale_h = int(_m_input_size_host.height / aspect_ratio);
    } else {
        rescale_w = int(_m_input_size_host.width * aspect_ratio);
        rescale_h = _m_input_size_host.height;
    }

    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(rescale_w, rescale_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat out = cv::Mat::zeros(_m_input_size_host, CV_8UC3);
    resized_image.copyTo(out(cv::Rect(0, 0, resized_image.cols, resized_image.rows)));

    // convert image type
    out.convertTo(out, CV_32FC3);

    // normalize image data
    cv::divide(out, 255, out);
    cv::subtract(out, cv::Scalar(0.406f, 0.456f, 0.485f), out);
    cv::divide(out, cv::Scalar(0.225f, 0.224f, 0.229f), out);

    return out;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DepthAnything<INPUT, OUTPUT>::Impl::init_trt(const toml::value& cfg) {
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
        LOG(ERROR) << "DepthAnything trt estimation model file: " << _m_trt_params.model_file_path << " not exist";
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
    _m_input_size_host.height = _m_trt_params.input_binding.dims().d[2];
    _m_input_size_host.width = _m_trt_params.input_binding.dims().d[3];

    // bind output tensor
    std::string output_node_name = "output";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind predicted depth output tensor failed";
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

    // init intrinsic and canonical size
    _m_focal_length = static_cast<float>(cfg.at("focal_length").as_floating());
    _m_intrinsic_params = {
        static_cast<float>(cfg.at("intrinsic").as_array()[0].as_floating()),
        static_cast<float>(cfg.at("intrinsic").as_array()[1].as_floating()),
        static_cast<float>(cfg.at("intrinsic").as_array()[2].as_floating()),
        static_cast<float>(cfg.at("intrinsic").as_array()[3].as_floating()),
    };

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
StatusCode DepthAnything<INPUT, OUTPUT>::Impl::trt_run(const INPUT& in, OUTPUT& out) {
    // init sess
    auto& context = _m_trt_params.context;
    auto& input_binding = _m_trt_params.input_binding;
    auto& output_binding = _m_trt_params.output_binding;
    auto& input_device = _m_trt_params.input_device;
    auto& output_device = _m_trt_params.output_device;
    auto& output_host = _m_trt_params.output_host;
    auto& cuda_stream = _m_trt_params.cuda_stream;

    // transform external input into internal input
    auto internal_in = depth_anything_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess input data
    auto& input_image = internal_in.input_image;
    _m_input_size_user = input_image.size();
    auto preprocessed_image = preprocess_image(input_image);
    auto input_chw_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // h2d data transfer
    auto input_mem_size = input_binding.volume() * sizeof(float);
    auto cuda_status = cudaMemcpyAsync(
        input_device, (float*)input_chw_data.data(), input_mem_size, cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    context->setTensorAddress("input", input_device);
    context->setTensorAddress("output", output_device);
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
    auto depth_out = trt_decode_output();

    // transform internal output into external output
    out = depth_anything_impl::transform_output<OUTPUT>(depth_out);
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
depth_anything_impl::internal_output DepthAnything<INPUT, OUTPUT>::Impl::trt_decode_output() {
    // fetch origin depth map
    cv::Mat depth_map(_m_input_size_host, CV_32FC1, _m_trt_params.output_host);

    // rescale depth map into origin scale
    int crop_w;
    int crop_h;
    if (_m_input_size_user.width > _m_input_size_user.height) {
        crop_w = _m_input_size_host.width;
        crop_h = _m_input_size_host.height * _m_input_size_user.height / _m_input_size_user.width;
    } else {
        crop_w = _m_input_size_host.width * _m_input_size_user.width / _m_input_size_user.height;
        crop_h = _m_input_size_host.height;
    }
    cv::resize(depth_map(cv::Rect(0, 0, crop_w, crop_h)), depth_map, _m_input_size_user);

    // colorize depth map
    cv::Mat colorized_depth_map;
    CvUtils::colorize_depth_map(depth_map, colorized_depth_map);

    // copy result
    std_mde_output out;
    out.depth_map = depth_map;
    out.colorized_depth_map = colorized_depth_map;
    return out;
}

/************* Export Function Sets *************/

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
DepthAnything<INPUT, OUTPUT>::DepthAnything() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
DepthAnything<INPUT, OUTPUT>::~DepthAnything() = default;

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param cfg
* @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode DepthAnything<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @return
 */
template <typename INPUT, typename OUTPUT>
bool DepthAnything<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode DepthAnything<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace mono_depth_estimation
} // namespace models
} // namespace jinq