/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: hrnet_segmentation.inl
 * Date: 23-11-17
 ************************************************/

#include "hrnet_segmentation.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "TensorRT-8.6.1.6/NvInferRuntime.h"

#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "common/base64.h"
#include "models/trt_helper/trt_helper.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::Base64;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;
using jinq::common::Timestamp;

namespace scene_segmentation {
using jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;

using trt_helper::EngineBinding;
using trt_helper::DeviceMemory;
using trt_helper::TrtHelper;
using trt_helper::TrtLogger;

namespace hrnet_impl {

using internal_input = mat_input;
using internal_output = std_scene_segmentation_output;

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
    return in;
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
    auto image_decode_string = jinq::common::Base64::base64_decode(in.input_image_content);
    std::vector<uchar> image_vec_data(image_decode_string.begin(), image_decode_string.end());

    if (image_vec_data.empty()) {
        DLOG(WARNING) << "image data empty";
        return result;
    } else {
        cv::Mat ret;
        result.input_image = cv::imdecode(image_vec_data, cv::IMREAD_UNCHANGED);
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_scene_segmentation_output>::type>::value, std_scene_segmentation_output>::type
transform_output(const hrnet_impl::internal_output& internal_out) {
    return internal_out;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class HRNetSegmentation<INPUT, OUTPUT>::Impl {
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
            cudaFreeHost(_m_trt_params.output_host);
            cudaStreamDestroy(_m_trt_params.cuda_stream);
        } else if (_m_backend_type == ONNX) {

        } else {
            if (nullptr != _m_mnn_params.net && nullptr != _m_mnn_params.session) {
                _m_mnn_params.net->releaseModel();
                _m_mnn_params.net->releaseSession(_m_mnn_params.session);
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
        // output tensor host
        std::vector<float > segment_ret_host;
    };

    struct TRTParams {
        std::string model_file_path;
        // trt env params
        TrtLogger logger;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        cudaStream_t cuda_stream = nullptr;
        // trt bindings
        EngineBinding input_image_binding;
        EngineBinding out_segment_binding;
        // trt memory
        void* input_device = nullptr;
        void* output_device = nullptr;
        int32_t* output_host = nullptr;
    };

    struct MNNParams {
        // model file path
        std::string model_file_path;
        // mnn env params
        MNN::Interpreter* net = nullptr;
        MNN::Session* session = nullptr;
        MNN::Tensor* input_tensor = nullptr;
        MNN::Tensor* output_tensor = nullptr;
        // threads count
        uint threads_nums = 4;
    };

    enum BackendType {
        TRT = 0,
        ONNX = 1,
        MNN = 2,
    };

  private:
    // model backend type
    BackendType _m_backend_type = TRT;
    // onnx net params
    ONNXParams _m_onnx_params;
    // trt net params
    TRTParams _m_trt_params;
    // mnn net params
    MNNParams _m_mnn_params;
    // user input tensor size
    cv::Size _m_input_size_user = cv::Size();
    // model input tensor size
    cv::Size _m_input_size_host = cv::Size();
    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     * preprocess image
     * @param input_image : 输入图像
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @param config
     * @return
     */
    StatusCode init_trt(const toml::value& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode trt_run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @param config
     * @return
     */
    StatusCode init_onnx(const toml::value& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode onnx_run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @param config
     * @return
     */
    StatusCode init_mnn(const toml::value& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode mnn_run(const INPUT& in, OUTPUT& out);
};

/***
*
* @param cfg_file_path
* @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("HRNET_SEGMENTATION").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init hrnet configs
    toml::value hrnet_cfg;
    if (_m_backend_type == TRT) {
        hrnet_cfg = config.at("HRNET_SEGMENTATION_TRT");
    } else if (_m_backend_type == ONNX) {
        hrnet_cfg = config.at("HRNET_SEGMENTATION_ONNX");
    } else {
        hrnet_cfg = config.at("HRNET_SEGMENTATION_MNN");
    }
    auto model_file_name = FilePathUtil::get_file_name(hrnet_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == TRT) {
        init_status = init_trt(hrnet_cfg);
    } else if (_m_backend_type == ONNX) {
        init_status = init_onnx(hrnet_cfg);
    } else {
        init_status = init_mnn(hrnet_cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << "Successfully load hrnet segmentation model from: " << model_file_name;
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load hrnet segmentation model from: " << model_file_name;
    }

    return init_status;
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param input_image
* @return
 */
template<typename INPUT, typename OUTPUT>
cv::Mat HRNetSegmentation<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
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

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    StatusCode infer_status;
    if (_m_backend_type == TRT) {
        infer_status = trt_run(in, out);
    } else if (_m_backend_type == ONNX) {
        infer_status = onnx_run(in, out);
    } else {
        infer_status = mnn_run(in, out);
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
template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::Impl::init_trt(const toml::value &config) {
    // init trt runtime
    _m_trt_params.logger = TrtLogger();
    _m_trt_params.runtime = nvinfer1::createInferRuntime(_m_trt_params.logger);
    if(nullptr == _m_trt_params.runtime) {
        LOG(ERROR) << "init tensorrt runtime failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init trt engine
    if (!config.contains("model_file_path")) {
        LOG(ERROR) << "config doesn\'t have model_file_path field";
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_trt_params.model_file_path = config.at("model_file_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_trt_params.model_file_path)) {
        LOG(ERROR) << "hrnet trt segmentation model file: " << _m_trt_params.model_file_path << " not exist";
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
    std::string input_node_name = "x";
    auto successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, input_node_name, _m_trt_params.input_image_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.input_image_binding.dims().nbDims != 4) {
        std::string input_shape_str = TrtHelper::dims_to_string(_m_trt_params.input_image_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = _m_trt_params.input_image_binding.dims().d[2];
    _m_input_size_host.width = _m_trt_params.input_image_binding.dims().d[3];

    // bind output tensor
    std::string output_node_name = "argmax_0.tmp_0";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.out_segment_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind predicted segmentation output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.out_segment_binding.dims().nbDims != 3) {
        std::string output_shape_str = TrtHelper::dims_to_string(_m_trt_params.out_segment_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup input host/device memory
    auto memo_size = _m_trt_params.input_image_binding.volume() * sizeof(float);
    auto cuda_status = cudaMalloc(&_m_trt_params.input_device, memo_size);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "allocate device memory for input image failed, err str: " << cudaGetErrorString(cuda_status);
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup output host/device memory
    memo_size = _m_trt_params.out_segment_binding.volume() * sizeof(int32_t );
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

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::Impl::trt_run(const INPUT &in, OUTPUT &out) {
    // init sess
    auto* context = _m_trt_params.context;
    auto& input_image_binding = _m_trt_params.input_image_binding;
    auto& out_segment_binding = _m_trt_params.out_segment_binding;
    auto& cuda_stream = _m_trt_params.cuda_stream;
    auto& input_device = _m_trt_params.input_device;
    auto& output_device = _m_trt_params.output_device;
    auto& output_host = _m_trt_params.output_host;

    // transform external input into internal input
    auto internal_in = hrnet_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess input data
    auto& input_image = internal_in.input_image;
    _m_input_size_user = input_image.size();
    auto preprocessed_image = preprocess_image(input_image);
    auto input_chw_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // h2d data transfer
    auto input_mem_size = input_image_binding.volume() * sizeof(float);
    auto cuda_status = cudaMemcpyAsync(
        input_device, (float*)input_chw_data.data(), input_mem_size, cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    context->setInputTensorAddress("x", input_device);
    context->setTensorAddress("argmax_0.tmp_0", output_device);
    if (!context->enqueueV3(cuda_stream)) {
        LOG(ERROR) << "execute input data for inference failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // async copy inference result back to host
    cuda_status = cudaMemcpyAsync(
        output_host, output_device, out_segment_binding.volume() * sizeof(int32_t ),
        cudaMemcpyDeviceToHost, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cudaStreamSynchronize(cuda_stream);

    // transform internal output into external output
    cv::Mat result_image(_m_input_size_host, CV_32S, output_host);
    cv::resize(result_image, result_image, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);
    hrnet_impl::internal_output internal_out;
    internal_out.segmentation_result = result_image;
    out = hrnet_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::Impl::init_onnx(const toml::value &config) {
    // ort env and memo info
    _m_onnx_params.env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "");

    // init light glue session
    _m_onnx_params.model_file_path = config.at("model_file_path").as_string();
    if (!FilePathUtil::is_file_exist(_m_onnx_params.model_file_path)) {
        LOG(ERROR) << "hrnet segmentation model file path: " << _m_onnx_params.model_file_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_onnx_params.device = config.at("compute_backend").as_string();
    if (std::strcmp(_m_onnx_params.device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_onnx_params.device_id = static_cast<int>(config.at("gpu_device_id").as_integer());
    }
    _m_onnx_params.thread_nums = config.at("model_threads_num").as_integer();
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
    }
    _m_input_size_host.height = _m_onnx_params.input_node_shapes[0][2];
    _m_input_size_host.width = _m_onnx_params.input_node_shapes[0][3];

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
template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::Impl::onnx_run(const INPUT &in, OUTPUT &out) {
    // init sess
    auto& sess = _m_onnx_params.session;
    auto& input_node_shapes = _m_onnx_params.input_node_shapes;
    auto& input_node_names = _m_onnx_params.input_node_names;
    auto& output_node_shapes = _m_onnx_params.output_node_shapes;
    auto& output_node_names = _m_onnx_params.output_node_names;
    auto& output_segment_value = _m_onnx_params.segment_ret_host;

    // transform external input into internal input
    auto internal_in = hrnet_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    input_node_shapes[0][2] = preprocessed_image.rows;
    input_node_shapes[0][3] = preprocessed_image.cols;
    auto input_image_chw_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // prepare input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)input_image_chw_data.data(), input_image_chw_data.size(),
        input_node_shapes[0].data(), input_node_shapes[0].size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    // run session
    auto output_tensors = sess->Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(),
        output_node_names.data() , output_node_names.size());

    // copy output tensor values
    auto& out_segment_tensor = output_tensors[0];
    int out_segment_tensor_size = 1;
    for (auto &val : out_segment_tensor.GetTensorTypeAndShapeInfo().GetShape()) {
        out_segment_tensor_size *= val;
    }
    output_segment_value.resize(out_segment_tensor_size);
    for (auto idx = 0; idx < out_segment_tensor_size; ++idx) {
        output_segment_value[idx] = out_segment_tensor.template GetTensorMutableData<int64_t>()[idx];
    }

    // transform output
    cv::Mat result_image(_m_input_size_host, CV_32F, output_segment_value.data());
    cv::resize(result_image, result_image, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);
    hrnet_impl::internal_output internal_out;
    internal_out.segmentation_result = result_image;
    out = hrnet_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::Impl::init_mnn(const toml::value &config) {
    // init threads
    if (!config.contains("model_threads_num")) {
        LOG(WARNING) << "Config doesn\'t have model_threads_num field default 4";
        _m_mnn_params.threads_nums = 4;
    } else {
        _m_mnn_params.threads_nums = static_cast<int>(config.at("model_threads_num").as_integer());
    }

    // init Interpreter
    if (!config.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_mnn_params.model_file_path = config.at("model_file_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_mnn_params.model_file_path)) {
        LOG(ERROR) << "metric3d model file: " << _m_mnn_params.model_file_path << " not exist";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_mnn_params.net = MNN::Interpreter::createFromFile(_m_mnn_params.model_file_path.c_str());
    if (nullptr == _m_mnn_params.net) {
        LOG(ERROR) << "create hrnetw48 model interpreter failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init Session
    MNN::ScheduleConfig mnn_config;
    if (!config.contains("compute_backend")) {
        LOG(WARNING) << "Config doesn\'t have compute_backend field default cpu";
        mnn_config.type = MNN_FORWARD_CPU;
    } else {
        std::string compute_backend = config.at("compute_backend").as_string();
        if (std::strcmp(compute_backend.c_str(), "cuda") == 0) {
            mnn_config.type = MNN_FORWARD_CUDA;
        } else if (std::strcmp(compute_backend.c_str(), "cpu") == 0) {
            mnn_config.type = MNN_FORWARD_CPU;
        } else {
            LOG(WARNING) << "not supported compute backend use default cpu instead";
            mnn_config.type = MNN_FORWARD_CPU;
        }
    }

    mnn_config.numThread = _m_mnn_params.threads_nums;
    MNN::BackendConfig backend_config;
    if (!config.contains("backend_precision_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_precision_mode field default Precision_Normal";
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>
            (config.at("backend_precision_mode").as_integer());
    }
    if (!config.contains("backend_power_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_power_mode field default Power_Normal";
        backend_config.power = MNN::BackendConfig::Power_Normal;
    } else {
        backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(config.at("backend_power_mode").as_integer());
    }
    mnn_config.backendConfig = &backend_config;

    _m_mnn_params.session = _m_mnn_params.net->createSession(mnn_config);
    if (nullptr == _m_mnn_params.session) {
        LOG(ERROR) << "create metric3d model session failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_mnn_params.input_tensor = _m_mnn_params.net->getSessionInput(_m_mnn_params.session, "x");
    _m_mnn_params.output_tensor = _m_mnn_params.net->getSessionOutput(_m_mnn_params.session, "argmax_0.tmp_0");
    if (nullptr == _m_mnn_params.input_tensor) {
        LOG(ERROR) << "fetch hrnet segmentation model input node failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (nullptr == _m_mnn_params.output_tensor) {
        LOG(ERROR) << "fetch hrnet segmentation model output node failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init hose size and user size
    _m_input_size_host.width = _m_mnn_params.input_tensor->width();
    _m_input_size_host.height = _m_mnn_params.input_tensor->height();

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
StatusCode HRNetSegmentation<INPUT, OUTPUT>::Impl::mnn_run(const INPUT &in, OUTPUT &out) {
    // init sess envs
    auto* net = _m_mnn_params.net;
    auto* session = _m_mnn_params.session;
    auto* input_tensor = _m_mnn_params.input_tensor;
    auto* output_tensor = _m_mnn_params.output_tensor;

    // transform external input into internal input
    auto internal_in = hrnet_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run session
    MNN::Tensor input_tensor_user(input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    ::memcpy(input_tensor_data, input_chw_data.data(), input_chw_data.size());
    input_tensor->copyFromHostTensor(&input_tensor_user);
    net->runSession(session);

    // fetch net output
    MNN::Tensor output_tensor_user(output_tensor, MNN::Tensor::DimensionType::CAFFE);
    output_tensor->copyToHostTensor(&output_tensor_user);
    auto host_data = output_tensor_user.host<int>();
    cv::Mat result_image(_m_input_size_host, CV_32SC1, host_data);
    cv::resize(result_image, result_image, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);

    // transform internal output into external output
    hrnet_impl::internal_output internal_out;
    internal_out.segmentation_result = result_image;
    out = hrnet_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
HRNetSegmentation<INPUT, OUTPUT>::HRNetSegmentation() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
HRNetSegmentation<INPUT, OUTPUT>::~HRNetSegmentation() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool HRNetSegmentation<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode HRNetSegmentation<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}