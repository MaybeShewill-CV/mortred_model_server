/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: lightglue.inl
* Date: 23-11-03
************************************************/

#include "lightglue.h"

#include <unordered_map>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "TensorRT-8.6.1.6/NvInferRuntime.h"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/trt_helper/trt_helper.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::Base64;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::pair_mat_input;

namespace feature_point {
using jinq::models::io_define::feature_point::fp;
using jinq::models::io_define::feature_point::matched_fp;
using jinq::models::io_define::feature_point::std_feature_point_match_output;

using trt_helper::EngineBinding;
using trt_helper::DeviceMemory;
using trt_helper::TrtHelper;
using trt_helper::TrtLogger;

namespace lightglue_impl {

struct internal_input {
    cv::Mat src_input_image;
    cv::Mat dst_input_image;
};
using internal_output = std_feature_point_match_output;

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<pair_mat_input>::type>::value, internal_input>::type transform_input(const INPUT &in) {
    internal_input result{};
    result.src_input_image = in.src_input_image;
    result.dst_input_image = in.dst_input_image;
    return result;
}

/***
 * transform different type of internal output into external output
 * @tparam EXTERNAL_OUTPUT
 * @tparam dummy
 * @param in
 * @return
 */
template <typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_feature_point_match_output>::type>::value, std_feature_point_match_output>::type
transform_output(const lightglue_impl::internal_output &internal_out) {
    std_feature_point_match_output result;
    for (auto& value : internal_out) {
        result.push_back(value);
    }
    return result;
}

/***
 * specified memo allocator for match result output tensor
 */
class MatchResultOutputAllocator : public nvinfer1::IOutputAllocator {
  public:
    /***
     *
     * @param tensorName
     * @param currentMemory
     * @param size
     * @param alignment
     * @return
     */
    void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override {
        LOG(INFO) << "reallocate cuda memo for dynamically shaped tensor: " << tensorName;
        // reallocate cuda memo
        if (size > output_size) {
            cudaFree(output_ptr);
            output_ptr = nullptr;
            output_size = 0;
            if (cudaMalloc(&output_ptr, size) == cudaSuccess) {
                LOG(INFO) << "successfully allocate cuda memo for: " << tensorName << ", size: " << size;
                output_size = size;
            }
        }
        return output_ptr;
    }

    /***
     *
     * @param tensorName
     * @param dims
     */
    void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override {
        output_dims = dims;
    }

    // Saved dimensions of the output tensor
    nvinfer1::Dims output_dims{};

    // nullptr if memory could not be allocated
    void* output_ptr{nullptr};

    // Size of allocation pointed to by output
    uint64_t output_size{0};

    /***
     *
     */
    ~MatchResultOutputAllocator() override {
        cudaFree(output_ptr);
    }
};

} // namespace lightglue_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT> 
class LightGlue<INPUT, OUTPUT>::Impl {
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
    StatusCode run(const INPUT &in, OUTPUT& out);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const { return _m_successfully_initialized; };

private:
    struct ONNXParams {
        std::string model_file_path;
        // ort env params
        int thread_nums = 1;
        std::string device = "cpu";
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
        std::vector<int64_t > kpts0_host;
        std::vector<int64_t > kpts1_host;
        std::vector<int64_t > matches_host;
        std::vector<float > match_scores_host;
        // feature point scales
        std::vector<float> scales = {1.0f , 1.0f};
    };

    struct TRTParams {
        std::string model_file_path;
        // trt env params
        TrtLogger logger;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* execution_context = nullptr;
        lightglue_impl::MatchResultOutputAllocator* match_output_allocator = nullptr;
        // trt bindings
        EngineBinding src_input_image_binding;
        EngineBinding dst_input_image_binding;
        EngineBinding out_kpts0_binding;
        EngineBinding out_kpts1_binding;
        EngineBinding out_matches_binding;
        EngineBinding out_match_scores_binding;
        // trt memory
        std::unordered_map<std::string, void*> device_memory;
        std::unordered_map<std::string, lightglue_impl::MatchResultOutputAllocator*> allocator_map;
        cudaStream_t cuda_stream = nullptr;
        // host memory
        std::vector<int64_t> out_kpts0_host;
        std::vector<int64_t> out_kpts1_host;
        std::vector<int64_t > out_matches_host;
        std::vector<float> out_match_scores_host;
        // max match points
        int max_matched_points = 2048;
    };

    enum BackendType {
        ONNX = 0,
        TRT = 1,
    };

  private:
    // model backend type
    BackendType _m_backend_type = ONNX;

    // onnx net params
    ONNXParams _m_onnx_params;

    // trt net params
    TRTParams _m_trt_params;

    // user input size
    cv::Size _m_src_input_size_user = cv::Size();
    cv::Size _m_dst_input_size_user = cv::Size();
    // match thresh value
    float _m_match_thresh = 0.0f;
    // long side length
    float _m_long_side_len = 512.0f;

    // flag
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
     * @return
     */
    lightglue_impl::internal_output onnx_decode_output();

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
     * @return
     */
    StatusCode setup_device_memory();
};

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode LightGlue<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("LIGHTGLUE").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init light-glue configs
    toml::value lightglue_cfg;
    if (_m_backend_type == ONNX) {
        lightglue_cfg = config.at("LIGHTGLUE_ONNX");
    } else {
        lightglue_cfg = config.at("LIGHTGLUE_TRT");
    }
    auto model_file_name = FilePathUtil::get_file_name(lightglue_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == ONNX) {
        init_status = init_onnx(lightglue_cfg);
    } else {
         init_status = init_trt(lightglue_cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << "Successfully load lightglue model from: " << model_file_name;
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load lightglue model from: " << model_file_name;
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
StatusCode LightGlue<INPUT, OUTPUT>::Impl::run(const INPUT &in, OUTPUT& out) {
    StatusCode infer_status;
    if (_m_backend_type == ONNX) {
        infer_status = onnx_run(in, out);
    } else {
        infer_status = trt_run(in, out);
    }
    return infer_status;
}

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT> 
cv::Mat LightGlue<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat &input_image) {
    // long side resize
    auto long_side = std::max(input_image.size().width, input_image.size().height);
    auto resize_scale = _m_long_side_len / static_cast<float>(long_side);
    auto resize_h = static_cast<int>(static_cast<float>(input_image.size().height) * resize_scale);
    auto resize_w = static_cast<int>(static_cast<float>(input_image.size().width) * resize_scale);
    cv::Mat tmp;
    cv::resize(input_image, tmp, cv::Size(resize_w, resize_h), 0.0, 0.0, cv::INTER_AREA);

    // convert bgr to gray
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);

    // normalize
    if (tmp.type() != CV_32FC1) {
        tmp.convertTo(tmp, CV_32FC1);
    }
    tmp /= 255.0;

    return tmp;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::Impl::init_onnx(const toml::value& config) {
    // ort env and memo info
    _m_onnx_params.env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "");

    // init light glue session
    _m_onnx_params.model_file_path = config.at("model_file_path").as_string();
    if (!FilePathUtil::is_file_exist(_m_onnx_params.model_file_path)) {
        LOG(ERROR) << "lightglue model file path: " << _m_onnx_params.model_file_path << " not exists";
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

    auto output_nodes_counts = _m_onnx_params.session->GetOutputCount();
    for (size_t i = 0 ; i < output_nodes_counts ; i++) {
        auto output_node_name = strdup(_m_onnx_params.session->GetOutputNameAllocated(i, _m_onnx_params.allocator).get());
        auto output_node_shape = _m_onnx_params.session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        _m_onnx_params.output_node_names.push_back(std::move(output_node_name));
        _m_onnx_params.output_node_shapes.push_back(output_node_shape);
    }

    // init match threshold
    _m_match_thresh = static_cast<float>(config.at("match_thresh").as_floating());

    // init long side length
    _m_long_side_len = static_cast<float>(config.at("long_side_length").as_floating());

    return StatusCode::OK;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::Impl::onnx_run(const INPUT &in, OUTPUT& out) {
    // init sess
    auto& sess = _m_onnx_params.session;
    auto& input_node_shapes = _m_onnx_params.input_node_shapes;
    auto& input_node_names = _m_onnx_params.input_node_names;
    auto& output_node_shapes = _m_onnx_params.output_node_shapes;
    auto& output_node_names = _m_onnx_params.output_node_names;
    auto& output_kpts0_value = _m_onnx_params.kpts0_host;
    auto& output_kpts1_value = _m_onnx_params.kpts1_host;
    auto& output_matches_value = _m_onnx_params.matches_host;
    auto& output_match_scores_value = _m_onnx_params.match_scores_host;

    // transform external input into internal input
    auto internal_in = lightglue_impl::transform_input(in);
    if (!internal_in.src_input_image.data || internal_in.src_input_image.empty() ||
        !internal_in.dst_input_image.data || internal_in.dst_input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_src_input_size_user = internal_in.src_input_image.size();
    cv::Mat src_preprocessed_image = preprocess_image(internal_in.src_input_image);
    input_node_shapes[0][2] = src_preprocessed_image.rows;
    input_node_shapes[0][3] = src_preprocessed_image.cols;
    _m_dst_input_size_user = internal_in.dst_input_image.size();
    cv::Mat dst_preprocessed_image = preprocess_image(internal_in.dst_input_image);
    input_node_shapes[1][2] = dst_preprocessed_image.rows;
    input_node_shapes[1][3] = dst_preprocessed_image.cols;

    // prepare input host data
    int src_input_tensor_size = 1;
    for (auto &val : input_node_shapes[0]) {
        src_input_tensor_size *= val;
    }
    std::vector<float> src_input_image_values(src_input_tensor_size);
    src_input_image_values.assign(src_preprocessed_image.begin<float>(), src_preprocessed_image.end<float>());

    int dst_input_tensor_size = 1;
    for (auto &val : input_node_shapes[1]) {
        dst_input_tensor_size *= val;
    }
    std::vector<float> dst_input_image_values(dst_input_tensor_size);
    dst_input_image_values.assign(dst_preprocessed_image.begin<float>(), dst_preprocessed_image.end<float>());

    // prepare input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
    auto src_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)src_input_image_values.data(), src_input_image_values.size(),
        input_node_shapes[0].data(), input_node_shapes[0].size());
    auto dst_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)dst_input_image_values.data(), dst_input_image_values.size(),
        input_node_shapes[1].data(), input_node_shapes[1].size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(src_input_tensor));
    input_tensors.push_back(std::move(dst_input_tensor));

    // run session
    auto output_tensors = sess->Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(),
        output_node_names.data() , output_node_names.size());

    // copy output tensor values
    auto& out_kpts0_tensor = output_tensors[0];
    int kpts0_size = 1;
    for (auto &val : out_kpts0_tensor.GetTensorTypeAndShapeInfo().GetShape()) {
        kpts0_size *= val;
    }
    output_kpts0_value.resize(kpts0_size);
    for (auto idx = 0; idx < kpts0_size; ++idx) {
        output_kpts0_value[idx] = out_kpts0_tensor.template GetTensorMutableData<int64_t>()[idx];
    }

    auto& out_kpts1_tensor = output_tensors[1];
    auto kpts1_size = 1;
    for (auto &val : out_kpts1_tensor.GetTensorTypeAndShapeInfo().GetShape()) {
        kpts1_size *= val;
    }
    output_kpts1_value.resize(kpts1_size);
    for (auto idx = 0; idx < kpts1_size; ++idx) {
        output_kpts1_value[idx] = out_kpts1_tensor.template GetTensorMutableData<int64_t>()[idx];
    }

    auto& out_matches_tensor = output_tensors[2];
    auto matches_size = 1;
    for (auto &val : out_matches_tensor.GetTensorTypeAndShapeInfo().GetShape()) {
        matches_size *= val;
    }
    output_matches_value.resize(matches_size);
    for (auto idx = 0; idx < matches_size; ++idx) {
        output_matches_value[idx] = out_matches_tensor.template GetTensorMutableData<int64_t>()[idx];
    }

    auto& out_match_scores_tensor = output_tensors[3];
    auto match_scores_size = 1;
    for (auto &val : out_match_scores_tensor.GetTensorTypeAndShapeInfo().GetShape()) {
        match_scores_size *= val;
    }
    output_match_scores_value.resize(match_scores_size);
    for (auto idx = 0; idx < match_scores_size; ++idx) {
        output_match_scores_value[idx] = out_match_scores_tensor.template GetTensorMutableData<float>()[idx];
    }

    // decode output
    auto decode_output = onnx_decode_output();

    // transform output
    out = lightglue_impl::transform_output<OUTPUT>(decode_output);

    return StatusCode::OK;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
std_feature_point_match_output LightGlue<INPUT, OUTPUT>::Impl::onnx_decode_output() {
    // init output
    auto& out_kpts0 = _m_onnx_params.kpts0_host;
    auto& out_kpts1 = _m_onnx_params.kpts1_host;
    auto& out_matches = _m_onnx_params.matches_host;
    auto& out_match_scores = _m_onnx_params.match_scores_host;
    auto& input_node_shapes = _m_onnx_params.input_node_shapes;

    // rescale kpts
    auto kpts0_w_scale = static_cast<float>(input_node_shapes[0][3]) / static_cast<float>(_m_src_input_size_user.width);
    auto kpts0_h_scale = static_cast<float>(input_node_shapes[0][2]) / static_cast<float>(_m_src_input_size_user.height);
    std::vector<cv::Point2f> kpts0;
    for (auto idx = 0; idx < out_kpts0.size(); idx += 2) {
        auto fpt = cv::Point2f((out_kpts0[idx] + 0.5) / kpts0_w_scale - 0.5 , (out_kpts0[idx + 1] + 0.5) / kpts0_h_scale - 0.5);
        kpts0.push_back(fpt);
    }

    auto kpts1_w_scale = static_cast<float>(input_node_shapes[1][3]) / static_cast<float>(_m_dst_input_size_user.width);
    auto kpts1_h_scale = static_cast<float>(input_node_shapes[1][2]) / static_cast<float>(_m_dst_input_size_user.height);
    std::vector<cv::Point2f> kpts1;
    for (auto idx = 0; idx < out_kpts1.size(); idx += 2) {
        auto fpt = cv::Point2f((out_kpts1[idx] + 0.5) / kpts1_w_scale - 0.5 , (out_kpts1[idx + 1] + 0.5) / kpts1_h_scale - 0.5);
        kpts1.push_back(fpt);
    }

    // fetch valid matched feature points
    assert(out_match_scores.size() * 2 == out_matches.size());
    std::vector<matched_fp> matched_fpts;
    for (int idx = 0; idx < out_match_scores.size(); idx++) {
        auto match_score = out_match_scores[idx];
        if (match_score < _m_match_thresh) {
            continue;
        }
        auto kpt0_idx = out_matches[idx * 2];
        auto kpt1_idx = out_matches[idx * 2 + 1];
        if (kpt0_idx < 0 || kpt0_idx >= kpts0.size() || kpt1_idx < 0 || kpt1_idx >= kpts1.size()) {
            continue;
        }
        auto kpt0 = kpts0[kpt0_idx];
        auto kpt1 = kpts1[kpt1_idx];
        fp f_kpt0 {kpt0, {}, 0.0};
        fp f_kpt1 {kpt1, {}, 0.0};
        matched_fp m_fp = {std::make_pair(f_kpt0, f_kpt1), match_score};
        matched_fpts.push_back(m_fp);
    }

    return matched_fpts;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::Impl::init_trt(const toml::value &config) {
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
        LOG(ERROR) << "superpoint lightglue fp match model file: " << _m_trt_params.model_file_path << " not exist";
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
    _m_trt_params.execution_context = _m_trt_params.engine->createExecutionContext();
    if (nullptr == _m_trt_params.execution_context) {
        LOG(ERROR) << "create trt engine failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind input image tensors
    std::string input_node_name = "image0";
    auto successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, input_node_name, _m_trt_params.src_input_image_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    input_node_name = "image1";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, input_node_name, _m_trt_params.dst_input_image_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind output tensor
    std::string output_node_name = "kpts0";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.out_kpts0_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    output_node_name = "kpts1";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.out_kpts1_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    output_node_name = "matches0";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.out_matches_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    output_node_name = "mscores0";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.out_match_scores_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // set specified allocator for dynamically shaped output
    _m_trt_params.match_output_allocator = new lightglue_impl::MatchResultOutputAllocator;
    _m_trt_params.execution_context->setOutputAllocator("matches0", _m_trt_params.match_output_allocator);
    _m_trt_params.execution_context->setOutputAllocator("mscores0", _m_trt_params.match_output_allocator);

    // init cuda stream
    if (cudaStreamCreate(&_m_trt_params.cuda_stream) != cudaSuccess) {
        LOG(ERROR) << "ERROR: cuda stream creation failed.";
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
StatusCode LightGlue<INPUT, OUTPUT>::Impl::trt_run(const INPUT &in, OUTPUT &out) {
    // init sess
    auto& context = _m_trt_params.execution_context;
    auto& device_memory = _m_trt_params.device_memory;
    auto& src_input_binding = _m_trt_params.src_input_image_binding;
    auto& dst_input_binding = _m_trt_params.dst_input_image_binding;
    auto cuda_stream = _m_trt_params.cuda_stream;

    // transform external input into internal input
    auto internal_in = lightglue_impl::transform_input(in);
    if (!internal_in.src_input_image.data || internal_in.src_input_image.empty() ||
        !internal_in.dst_input_image.data || internal_in.dst_input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_src_input_size_user = internal_in.src_input_image.size();
    cv::Mat src_preprocessed_image = preprocess_image(internal_in.src_input_image);
    auto src_input_chw_data = CvUtils::convert_to_chw_vec(src_preprocessed_image);
    _m_dst_input_size_user = internal_in.dst_input_image.size();
    cv::Mat dst_preprocessed_image = preprocess_image(internal_in.dst_input_image);
    auto dst_input_chw_data = CvUtils::convert_to_chw_vec(dst_preprocessed_image);

    // setup trt bindings
    nvinfer1::Dims4 src_input_shape(1, 1, _m_src_input_size_user.height, _m_src_input_size_user.width);
    src_input_binding.set_dims(src_input_shape);
    context->setInputShape("image0", src_input_shape);
    nvinfer1::Dims4 dst_input_shape(1, 1, _m_dst_input_size_user.height, _m_dst_input_size_user.width);
    dst_input_binding.set_dims(dst_input_shape);
    context->setInputShape("image1", dst_input_shape);

    // allocate device memory
    auto status = setup_device_memory();
    if (status != StatusCode::OK) {
        LOG(ERROR) << "setup cuda memory for inference failed status code: " << status;
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // copy input image memo
    auto* cuda_mem_input = (float*)device_memory["image0"];
    auto input_mem_size = static_cast<int32_t >(src_input_chw_data.size() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
        cuda_mem_input, (float*)src_input_chw_data.data(), input_mem_size, cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cuda_mem_input = (float*)device_memory["image1"];
    input_mem_size = static_cast<int32_t >(dst_input_chw_data.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        cuda_mem_input, (float*)dst_input_chw_data.data(), input_mem_size, cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    std::vector<const char*> tensor_names = {"image0", "image1", "kpts0", "kpts1", "matches0", "mscores0"};
    for (auto& tensor_name : tensor_names) {
        if (!context->setTensorAddress(tensor_name, device_memory[tensor_name])) {
            LOG(ERROR) << "set addr for tensor: " << tensor_name << "failed";
        }
    }
    if (!context->enqueueV3(cuda_stream)) {
        LOG(ERROR) << "excute input data for inference failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cudaStreamSynchronize(cuda_stream);

    // transform internal output
    // todo fix tensorrt engine convention problem
    lightglue_impl::internal_output internal_out;
    out = lightglue_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::MODEL_RUN_SESSION_FAILED;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::Impl::setup_device_memory() {
    // init global params
    auto& context = _m_trt_params.execution_context;
    auto& device_memory = _m_trt_params.device_memory;
    auto& allocator_map = _m_trt_params.allocator_map;
    auto& engine = _m_trt_params.engine;

    // clear device memory
    for (auto& node_memo : device_memory) {
        auto& node_ptr = node_memo.second;
        if (nullptr != node_ptr) {
            cudaFree(node_ptr);
            node_ptr = nullptr;
        }
    }

    // init input node memory
    for (auto& node_name : {"image0", "image1"}) {
        auto shape = context->getTensorShape(node_name);
        LOG(INFO) << "node: " << node_name << ", shape: " << TrtHelper::dims_to_string(shape);
        LOG(INFO) << "node: " << node_name << ", tensor location: " << static_cast<int>(engine->getTensorLocation(node_name));
        auto volume = TrtHelper::dims_volume(shape);
        void* cuda_memo = nullptr;
        auto r = cudaMalloc(&cuda_memo, volume * sizeof(float));
        if (r != 0 || cuda_memo == nullptr) {
            LOG(ERROR) << "Setup device memory failed error str: " << cudaGetErrorString(r);
            return StatusCode::TRT_ALLOC_MEMO_FAILED;
        }
        device_memory.insert(std::make_pair(node_name, cuda_memo));
    }

    // init output node memory
    std::vector<std::string> output_names = {"kpts0", "kpts1", "matches0", "mscores0"};
    for (auto& node_name : output_names) {
        auto dims = context->getTensorShape(node_name.c_str());
        LOG(INFO) << "node: " << node_name << ", shape: " << TrtHelper::dims_to_string(dims);
        LOG(INFO) << "node: " << node_name << ", tensor location: " << static_cast<int>(engine->getTensorLocation(node_name.c_str()));
        bool has_dynamic_shape = false;
        for (auto& dim : dims.d) {
            if (dim == -1) {
                has_dynamic_shape = true;
            }
        }
        if (has_dynamic_shape) {
            LOG(INFO) << "set specific allocator";
            auto* allocator = new lightglue_impl::MatchResultOutputAllocator;
            context->setOutputAllocator(node_name.c_str(), allocator);
            allocator_map.emplace(node_name, allocator);
            void* cuda_memo;
            device_memory.insert(std::make_pair(node_name, cuda_memo));
        } else {
            auto volume = TrtHelper::dims_volume(dims);
            void* cuda_memo = nullptr;
            auto r = cudaMalloc(&cuda_memo, volume * sizeof(int64_t ));
            if (r != 0 || cuda_memo == nullptr) {
                LOG(ERROR) << "Setup device memory failed error str: " << cudaGetErrorString(r);
                return StatusCode::TRT_ALLOC_MEMO_FAILED;
            }
            device_memory.insert(std::make_pair(node_name, cuda_memo));
        }
    }
    return StatusCode::OJBK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> LightGlue<INPUT, OUTPUT>::LightGlue() {
    _m_pimpl = std::make_unique<Impl>(); 
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> LightGlue<INPUT, OUTPUT>::~LightGlue() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT> StatusCode LightGlue<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT> bool LightGlue<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode LightGlue<INPUT, OUTPUT>::run(const INPUT &input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace feature_point
} // namespace models
} // namespace jinq