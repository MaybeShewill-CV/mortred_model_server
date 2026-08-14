/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: msocrnet.cpp
* Date: 23-3-11
************************************************/

#include "msocrnet.h"
#include "models/cv_image_input.h"
#include "models/cv_image_input.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"
#include "onnxruntime/onnxruntime_cxx_api.h"

#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "common/base64.h"

namespace jinq {
namespace models {

using jinq::common::cv_utils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::base64;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;
using jinq::common::Timestamp;

namespace scene_segmentation {
using jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;

namespace msocrnet_impl {

using internal_input = mat_input;
using internal_output = std_scene_segmentation_output;

/***
*
* @tparam INPUT
* @param in
* @return
 */
template<typename INPUT>
internal_input transform_input(const INPUT& in) {
    internal_input result{};
    result.input_image = jinq::models::cv_input::load_image(in);
    return result;
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
transform_output(const msocrnet_impl::internal_output& internal_out) {
    return internal_out;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class MsOcrNet<INPUT, OUTPUT>::Impl {
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
    StatusCode init(const toml::table& config);

    /***
     *
     * @param config
     * @return
     */
    StatusCode init_onnx(const toml::table& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT& in, OUTPUT& out);

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
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  private:
    enum BackendType {
        MNN = 0,
        ONNX = 1,
    };

    struct ONNXParams {
        std::string model_file_path;
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, ""};
        Ort::SessionOptions session_options;
        Ort::Session* session = nullptr;
        Ort::AllocatorWithDefaultOptions allocator;
        int thread_nums = 1;
        std::string device = "cpu";
        int device_id = 0;
        std::vector<const char*> input_node_names;
        std::vector<std::vector<int64_t>> input_node_shapes;
        std::vector<const char*> output_node_names;
        std::vector<std::vector<int64_t>> output_node_shapes;
    };

    BackendType _m_backend_type = MNN;
    ONNXParams _m_onnx_params;
    // model file path
    std::string _m_model_file_path;
    // MNN Interpreter
    MNN::Interpreter* _m_net = nullptr;
    // MNN Session
    MNN::Session* _m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor* _m_input_tensor = nullptr;
    // MNN Output tensor node
    MNN::Tensor* _m_output_tensor = nullptr;
    // MNN threads nums
    uint _m_threads_nums = 4;
    // user input size
    cv::Size _m_input_size_user = cv::Size();
    //　input node size
    cv::Size _m_input_size_host = cv::Size();
    // flag
    bool _m_successfully_initialized = false;

  private:
    /***
     * preprocess image
     * @param input_image : input image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;
};

/***
*
* @param cfg_file_path
* @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode MsOcrNet<INPUT, OUTPUT>::Impl::init(const toml::table& config) {
    if (!config.contains("MSOCRNET")) {
        LOG(ERROR) << "Config missing MSOCRNET section please check config file";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* cfg_content_ptr = config["MSOCRNET"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section MSOCRNET missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    // backend type dispatch
    if (config.contains("BACKEND_DICT") && cfg_content.contains("backend_type")) {
        auto backend_dict = config["BACKEND_DICT"];
        auto backend_name = cfg_content["backend_type"].value_or<std::string>("");
        _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].value_or<int64_t>(0));
    }
    if (_m_backend_type == ONNX) {
        const toml::table* onnx_cfg_ptr = config["MSOCRNET_ONNX"].as_table();
        if (onnx_cfg_ptr == nullptr) {
            LOG(ERROR) << "Config section MSOCRNET_ONNX missing or not a table";
            _m_successfully_initialized = false;
            return StatusCode::MODEL_INIT_FAILED;
        }
        auto onnx_status = init_onnx(*onnx_cfg_ptr);
        _m_successfully_initialized = (onnx_status == StatusCode::OK);
        return onnx_status;
    }

    // init model threads nums
    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config file doesn\'t contain model_threads_num field, using default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = cfg_content["model_threads_num"].value_or<int64_t>(0);
    }

    // init interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config file doesn\'t contain model_file_path field, please check again";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content["model_file_path"].value_or<std::string>("");
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "MsOcrNet model file path: " << _m_model_file_path << ", not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = MNN::Interpreter::createFromFile(_m_model_file_path.c_str());
    if (nullptr == _m_net) {
        LOG(ERROR) << "Create MsOcrNet Interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init session
    MNN::ScheduleConfig mnn_config;
    if (!cfg_content.contains("compute_backend")) {
        LOG(WARNING) << "Config doesn\'t contain compute_backend field, using default backend cpu";
        LOG(INFO) << "Using CPU compute backend...";
        mnn_config.type = MNN_FORWARD_CPU;
    } else {
        std::string compute_backend = cfg_content["compute_backend"].value_or<std::string>("");
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
    if (!cfg_content.contains("backend_precision_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_precision_mode field default Precision_Normal";
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(cfg_content["backend_precision_mode"].value_or<int64_t>(0));
    }
    if (!cfg_content.contains("backend_power_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_power_mode field default Power_Normal";
        backend_config.power = MNN::BackendConfig::Power_Normal;
    } else {
        backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(cfg_content["backend_power_mode"].value_or<int64_t>(0));
    }
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);

    if (nullptr == _m_session) {
        LOG(ERROR) << "Create MsOcrNet Session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init input_tensor/output_tensor
    _m_input_tensor = _m_net->getSessionInput(_m_session, "x");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "argmax_0.tmp_0");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch MsOcrNet Input Node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch MsOcrNet Output Node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

    if (!cfg_content.contains("model_input_image_size")) {
        LOG(WARNING) << "Config doesn\'t contain model_input_image_size filed, using default value [1024, 512]";
        _m_input_size_user.width = 2048;
        _m_input_size_user.height = 1024;
    } else {
        _m_input_size_user.width = static_cast<int>(
            cfg_content["model_input_image_size"][1].value_or<int64_t>(0));
        _m_input_size_user.height = static_cast<int>(
            cfg_content["model_input_image_size"][0].value_or<int64_t>(0));
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "MSOCRNet detection model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
    return StatusCode::OK;
}

/***
*
* @param config
* @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode MsOcrNet<INPUT, OUTPUT>::Impl::init_onnx(const toml::table& config) {
    // init onnx runtime configs
    _m_onnx_params.model_file_path = config["model_file_path"].value_or<std::string>("");
    if (!FilePathUtil::is_file_exist(_m_onnx_params.model_file_path)) {
        LOG(ERROR) << "MsOcrNet onnx model file path: " << _m_onnx_params.model_file_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_onnx_params.device = config["compute_backend"].value_or<std::string>("");
    if (std::strcmp(_m_onnx_params.device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_onnx_params.device_id = config["gpu_device_id"].value_or<int64_t>(0);
    }
    _m_onnx_params.thread_nums = config["model_threads_num"].value_or<int64_t>(0);
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
    _m_onnx_params.session = new Ort::Session(
        _m_onnx_params.env, _m_onnx_params.model_file_path.c_str(), _m_onnx_params.session_options);

    // init input/output node info
    auto input_nodes_counts = _m_onnx_params.session->GetInputCount();
    for (size_t i = 0; i < input_nodes_counts; ++i) {
        auto input_node_name = strdup(_m_onnx_params.session->GetInputNameAllocated(i, _m_onnx_params.allocator).get());
        auto input_node_shape = _m_onnx_params.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        _m_onnx_params.input_node_names.push_back(std::move(input_node_name));
        _m_onnx_params.input_node_shapes.push_back(input_node_shape);
    }
    if (_m_onnx_params.input_node_shapes[0].size() >= 4 &&
        _m_onnx_params.input_node_shapes[0][2] > 0) {
        _m_input_size_host.height = static_cast<int>(_m_onnx_params.input_node_shapes[0][2]);
        _m_input_size_host.width = static_cast<int>(_m_onnx_params.input_node_shapes[0][3]);
    } else if (config.contains("model_input_image_size")) {
        _m_input_size_host.height = static_cast<int>(
            config["model_input_image_size"][0].value_or<int64_t>(0));
        _m_input_size_host.width = static_cast<int>(
            config["model_input_image_size"][1].value_or<int64_t>(0));
    } else {
        LOG(ERROR) << "dynamic onnx input requires model_input_image_size field";
        return StatusCode::MODEL_INIT_FAILED;
    }

    auto output_nodes_counts = _m_onnx_params.session->GetOutputCount();
    for (size_t i = 0; i < output_nodes_counts; ++i) {
        auto output_node_name = strdup(_m_onnx_params.session->GetOutputNameAllocated(i, _m_onnx_params.allocator).get());
        auto output_node_shape = _m_onnx_params.session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        _m_onnx_params.output_node_names.push_back(std::move(output_node_name));
        _m_onnx_params.output_node_shapes.push_back(output_node_shape);
    }

    return StatusCode::OK;
}

/***
*
* @param in
* @param out
* @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode MsOcrNet<INPUT, OUTPUT>::Impl::onnx_run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = msocrnet_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_image_chw_data = cv_utils::convert_to_chw_vec(preprocessed_image);

    // prepare input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, _m_input_size_host.height, _m_input_size_host.width};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_image_chw_data.data(), input_image_chw_data.size(),
        input_shape.data(), input_shape.size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    // run session
    auto output_tensors = _m_onnx_params.session->Run(
        Ort::RunOptions{nullptr}, _m_onnx_params.input_node_names.data(), input_tensors.data(),
        input_tensors.size(), _m_onnx_params.output_node_names.data(), _m_onnx_params.output_node_names.size());

    // copy output tensor values (int64 argmax) into int32 mask
    auto& out_tensor = output_tensors[0];
    auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto* out_data = out_tensor.template GetTensorMutableData<int64_t>();
    auto out_counts = static_cast<int>(out_shape[1] * out_shape[2]);
    cv::Mat seg_mask(_m_input_size_host, CV_32SC1, cv::Scalar(0));
    for (auto idx = 0; idx < out_counts; ++idx) {
        seg_mask.at<int32_t>(idx) = static_cast<int32_t>(out_data[idx]);
    }
    cv::Mat resized_mask;
    cv::resize(seg_mask, resized_mask, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);

    // transform internal output into external output
    msocrnet_impl::internal_output internal_out;
    internal_out.segmentation_result = resized_mask.clone();
    out = msocrnet_impl::transform_output<OUTPUT>(internal_out);

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
cv::Mat MsOcrNet<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
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
StatusCode MsOcrNet<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    if (_m_backend_type == ONNX) {
        return onnx_run(in, out);
    }

    // transform external input into internal input
    auto internal_in = msocrnet_impl::transform_input(in);

    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    // run session
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!cv_utils::copy_image_to_tensor(input_tensor_data, preprocessed_image, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);
    // fetch net output
    MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    _m_output_tensor->copyToHostTensor(&output_tensor_user);
    auto host_data = output_tensor_user.host<int>();
    cv::Mat result_image(_m_input_size_host, CV_32SC1, host_data);
    cv::resize(result_image, result_image, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);

    // transform internal output into external output
    msocrnet_impl::internal_output internal_out;
    // clone the result to avoid referencing the MNN host tensor memory which
    // will be released when output_tensor_user is destructed
    internal_out.segmentation_result = result_image.clone();
    out = msocrnet_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
MsOcrNet<INPUT, OUTPUT>::MsOcrNet() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
MsOcrNet<INPUT, OUTPUT>::~MsOcrNet() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode MsOcrNet<INPUT, OUTPUT>::init(const toml::table& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool MsOcrNet<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode MsOcrNet<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}
