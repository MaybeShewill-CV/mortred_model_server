/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: metric3d.inl
* Date: 23-10-26
************************************************/

#include "metric3d.h"

#include <random>

#include "MNN/Interpreter.hpp"
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

namespace metric3d_impl {

struct internal_input {
    cv::Mat input_image;
};

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
template <typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_mde_output>::type>::value, std_mde_output>::type
transform_output(const metric3d_impl::internal_output& internal_out) {
    std_mde_output result;
    internal_out.depth_map.copyTo(result.depth_map);
    return result;
}

} // namespace metric3d_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT>
class Metric3D<INPUT, OUTPUT>::Impl {
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
            auto status = cudaStreamDestroy(_m_trt_params.cuda_stream);
            if (status != cudaSuccess) {
                LOG(ERROR) << "Failed to free metric3d trt object. Destruct cuda stream "
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
    struct MNNParams {
        std::string model_file_path;
        MNN::Interpreter* net = nullptr;
        MNN::Session* session = nullptr;
        MNN::Tensor* input_tensor = nullptr;
        MNN::Tensor* confidence_output_tensor = nullptr;
        MNN::Tensor* preds_depth_output_tensor = nullptr;
        uint threads_nums = 4;
    };

    struct TRTParams {
        // model file path
        std::string model_file_path;
        // trt context
        std::unique_ptr<nvinfer1::IExecutionContext> execution_context;
        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<TrtLogger> logger;
        // trt bindings
        EngineBinding input_image_binding;
        EngineBinding output_depth_binding;
        EngineBinding output_confidence_binding;
        // trt memory
        DeviceMemory device_memory;
        cudaStream_t cuda_stream = nullptr;
        // host memory
        std::vector<float> output_depth_host;
        std::vector<float> output_confidence_host;
    };

    enum BackendType {
        MNN = 0,
        TRT = 1,
    };

private:
    // model backend type
    BackendType _m_backend_type = MNN;

    // mnn net params
    MNNParams _m_mnn_params;

    // trt net params
    TRTParams _m_trt_params;

    // input image size
    cv::Size _m_input_size_user = cv::Size();
    //　input node size
    cv::Size _m_input_size_host = cv::Size();
    // canonical size
    cv::Size _m_canonical_size = cv::Size();
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
    StatusCode init_mnn(const toml::value& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode mnn_run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @return
     */
    metric3d_impl::internal_output mnn_decode_output();

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
    metric3d_impl::internal_output trt_decode_output();

    /***
     *
     * @return
     */
    float calculate_label_scale_factor() {
        auto ori_focal = (_m_intrinsic_params[0] + _m_intrinsic_params[1]) / 2.0f;
        auto canonical_focal = _m_focal_length;
        auto src_w = _m_input_size_user.width;
        auto src_h = _m_input_size_user.height;
        auto resize_ratio_h = static_cast<float>(_m_input_size_host.height) / static_cast<float>(src_h);
        auto resize_ratio_w = static_cast<float>(_m_input_size_host.width) / static_cast<float>(src_w);
        auto to_scale_ratio = std::min(resize_ratio_h, resize_ratio_w);
        auto resize_label_scale_factor = 1.0f / to_scale_ratio;
        auto cano_label_scale_ratio = canonical_focal / ori_focal;
        auto label_scale_factor = cano_label_scale_ratio * resize_label_scale_factor;

        return label_scale_factor;
    }

    /***
     *
     * @param pad_h
     * @param pad_w
     */
    void calculate_pad_info(int& pad_h, int& pad_w) {
        auto src_w = _m_input_size_user.width;
        auto src_h = _m_input_size_user.height;
        auto resize_ratio_h = static_cast<float>(_m_input_size_host.height) / static_cast<float>(src_h);
        auto resize_ratio_w = static_cast<float>(_m_input_size_host.width) / static_cast<float>(src_w);
        auto to_scale_ratio = std::min(resize_ratio_h, resize_ratio_w);
        auto resize_ratio = 1.0f * to_scale_ratio;
        auto reshape_h = static_cast<int>(resize_ratio * static_cast<float>(src_h));
        auto reshape_w = static_cast<int>(resize_ratio * static_cast<float>(src_w));
        pad_h = std::max(_m_input_size_host.height - reshape_h, 0);
        pad_w = std::max(_m_input_size_host.width - reshape_w, 0);
    }

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

/***
*
* @param cfg_file_path
* @return
*/
template <typename INPUT, typename OUTPUT>
StatusCode Metric3D<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("METRIC3D").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init metric3d configs
    toml::value metric3d_cfg;
    if (_m_backend_type == MNN) {
        metric3d_cfg = config.at("METRIC3D_MNN");
    } else {
        metric3d_cfg = config.at("METRIC3D_TRT");
    }
    auto model_file_name = FilePathUtil::get_file_name(metric3d_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == MNN) {
        init_status = init_mnn(metric3d_cfg);
    } else {
        init_status = init_trt(metric3d_cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << "Successfully load metric3d model from: " << model_file_name;
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load metric3d model from: " << model_file_name;
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
StatusCode Metric3D<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    StatusCode infer_status;
    if (_m_backend_type == MNN) {
        infer_status = mnn_run(in, out);
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
cv::Mat Metric3D<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) {
    cv::Mat tmp;
    // bgr to rgb
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);

    // convert to float32
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    // resize image
    auto src_w = _m_input_size_user.width;
    auto src_h = _m_input_size_user.height;
    auto resize_ratio_h = static_cast<float>(_m_input_size_host.height) / static_cast<float>(src_h);
    auto resize_ratio_w = static_cast<float>(_m_input_size_host.width) / static_cast<float>(src_w);
    auto to_scale_ratio = std::min(resize_ratio_h, resize_ratio_w);
    auto resize_ratio = 1.0f * to_scale_ratio;
    auto reshape_h = static_cast<int>(resize_ratio * static_cast<float>(src_h));
    auto reshape_w = static_cast<int>(resize_ratio * static_cast<float>(src_w));
    auto pad_h = std::max(_m_input_size_host.height - reshape_h, 0);
    auto pad_w = std::max(_m_input_size_host.width - reshape_w, 0);
    auto pad_h_half = static_cast<int>(pad_h / 2);
    auto pad_w_half = static_cast<int>(pad_w / 2);

    cv::resize(tmp, tmp, cv::Size(reshape_w, reshape_h));
    cv::copyMakeBorder(
        tmp, tmp, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
        cv::BORDER_CONSTANT, cv::Scalar(123.675, 116.28, 103.53));

    // subtract mean
    cv::subtract(tmp, cv::Scalar(123.675, 116.28, 103.53), tmp);

    // div std
    cv::divide(tmp, cv::Scalar(58.395, 57.12, 57.375), tmp);

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
StatusCode Metric3D<INPUT, OUTPUT>::Impl::init_mnn(const toml::value& config) {
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
        LOG(ERROR) << "Create metric3d model interpreter failed";
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

    _m_mnn_params.input_tensor = _m_mnn_params.net->getSessionInput(_m_mnn_params.session, "input_image");
    _m_mnn_params.confidence_output_tensor = _m_mnn_params.net->getSessionOutput(_m_mnn_params.session, "confidence");
    _m_mnn_params.preds_depth_output_tensor = _m_mnn_params.net->getSessionOutput(_m_mnn_params.session, "prediction");
    if (nullptr == _m_mnn_params.input_tensor) {
        LOG(ERROR) << "fetch metric3d model input node failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (nullptr == _m_mnn_params.confidence_output_tensor || nullptr == _m_mnn_params.preds_depth_output_tensor) {
        LOG(ERROR) << "fetch metric3d model output node failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init hose size and user size
    _m_input_size_host.width = _m_mnn_params.input_tensor->width();
    _m_input_size_host.height = _m_mnn_params.input_tensor->height();

    // init intrinsic and canonical size
    _m_focal_length = static_cast<float>(config.at("focal_length").as_floating());
    _m_canonical_size.width = static_cast<int>(config.at("canonical_size").as_array()[1].as_integer());
    _m_canonical_size.height = static_cast<int>(config.at("canonical_size").as_array()[0].as_integer());
    _m_intrinsic_params = {
        static_cast<float>(config.at("intrinsic").as_array()[0].as_floating()),
        static_cast<float>(config.at("intrinsic").as_array()[1].as_floating()),
        static_cast<float>(config.at("intrinsic").as_array()[2].as_floating()),
        static_cast<float>(config.at("intrinsic").as_array()[3].as_floating()),
    };

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode Metric3D<INPUT, OUTPUT>::Impl::mnn_run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = metric3d_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    auto& net = _m_mnn_params.net;
    auto& session = _m_mnn_params.session;
    auto& input_tensor = _m_mnn_params.input_tensor;

    // preprocess
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run session
    MNN::Tensor input_tensor_user(input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, input_chw_image_data.data(), input_tensor_size);
    input_tensor->copyFromHostTensor(&input_tensor_user);
    net->runSession(session);

    // decode output tensor
    auto depth_out = mnn_decode_output();

    // transform internal output into external output
    out = metric3d_impl::transform_output<OUTPUT>(depth_out);
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template <typename INPUT, typename OUTPUT>
metric3d_impl::internal_output Metric3D<INPUT, OUTPUT>::Impl::mnn_decode_output() {
    // fetch output value
    auto& confidence_out_tensor_host = _m_mnn_params.confidence_output_tensor;
    auto& depth_out_tensor_host = _m_mnn_params.preds_depth_output_tensor;
    MNN::Tensor confidence_out_tensor_user(confidence_out_tensor_host, MNN::Tensor::DimensionType::CAFFE);
    confidence_out_tensor_host->copyToHostTensor(&confidence_out_tensor_user);
    MNN::Tensor depth_out_tensor_user(depth_out_tensor_host, MNN::Tensor::DimensionType::CAFFE);
    depth_out_tensor_host->copyToHostTensor(&depth_out_tensor_user);

    // make depth and confidence map
    cv::Mat depth_map = cv::Mat::zeros(_m_input_size_host, CV_32FC1);
    cv::Mat confidence_map = cv::Mat::zeros(_m_input_size_host, CV_32FC1);
    for (auto row = 0; row < _m_input_size_host.height; ++row) {
        auto depth_row_data = depth_map.ptr<float>(row);
        auto conf_row_data = confidence_map.ptr<float>(row);
        for (auto col = 0; col < _m_input_size_host.width; ++col) {
            auto idx = 1 * 1 * row * _m_input_size_host.width + col;
            depth_row_data[col] = depth_out_tensor_user.host<float>()[idx] < 0 ? 0 : depth_out_tensor_user.host<float>()[idx];
            conf_row_data[col] = confidence_out_tensor_user.host<float>()[idx];
        }
    }

    // crop pad info
    int pad_h = 0;
    int pad_w = 0;
    calculate_pad_info(pad_h, pad_w);
    auto crop_roi = cv::Rect(
        static_cast<int>(pad_w / 2), static_cast<int>(pad_h / 2),depth_map.cols - pad_w, depth_map.rows - pad_h);
    crop_roi = crop_roi & cv::Rect(0, 0, depth_map.cols, depth_map.rows);
    depth_map(crop_roi).copyTo(depth_map);
    confidence_map(crop_roi).copyTo(confidence_map);

    // rescale into input image size
    cv::resize(depth_map, depth_map, _m_input_size_user);
    cv::resize(confidence_map, confidence_map, _m_input_size_user);

    // rescale depth value
    auto label_scale_factor = calculate_label_scale_factor();
    cv::divide(depth_map, label_scale_factor, depth_map);

    // copy result
    std_mde_output out;
    depth_map.copyTo(out.depth_map);
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
StatusCode Metric3D<INPUT, OUTPUT>::Impl::init_trt(const toml::value& cfg) {
    // init trt runtime
    _m_trt_params.logger = std::make_unique<TrtLogger>();
    auto* trt_runtime = nvinfer1::createInferRuntime(*_m_trt_params.logger);
    if(trt_runtime == nullptr) {
        LOG(ERROR) << "init tensorrt runtime failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_trt_params.runtime = std::unique_ptr<nvinfer1::IRuntime>(trt_runtime);

    // init trt engine
    if (!cfg.contains("model_file_path")) {
        LOG(ERROR) << "config doesn\'t have model_file_path field";
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_trt_params.model_file_path = cfg.at("model_file_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_trt_params.model_file_path)) {
        LOG(ERROR) << "metric3d trt estimation model file: " << _m_trt_params.model_file_path << " not exist";
        return StatusCode::MODEL_INIT_FAILED;
    }
    std::vector<unsigned char> model_file_content;
    if (!read_model_file(_m_trt_params.model_file_path, model_file_content)) {
        LOG(ERROR) << "read model file: " << _m_trt_params.model_file_path << " failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    auto model_content_length = sizeof(model_file_content[0]) * model_file_content.size();
    _m_trt_params.engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        _m_trt_params.runtime->deserializeCudaEngine(model_file_content.data(), model_content_length));
    if (_m_trt_params.engine == nullptr) {
        LOG(ERROR) << "deserialize trt engine failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init trt execution context
    _m_trt_params.execution_context = std::unique_ptr<nvinfer1::IExecutionContext>(_m_trt_params.engine->createExecutionContext());
    if (_m_trt_params.execution_context == nullptr) {
        LOG(ERROR) << "create trt engine failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind input tensor
    std::string input_node_name = "input_image";
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
    if (_m_trt_params.input_image_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = _m_trt_params.input_image_binding.dims().d[2];
    _m_input_size_host.width = _m_trt_params.input_image_binding.dims().d[3];

    // bind output tensor
    std::string output_node_name = "prediction";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.output_depth_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind predicted depth output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.output_depth_binding.dims().nbDims != 4) {
        std::string output_shape_str = TrtHelper::dims_to_string(_m_trt_params.output_depth_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.output_depth_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic output tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    output_node_name = "confidence";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_params.engine, output_node_name, _m_trt_params.output_confidence_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind predicted confidence output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.output_confidence_binding.dims().nbDims != 4) {
        std::string output_shape_str = TrtHelper::dims_to_string(_m_trt_params.output_confidence_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.output_confidence_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic output tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup device memory
    auto set_device_memo_status = TrtHelper::setup_device_memory(
        _m_trt_params.engine, _m_trt_params.execution_context, _m_trt_params.device_memory);
    if (set_device_memo_status != StatusCode::OK) {
        LOG(ERROR) << "setup device memory for model failed, status code: " << set_device_memo_status;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init cuda stream
    if (cudaStreamCreate(&_m_trt_params.cuda_stream) != cudaSuccess) {
        LOG(ERROR) << "ERROR: cuda stream creation failed." << std::endl;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // allocate output host tensor memo
    _m_trt_params.output_depth_host.resize(_m_trt_params.output_depth_binding.volume());
    _m_trt_params.output_confidence_host.resize(_m_trt_params.output_confidence_binding.volume());

    // init intrinsic and canonical size
    _m_focal_length = static_cast<float>(cfg.at("focal_length").as_floating());
    _m_canonical_size.width = static_cast<int>(cfg.at("canonical_size").as_array()[1].as_integer());
    _m_canonical_size.height = static_cast<int>(cfg.at("canonical_size").as_array()[0].as_integer());
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
StatusCode Metric3D<INPUT, OUTPUT>::Impl::trt_run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = metric3d_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess input data
    _m_input_size_user = internal_in.input_image.size();
    auto preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // copy input data from host to device
    auto* cuda_mem_input = (float*)_m_trt_params.device_memory.at(_m_trt_params.input_image_binding.index());
    auto input_mem_size = static_cast<int32_t >(preprocessed_image.channels() * preprocessed_image.size().area() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
        cuda_mem_input, (float*)input_chw_data.data(), input_mem_size, cudaMemcpyHostToDevice, _m_trt_params.cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    _m_trt_params.execution_context->setTensorAddress("input_image", cuda_mem_input);
    _m_trt_params.execution_context->setTensorAddress(
        "confidence", _m_trt_params.device_memory.at(_m_trt_params.output_confidence_binding.index()));
    _m_trt_params.execution_context->setTensorAddress(
        "prediction", _m_trt_params.device_memory.at(_m_trt_params.output_depth_binding.index()));
    if (!_m_trt_params.execution_context->enqueueV3(_m_trt_params.cuda_stream)) {
        LOG(ERROR) << "execute input data for inference failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // async copy inference result back to host
    cuda_status = cudaMemcpyAsync(_m_trt_params.output_confidence_host.data(),
                                  _m_trt_params.device_memory.at(_m_trt_params.output_confidence_binding.index()),
                                  (int)(_m_trt_params.output_confidence_binding.volume() * sizeof(float)),
                                  cudaMemcpyDeviceToHost, _m_trt_params.cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cuda_status = cudaMemcpyAsync(_m_trt_params.output_depth_host.data(),
                                  _m_trt_params.device_memory.at(_m_trt_params.output_depth_binding.index()),
                                  (int)(_m_trt_params.output_depth_binding.volume() * sizeof(float)),
                                  cudaMemcpyDeviceToHost, _m_trt_params.cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cudaStreamSynchronize(_m_trt_params.cuda_stream);

    // decode output
    auto depth_out = trt_decode_output();

    // transform internal output into external output
    out = metric3d_impl::transform_output<OUTPUT>(depth_out);
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT>
metric3d_impl::internal_output Metric3D<INPUT, OUTPUT>::Impl::trt_decode_output() {
    // make depth and confidence map
    cv::Mat depth_map = cv::Mat::zeros(_m_input_size_host, CV_32FC1);
    cv::Mat confidence_map = cv::Mat::zeros(_m_input_size_host, CV_32FC1);
    for (auto row = 0; row < _m_input_size_host.height; ++row) {
        auto depth_row_data = depth_map.ptr<float>(row);
        auto conf_row_data = confidence_map.ptr<float>(row);
        for (auto col = 0; col < _m_input_size_host.width; ++col) {
            auto idx = 1 * 1 * row * _m_input_size_host.width + col;
            depth_row_data[col] = _m_trt_params.output_depth_host[idx] < 0 ? 0 : _m_trt_params.output_depth_host[idx];
            conf_row_data[col] = _m_trt_params.output_confidence_host[idx];
        }
    }

    // crop pad info
    int pad_h = 0;
    int pad_w = 0;
    calculate_pad_info(pad_h, pad_w);
    auto crop_roi = cv::Rect(
        static_cast<int>(pad_w / 2), static_cast<int>(pad_h / 2),depth_map.cols - pad_w, depth_map.rows - pad_h);
    crop_roi = crop_roi & cv::Rect(0, 0, depth_map.cols, depth_map.rows);
    depth_map(crop_roi).copyTo(depth_map);
    confidence_map(crop_roi).copyTo(confidence_map);

    // rescale into input image size
    cv::resize(depth_map, depth_map, _m_input_size_user);
    cv::resize(confidence_map, confidence_map, _m_input_size_user);

    // rescale depth value
    auto label_scale_factor = calculate_label_scale_factor();
    cv::divide(depth_map, label_scale_factor, depth_map);

    // copy result
    std_mde_output out;
    depth_map.copyTo(out.depth_map);
    return out;
}

/************* Export Function Sets *************/

/***
*
* @tparam INPUT
* @tparam OUTPUT
*/
template <typename INPUT, typename OUTPUT>
Metric3D<INPUT, OUTPUT>::Metric3D() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
*/
template <typename INPUT, typename OUTPUT>
Metric3D<INPUT, OUTPUT>::~Metric3D() = default;

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @param cfg
* @return
*/
template <typename INPUT, typename OUTPUT>
StatusCode Metric3D<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
*
* @tparam INPUT
* @tparam OUTPUT
* @return
*/
template <typename INPUT, typename OUTPUT>
bool Metric3D<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode Metric3D<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace mono_depth_estimation
} // namespace models
} // namespace jinq
