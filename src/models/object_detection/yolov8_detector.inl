/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_detector.inl
 * Date: 24-3-13
 ************************************************/

#include "yolov8_detector.h"

#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "TensorRT-8.6.1.6/NvInferRuntime.h"

#include "common/time_stamp.h"
#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/trt_helper/trt_helper.h"

namespace jinq {
namespace models {

using jinq::common::Base64;
using jinq::common::CvUtils;
using jinq::common::Timestamp;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;

namespace object_detection {

using trt_helper::TrtHelper;
using trt_helper::TrtLogger;
using trt_helper::DeviceMemory;
using trt_helper::EngineBinding;
using jinq::models::io_define::object_detection::bbox;
using jinq::models::io_define::object_detection::std_object_detection_output;

namespace yolov8_impl {

using internal_input = mat_input;
using internal_output = std_object_detection_output;

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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_object_detection_output>::type>::value, std_object_detection_output>::type
transform_output(const yolov8_impl::internal_output& internal_out) {
    return internal_out;
}
}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class YoloV8Detector<INPUT, OUTPUT>::Impl {
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
            cudaStreamDestroy(_m_trt_params.cuda_stream);
            cudaFree(_m_trt_params.input_device);
            cudaFree(_m_trt_params.output_device);
            cudaFreeHost(_m_trt_params.output_host);
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
        // trt host/device memory
        float* output_host = nullptr;
        void* input_device = nullptr;
        void* output_device = nullptr;
    };

    enum BackendType {
        TRT = 0,
    };

  private:
    // model backend type
    BackendType _m_backend_type = TRT;

    // trt net params
    TRTParams _m_trt_params;

    // score thresh
    double _m_score_threshold = 0.4;
    // nms thresh
    double _m_nms_threshold = 0.35;
    // top_k keep thresh
    long _m_keep_topk = 250;
    // class nums
    int _m_class_nums = 80;
    // class id to names
    std::map<int, std::string> _m_class_id2names;
    // input image size
    cv::Size _m_input_size_user = cv::Size();
    //　input node size
    cv::Size _m_input_size_host = cv::Size();
    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     * preprocess
     * @param input_image : input image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @param input_image
     * @return
     */
    StatusCode maybe_reallocate_input_device_memory(const cv::Mat& input_image);

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
     * @tparam T
     * @param bbox
     * @return
     */
    cv::Rect2f transform_bboxes(const cv::Rect2d& bbox) {
        auto w_scale = static_cast<float>(_m_input_size_user.width) / static_cast<float>(_m_input_size_host.width);
        auto h_scale = static_cast<float>(_m_input_size_user.height) / static_cast<float>(_m_input_size_host.height);
        cv::Rect2f result;
        result.x = static_cast<float>(bbox.x * w_scale);
        result.y = static_cast<float>(bbox.y * h_scale);
        result.width = static_cast<float>(bbox.width * w_scale);
        result.height = static_cast<float>(bbox.height * h_scale);

        return result;
    }
};

/***
*
* @param cfg_file_path
* @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {

    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("YOLOV8").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init metric3d configs
    toml::value yolov8_cfg;
    if (_m_backend_type == TRT) {
        yolov8_cfg = config.at("YOLOV8_TRT");
    } else {
        // todo implment other backend
    }
    auto model_file_name = FilePathUtil::get_file_name(yolov8_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == TRT) {
        init_status = init_trt(yolov8_cfg);
    } else {
        // todo implment other backend
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << "Successfully load yolov8 model from: " << model_file_name;
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load yolov8 model from: " << model_file_name;
    }

    return init_status;
}

/***
*
* @param in
* @param out
* @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    StatusCode infer_status;
    if (_m_backend_type == TRT) {
        infer_status = trt_run(in, out);
    } else {
        // todo implment other backend
    }
    return infer_status;
}

/***
 *
 * @param input_image
 * @return
 */
template<typename INPUT, typename OUTPUT>
cv::Mat YoloV8Detector<INPUT, OUTPUT>::Impl::preprocess_image(
    const cv::Mat& input_image) const {
    // convert bgr 2 rgb
    cv::Mat tmp;
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);

    // resize image
    cv::resize(tmp, tmp, _m_input_size_host);

    // type cast
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    return tmp;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param input_image
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::Impl::maybe_reallocate_input_device_memory(const cv::Mat& input_image) {
    auto current_input_binding_volume = _m_trt_params.input_binding.volume();
    auto current_input_image_ele_size = input_image.rows * input_image.cols * input_image.channels();
    // no need to reallocate
    if (current_input_image_ele_size == current_input_binding_volume) {
        return StatusCode::OK;
    }
    // reallocate input device memory
    uint32_t bytes = current_input_image_ele_size * sizeof(uint8_t);
    if (nullptr != _m_trt_params.input_device) {
        cudaFree(_m_trt_params.input_device);
    }
    auto cuda_status = cudaMalloc(&_m_trt_params.input_device, bytes);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "reallocate input device memory failed, err str: " << cudaGetErrorString(cuda_status);
        return StatusCode::TRT_ALLOC_MEMO_FAILED;
    }
    // set new binding info
    _m_trt_params.input_binding.set_volume(current_input_image_ele_size);
    nvinfer1::Dims4 input_dims(1, input_image.rows, input_image.cols, input_image.channels());
    _m_trt_params.input_binding.set_dims(input_dims);

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::Impl::init_trt(const toml::value& cfg) {
    // init threshold
    if (!cfg.contains("model_score_threshold")) {
        _m_score_threshold = 0.4;
    } else {
        _m_score_threshold = cfg.at("model_score_threshold").as_floating();
    }

    if (!cfg.contains("model_nms_threshold")) {
        _m_nms_threshold = 0.35;
    } else {
        _m_nms_threshold = cfg.at("model_nms_threshold").as_floating();
    }

    if (!cfg.contains("model_keep_top_k")) {
        _m_keep_topk = 250;
    } else {
        _m_keep_topk = cfg.at("model_keep_top_k").as_integer();
    }

    if (!cfg.contains("model_class_nums")) {
        _m_class_nums = 80;
    } else {
        _m_class_nums = static_cast<int>(cfg.at("model_class_nums").as_integer());
    }

    if (!cfg.contains("class_names")) {
        for (auto idx = 0; idx < _m_class_nums; ++idx) {
            _m_class_id2names.insert(std::make_pair(idx, ""));
        }
    } else {
        auto cls_names = cfg.at("class_names").as_array();
        for (auto idx = 0; idx < cls_names.size(); ++idx) {
            _m_class_id2names.insert(std::make_pair(idx, cls_names[idx].as_string()));
        }
    }

    // init trt runtime
    _m_trt_params.logger = TrtLogger();
    _m_trt_params.runtime = nvinfer1::createInferRuntime(_m_trt_params.logger);
    if (nullptr == _m_trt_params.runtime) {
        LOG(ERROR) << "init tensorrt runtime failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init trt engine
    if (!cfg.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_trt_params.model_file_path = cfg.at("model_file_path").as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_trt_params.model_file_path)) {
        LOG(ERROR) << "Privacy trt detection model file: " << _m_trt_params.model_file_path << " not exist";
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
        LOG(ERROR) << "create trt context failed";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind input tensor
    std::string input_node_name = "images";
    auto successfully_bind = TrtHelper::setup_engine_binding(
        _m_trt_params.engine, input_node_name, _m_trt_params.input_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.input_binding.dims().nbDims != 4) {
        std::string input_shape_str = TrtHelper::dims_to_string(_m_trt_params.input_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind output tensor
    std::string output_node_name = "output0";
    successfully_bind = TrtHelper::setup_engine_binding(
        _m_trt_params.engine, output_node_name, _m_trt_params.output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_trt_params.output_binding.dims().nbDims != 3) {
        std::string output_shape_str = TrtHelper::dims_to_string(_m_trt_params.output_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C, H]";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup input host/device memory
    auto memo_size = _m_trt_params.input_binding.volume() * sizeof(uint8_t);
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

    // set input node size
    _m_input_size_host.height = static_cast<int>(cfg.at("input_node_size").as_array()[1].as_integer());
    _m_input_size_host.width = static_cast<int>(cfg.at("input_node_size").as_array()[0].as_integer());

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
StatusCode YoloV8Detector<INPUT, OUTPUT>::Impl::trt_run(const INPUT& in, OUTPUT& out) {
    // init envs
    auto& context = _m_trt_params.context;
    auto& cuda_stream = _m_trt_params.cuda_stream;
    auto& input_binding = _m_trt_params.input_binding;
    auto& output_binding = _m_trt_params.output_binding;
    auto& output_host = _m_trt_params.output_host;
    auto& input_device = _m_trt_params.input_device;
    auto& output_device = _m_trt_params.output_device;

    // transform external input into internal input
    auto internal_in = yolov8_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess input data
    cv::Mat& input_image = internal_in.input_image;
    _m_input_size_user = input_image.size();
    auto preprocessed_image = preprocess_image(input_image);
    auto input_chw_image_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // maybe reallocate device memory
    maybe_reallocate_input_device_memory(preprocessed_image);
    auto input_mem_size = input_binding.volume() * sizeof(uint8_t);

    // H2D data transfer
    auto cuda_status = cudaMemcpyAsync(
        input_device, (float*)input_chw_image_data.data(), input_mem_size, cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    context->setInputTensorAddress("images", input_device);
    context->setTensorAddress("output0", output_device);
    if (!context->enqueueV3(cuda_stream)) {
        LOG(ERROR) << "enqueue input data for inference failed";
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

    // decode output bboxes
    auto proposal_counts = output_binding.dims().d[2];
    auto row_size = output_binding.dims().d[1];
    std::vector<cv::Rect2d> bboxes;
    std::vector<float> scores;
    std::vector<int> cls_ids;

    auto begin = output_host;
    for (int i = 0; i < proposal_counts; ++i) {
        auto cls_nums = row_size - 4;
        float cx = begin[0 * proposal_counts + i];
        float cy = begin[1 * proposal_counts + i];
        float w = begin[2 * proposal_counts + i];
        float h = begin[3 * proposal_counts + i];

        std::vector<float> tmp_scores;
        for (auto j = 4; j < cls_nums; ++j) {
            float score = begin[j * proposal_counts + i];
            tmp_scores.push_back(score);
        }
        auto max_score = std::max_element(std::begin(tmp_scores), std::end(tmp_scores));
        if (*max_score < _m_score_threshold) {
            continue;
        } else {
            float cls_score = *max_score;
            int cls_id = static_cast<int>(std::distance(tmp_scores.begin(), max_score));
            float x = cx - w / 2.0f;
            float y = cy - h / 2.0f;

            bboxes.emplace_back(x, y, w, h);
            scores.push_back(cls_score);
            cls_ids.push_back(cls_id);
        }
    }

    // nms thresh
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, _m_score_threshold, _m_nms_threshold, indices);
    if (indices.size() >= _m_keep_topk) {
        indices.resize(_m_keep_topk);
    }

    // transform bboxes from network space to user space
    yolov8_impl::internal_output decode_result;
    for (auto& idx : indices) {
        float score = scores[idx];
        int cls_id = cls_ids[idx];
        cv::Rect2d ori_bbox = bboxes[idx];
        auto converted_bboxes = transform_bboxes(ori_bbox);
        bbox tmp_bbox {converted_bboxes, score, cls_id};
        decode_result.push_back(tmp_bbox);
    }

    // transform internal output into external output
    out = yolov8_impl::transform_output<OUTPUT>(decode_result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
YoloV8Detector<INPUT, OUTPUT>::YoloV8Detector() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
YoloV8Detector<INPUT, OUTPUT>::~YoloV8Detector() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool YoloV8Detector<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode YoloV8Detector<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace object_detection
} // namespace models
} // namespace jinq
