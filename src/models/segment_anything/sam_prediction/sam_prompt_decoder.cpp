/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamPromptDecoder.cpp
 * Date: 23-6-7
 ************************************************/

#include "sam_prompt_decoder.h"

#include "glog/logging.h"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "TensorRT-8.6.1.6/NvInferRuntime.h"

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

class SamPromptDecoder::Impl {
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
     * @param cfg
     * @return
     */
    StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param image_embeddings
     * @param bboxes
     * @param predicted_masks
     * @return
     */
    StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<cv::Rect2f>& bboxes,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param image_embeddings
     * @param prompt_points
     * @param predicted_masks
     * @return
     */
    StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<std::vector<cv::Point2f> >& prompt_points,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param ori_image_size
     */
    void set_ori_image_size(const cv::Size& ori_image_size) {
        _m_ori_image_size = ori_image_size;
    }

    /***
     *
     * @param input_node_size
     */
    void set_encoder_input_size(const cv::Size& input_node_size) {
        _m_encoder_input_size = input_node_size;
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

    // model backend device id
    uint8_t _m_device_id = 0;

    // model input/output names
    std::vector<const char*> _m_input_names;
    std::vector<const char*> _m_output_names;

    // model envs
    Ort::Env _m_env;
    Ort::MemoryInfo _m_memo_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // model session options
    Ort::SessionOptions _m_sess_options;

    // model session
    std::unique_ptr<Ort::Session> _m_decoder_sess;

    // model input/output shape info`
    std::vector<int> _m_encoder_input_shape;

    // tensorrt engine
    std::unique_ptr<nvinfer1::IRuntime> _m_trt_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> _m_trt_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> _m_trt_execution_context;
    std::unique_ptr<TrtLogger> _m_trt_logger;

    // input/output tensor binding
    EngineBinding _m_image_embedding_binding;
    EngineBinding _m_point_coords_binding;
    EngineBinding _m_point_labels_binding;
    EngineBinding _m_mask_input_binding;
    EngineBinding _m_has_mask_input_binding;
    EngineBinding _m_low_res_masks_output_binding;
    EngineBinding _m_iou_predictions_output_binding;

    // trt device memory
    DeviceMemory _m_device_memory;
    cudaStream_t _m_cuda_stream = nullptr;
    int32_t _m_max_decoder_point_counts = 128;

    // origin image size
    cv::Size _m_ori_image_size;
    // vit encoder input node size
    cv::Size _m_encoder_input_size = cv::Size(1024, 1024);

    // init flag
    bool _m_successfully_init_model = false;
    
    // use onnx mnn or trt
    enum model_type {
        TRT = 0,
        ONNX = 1,
    };
    model_type _m_backend_type = ONNX;

  private:
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
      * @param decoder_inputs
      * @param bbox
      * @param points
      * @param out_mask
      * @return
     */
    StatusCode get_mask(
        const std::vector<float>& image_embeddings,
        const cv::Rect2f& bbox,
        cv::Mat& out_mask);
    
    /***
      *
      * @param decoder_inputs
      * @param bbox
      * @param points
      * @param out_mask
      * @return
     */
    StatusCode onnx_get_mask(
        const std::vector<float>& image_embeddings,
        const cv::Rect2f& bbox,
        cv::Mat& out_mask);
    
    /***
      *
      * @param decoder_inputs
      * @param bbox
      * @param points
      * @param out_mask
      * @return
     */
    StatusCode trt_get_mask(
        const std::vector<float>& image_embeddings,
        const cv::Rect2f& bbox,
        cv::Mat& out_mask);
    
    /***
     * 
     * @param image_embeddings 
     * @param prompt_points 
     * @param out_mask 
     * @return 
     */
    StatusCode get_mask(
        const std::vector<float>& image_embeddings,
        const std::vector<cv::Point2f>& prompt_points,
        cv::Mat& out_mask);
    
    /***
     * 
     * @param image_embeddings 
     * @param prompt_points 
     * @param out_mask 
     * @return 
     */
    StatusCode onnx_get_mask(
        const std::vector<float>& image_embeddings,
        const std::vector<cv::Point2f>& prompt_points,
        cv::Mat& out_mask);
    
    /***
     * 
     * @param image_embeddings 
     * @param prompt_points 
     * @param out_mask 
     * @return 
     */
    StatusCode trt_get_mask(
        const std::vector<float>& image_embeddings,
        const std::vector<cv::Point2f>& prompt_points,
        cv::Mat& out_mask);

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

    /***
     *
     * @param low_res_mask_value
     * @param mask_idx
     * @param out_mask
     */
    void trt_decode_output_mask(const std::vector<float> &low_res_mask_value, int mask_idx, cv::Mat &out_mask);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamPromptDecoder::Impl::init(const decltype(toml::parse("")) &cfg) {
    // choose backend type
    auto backend_dict = cfg.at("BACKEND_DICT");
    auto backend_name = cfg.at("SAM_DECODER").at("backend_type").as_string();
    _m_backend_type = static_cast<model_type>(backend_dict[backend_name].as_integer());

    // init sam decoder configs
    toml::value sam_decoder_cfg;
    if (_m_backend_type == TRT) {
        sam_decoder_cfg = cfg.at("SAM_TRT_DECODER");
    } else {
        sam_decoder_cfg = cfg.at("SAM_ONNX_DECODER");
    }
    auto model_file_name = FilePathUtil::get_file_name(sam_decoder_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == TRT) {
        init_status = init_trt_model(sam_decoder_cfg);
    } else {
        init_status = init_onnx_model(sam_decoder_cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_init_model = true;
        LOG(INFO) << "Successfully load sam prompt decoder from: " << model_file_name;
    } else {
        _m_successfully_init_model = false;
        LOG(INFO) << "Failed load sam prompt decoder from: " << model_file_name;
    }

    return init_status;
}

/***
 *
 * @param image_embeddings
 * @param bboxes
 * @param predicted_masks
 * @return
 */
StatusCode SamPromptDecoder::Impl::decode(
    const std::vector<float>& image_embeddings,
    const std::vector<cv::Rect2f>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    // decoder masks
    for (auto& bbox : bboxes) {
        cv::Mat out_mask;
        auto status_code= get_mask(image_embeddings, bbox, out_mask);
        if (status_code != StatusCode::OJBK) {
            return status_code;
        }
        predicted_masks.push_back(out_mask);
    }

    return StatusCode::OJBK;
}

/***
 *
 * @param image_embeddings
 * @param prompt_points
 * @param predicted_masks
 * @return
 */
StatusCode SamPromptDecoder::Impl::decode(
    const std::vector<float>& image_embeddings,
    const std::vector<std::vector<cv::Point2f> >& prompt_points,
    std::vector<cv::Mat>& predicted_masks) {
    // decoder masks
    for (auto& points_per_obj : prompt_points) {
        cv::Mat out_mask;
        auto status_code= get_mask(image_embeddings, points_per_obj, out_mask);
        if (status_code != StatusCode::OJBK) {
            return status_code;
        }
        predicted_masks.push_back(out_mask);
    }

    return StatusCode::OJBK;
}

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamPromptDecoder::Impl::init_onnx_model(const toml::value &cfg) {
    // ort env and memo info
    _m_env = {ORT_LOGGING_LEVEL_WARNING, ""};

    // init sam decoder configs
    _m_model_path = cfg.at("model_file_path").as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "sam onnx prompt decoder model file path: " << _m_model_path << " not exists";
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_model_device = cfg.at("compute_backend").as_string();
    if (std::strcmp(_m_model_device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_device_id = cfg.at("gpu_device_id").as_integer();
    }
    _m_thread_nums = cfg.at("model_threads_num").as_integer();
    _m_sess_options.SetIntraOpNumThreads(_m_thread_nums);
    _m_sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    _m_sess_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_device_id;
        _m_sess_options.AppendExecutionProvider_CUDA(cuda_options);
    }
    _m_decoder_sess = std::make_unique<Ort::Session>(_m_env, _m_model_path.c_str(), _m_sess_options);

    _m_input_names = {"image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"};
    _m_output_names = {"masks", "iou_predictions", "low_res_masks"};

    return StatusCode::OJBK;
}

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamPromptDecoder::Impl::init_trt_model(const toml::value &cfg) {
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

    // bind image embedding tensor
    std::string input_node_name = "image_embeddings";
    auto successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, _m_image_embedding_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor image_embeddings failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_image_embedding_binding.dims().nbDims != 4) {
        std::string input_shape_str = TrtHelper::dims_to_string(_m_image_embedding_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_image_embedding_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind point coords tensor
    input_node_name = "point_coords";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, _m_point_coords_binding);
    nvinfer1::Dims3 point_coords_dims(1, _m_max_decoder_point_counts, 2);
    _m_point_coords_binding.set_dims(point_coords_dims);
    _m_trt_execution_context->setInputShape(input_node_name.c_str(), point_coords_dims);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor point_coords failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_point_coords_binding.dims().nbDims != 3) {
        auto input_shape_str = TrtHelper::dims_to_string(_m_point_coords_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [B, N, 2]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_point_coords_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind point labels tensor
    input_node_name = "point_labels";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, _m_point_labels_binding);
    nvinfer1::Dims2 point_labels_dims(1, _m_max_decoder_point_counts);
    _m_point_labels_binding.set_dims(point_labels_dims);
    _m_trt_execution_context->setInputShape(input_node_name.c_str(), point_labels_dims);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor point_labels failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_point_labels_binding.dims().nbDims != 2) {
        auto input_shape_str = TrtHelper::dims_to_string(_m_point_labels_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [B, N]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_point_labels_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind mask input tensor
    input_node_name = "mask_input";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, _m_mask_input_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor mask_input failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_mask_input_binding.dims().nbDims != 4) {
        auto input_shape_str = TrtHelper::dims_to_string(_m_mask_input_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [B, N, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_mask_input_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind has mask input tensor
    input_node_name = "has_mask_input";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, _m_has_mask_input_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor mask_input failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_has_mask_input_binding.dims().nbDims != 1) {
        auto input_shape_str = TrtHelper::dims_to_string(_m_has_mask_input_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N,]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_has_mask_input_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind low res masks output tensor
    std::string output_node_name = "low_res_masks";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, output_node_name, _m_low_res_masks_output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_low_res_masks_output_binding.dims().nbDims != 4) {
        auto output_shape_str = TrtHelper::dims_to_string(_m_low_res_masks_output_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_low_res_masks_output_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic output tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind iou predictions output tensor
    output_node_name = "iou_predictions";
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, output_node_name, _m_iou_predictions_output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_iou_predictions_output_binding.dims().nbDims != 2) {
        auto output_shape_str = TrtHelper::dims_to_string(_m_iou_predictions_output_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_iou_predictions_output_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic output tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup device memory
    auto set_device_memo_status = TrtHelper::setup_device_memory(
        _m_trt_engine, _m_trt_execution_context, _m_device_memory);
    if (set_device_memo_status != StatusCode::OK) {
        LOG(ERROR) << "setup device memory for model failed, status code: " << set_device_memo_status;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init cuda stream
    if (cudaStreamCreate(&_m_cuda_stream) != cudaSuccess) {
        LOG(ERROR) << "ERROR: cuda stream creation failed.";
        return StatusCode::MODEL_INIT_FAILED;
    }

    return StatusCode::OK;
}


/***
 *
 * @param decoder_inputs
 * @param bbox
 * @param points
 * @param out_mask
 * @return
 */
StatusCode SamPromptDecoder::Impl::get_mask(
    const std::vector<float>& image_embeddings,
    const cv::Rect2f &bbox,
    cv::Mat &out_mask) {
    StatusCode status;
    if (_m_backend_type == ONNX) {
        status = onnx_get_mask(image_embeddings, bbox, out_mask);
    } else {
        status = trt_get_mask(image_embeddings, bbox, out_mask);
    }
    return status;
}

/***
 *
 * @param image_embeddings
 * @param bbox
 * @param out_mask
 * @return
 */
StatusCode SamPromptDecoder::Impl::onnx_get_mask(const std::vector<float> &image_embeddings, const cv::Rect2f &bbox, cv::Mat &out_mask) {
    // init decoder inputs
    std::vector<Ort::Value> decoder_input_tensor;

    // init image embedding tensors
    std::vector<int64_t> encoder_output_shape = {1, 256, 64, 64};
    auto embedding_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, (float*)image_embeddings.data(), image_embeddings.size(),
        encoder_output_shape.data(), encoder_output_shape.size());
    decoder_input_tensor.push_back(std::move(embedding_tensor));

    // init points tensor and label tensor
    std::vector<float> total_points;
    std::vector<float> total_labels;
    // top left point
    auto tl_pt = bbox.tl();
    auto tl_x = tl_pt.x;
    auto tl_y = tl_pt.y;
    total_points.push_back(tl_x);
    total_points.push_back(tl_y);
    total_labels.push_back(2.0);
    // bottom right point
    auto br_x = tl_x + bbox.width;
    auto br_y = tl_y + bbox.height;
    total_points.push_back(br_x);
    total_points.push_back(br_y);
    total_labels.push_back(3.0);
    total_points.push_back(0.0);
    total_points.push_back(0.0);
    total_labels.push_back(-1.0);

    std::vector<int64_t> point_tensor_shape({1, static_cast<int64_t>(total_points.size() / 2), 2});
    auto point_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, total_points.data(),
        total_points.size(), point_tensor_shape.data(), point_tensor_shape.size());
    if (!point_tensor.IsTensor() || !point_tensor.HasValue()) {
        LOG(ERROR) << "create point tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(point_tensor));

    std::vector<int64_t> point_labels_tensor_shape({1, static_cast<int64_t>(total_labels.size())});
    auto point_label_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, total_labels.data(),
        total_labels.size(), point_labels_tensor_shape.data(),point_labels_tensor_shape.size());
    if (!point_label_tensor.IsTensor() || !point_label_tensor.HasValue()) {
        LOG(ERROR) << "create point labels tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(point_label_tensor));

    // init mask input tensor
    std::vector<float> mask_tensor_values(1 * 1 * 256 * 256, 0.0);
    std::vector<int64_t> mask_tensor_shape({1, 1, 256, 256});
    auto mask_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, mask_tensor_values.data(),
        mask_tensor_values.size(), mask_tensor_shape.data(),mask_tensor_shape.size());
    if (!mask_tensor.IsTensor() || !mask_tensor.HasValue()) {
        LOG(ERROR) << "create mask tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(mask_tensor));

    // init has mask input tensor
    std::vector<float> has_mask_tensor_values(1, 0.0);
    std::vector<int64_t> has_mask_tensor_shape({1});
    auto has_mask_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, has_mask_tensor_values.data(),
        has_mask_tensor_values.size(), has_mask_tensor_shape.data(),has_mask_tensor_shape.size());
    if (!has_mask_tensor.IsTensor() || !has_mask_tensor.HasValue()) {
        LOG(ERROR) << "create has mask tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(has_mask_tensor));

    // init ori image size input tensor
    std::vector<float> ori_image_size_tensor_values = {
        static_cast<float>(_m_ori_image_size.height), static_cast<float>(_m_ori_image_size.width)};
    std::vector<int64_t> ori_image_size_tensor_shape({2});
    auto ori_img_size_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, ori_image_size_tensor_values.data(),
        ori_image_size_tensor_values.size(), ori_image_size_tensor_shape.data(),
        ori_image_size_tensor_shape.size());
    if (!ori_img_size_tensor.IsTensor() || !ori_img_size_tensor.HasValue()) {
        LOG(ERROR) << "create ori image size tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(ori_img_size_tensor));

    // run decoder
    auto output_tensors = _m_decoder_sess->Run(
        Ort::RunOptions{nullptr}, _m_input_names.data(), decoder_input_tensor.data(),
        decoder_input_tensor.size(), _m_output_names.data(), _m_output_names.size());
    auto masks_preds_value = output_tensors[0].GetTensorMutableData<float>();

    auto output_mask_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int output_mask_h = static_cast<int>(output_mask_shape[2]);
    int output_mask_w = static_cast<int>(output_mask_shape[3]);
    cv::Mat mask(cv::Size(output_mask_w, output_mask_h), CV_8UC1);
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            mask.at<uchar>(row, col) = masks_preds_value[row * mask.cols + col] > 0 ? 255 : 0;
        }
    }
    mask.copyTo(out_mask);

    return StatusCode::OJBK;
}

/***
 *
 * @param image_embeddings
 * @param bbox
 * @param out_mask
 * @return
 */
StatusCode SamPromptDecoder::Impl::trt_get_mask(const std::vector<float> &image_embeddings, const cv::Rect2f &bbox, cv::Mat &out_mask) {
    // init image embedding cuda memo copy
    auto input_mem_size = static_cast<int32_t>(image_embeddings.size() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_image_embedding_binding.index()), image_embeddings.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    // init point coords and point labels cuda memo
    std::vector<float> total_points;
    std::vector<float> total_labels;
    // top left point
    auto tl_pt = bbox.tl();
    auto tl_x = tl_pt.x;
    auto tl_y = tl_pt.y;
    total_points.push_back(tl_x);
    total_points.push_back(tl_y);
    total_labels.push_back(2.0);
    // bottom right point
    auto br_x = tl_x + bbox.width;
    auto br_y = tl_y + bbox.height;
    total_points.push_back(br_x);
    total_points.push_back(br_y);
    total_labels.push_back(3.0);
    total_points.push_back(0.0);
    total_points.push_back(0.0);
    total_labels.push_back(-1.0);

    nvinfer1::Dims3 points_shape(1, static_cast<int>(total_points.size() / 2), 2);
    nvinfer1::Dims2 labels_shape(1, static_cast<int>(total_labels.size()));
    _m_point_coords_binding.set_dims(points_shape);
    _m_point_labels_binding.set_dims(labels_shape);
    _m_trt_execution_context->setInputShape("point_coords", points_shape);
    _m_trt_execution_context->setInputShape("point_labels", labels_shape);
    input_mem_size = static_cast<int32_t >(total_points.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_point_coords_binding.index()), total_points.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    input_mem_size = static_cast<int32_t >(total_labels.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_point_labels_binding.index()), total_labels.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // init masks cuda memo
    std::vector<float> mask_tensor_values(1 * 1 * 256 * 256, 0.0);
    input_mem_size = static_cast<int32_t >(mask_tensor_values.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_mask_input_binding.index()), mask_tensor_values.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // init has mask input tensor
    std::vector<float> has_mask_tensor_values(1, 0.0);
    input_mem_size = static_cast<int32_t >(has_mask_tensor_values.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_has_mask_input_binding.index()), has_mask_tensor_values.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    _m_trt_execution_context->setInputTensorAddress(
        "image_embeddings", _m_device_memory.at(_m_image_embedding_binding.index()));
    _m_trt_execution_context->setInputTensorAddress(
        "point_coords", _m_device_memory.at(_m_point_coords_binding.index()));
    _m_trt_execution_context->setInputTensorAddress(
        "point_labels", _m_device_memory.at(_m_point_labels_binding.index()));
    _m_trt_execution_context->setInputTensorAddress(
        "mask_input", _m_device_memory.at(_m_mask_input_binding.index()));
    _m_trt_execution_context->setInputTensorAddress(
        "has_mask_input", _m_device_memory.at(_m_has_mask_input_binding.index()));
    _m_trt_execution_context->setTensorAddress(
        "low_res_masks", _m_device_memory.at(_m_low_res_masks_output_binding.index()));
    _m_trt_execution_context->setTensorAddress(
        "iou_predictions", _m_device_memory.at(_m_iou_predictions_output_binding.index()));
    if (!_m_trt_execution_context->enqueueV3(_m_cuda_stream)) {
        LOG(ERROR) << "excute input data for inference failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    std::vector<float> low_res_mask_data;
    low_res_mask_data.resize(_m_low_res_masks_output_binding.volume());
    cuda_status = cudaMemcpyAsync(low_res_mask_data.data(),
                                  _m_device_memory.at(_m_low_res_masks_output_binding.index()),
                                  _m_low_res_masks_output_binding.volume() * sizeof(float),
                                  cudaMemcpyDeviceToHost, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    std::vector<float> iou_preds_data;
    iou_preds_data.resize(_m_iou_predictions_output_binding.volume());
    cuda_status = cudaMemcpyAsync(iou_preds_data.data(),
                                  _m_device_memory.at(_m_iou_predictions_output_binding.index()),
                                  _m_iou_predictions_output_binding.volume() * sizeof(float),
                                  cudaMemcpyDeviceToHost, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cudaStreamSynchronize(_m_cuda_stream);

    // parse output mask
    int best_mask_idx = static_cast<int>(
        std::distance(iou_preds_data.begin(), std::max_element(iou_preds_data.begin(), iou_preds_data.end())));
    trt_decode_output_mask(low_res_mask_data, best_mask_idx, out_mask);

    return StatusCode::OJBK;
}

/***
 *
 * @param image_embeddings
 * @param prompt_points
 * @param out_mask
 * @return
 */
StatusCode SamPromptDecoder::Impl::get_mask(
    const std::vector<float> &image_embeddings,
    const std::vector<cv::Point2f> &prompt_points,
    cv::Mat &out_mask) {
    StatusCode status;
    if (_m_backend_type == ONNX) {
        status = onnx_get_mask(image_embeddings, prompt_points, out_mask);
    } else {
        status = trt_get_mask(image_embeddings, prompt_points, out_mask);
    }
    return status;
}

/***
 *
 * @param image_embeddings
 * @param prompt_points
 * @param out_mask
 * @return
 */
StatusCode SamPromptDecoder::Impl::onnx_get_mask(
    const std::vector<float> &image_embeddings,
    const std::vector<cv::Point2f> &prompt_points,
    cv::Mat &out_mask) {
    // init decoder inputs
    std::vector<Ort::Value> decoder_input_tensor;

    // init image embedding tensors
    std::vector<int64_t> encoder_output_shape = {1, 256, 64, 64};
    auto embedding_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, (float*)image_embeddings.data(), image_embeddings.size(),
        encoder_output_shape.data(), encoder_output_shape.size());
    decoder_input_tensor.push_back(std::move(embedding_tensor));

    // init points tensor and label tensor
    std::vector<float> total_points;
    std::vector<float> total_labels;
    for (auto& pt : prompt_points) {
        total_points.push_back(pt.x);
        total_points.push_back(pt.y);
        total_labels.push_back(1.0);
    }
    total_points.push_back(0.0);
    total_points.push_back(0.0);
    total_labels.push_back(-1.0);

    std::vector<int64_t> point_tensor_shape({1, static_cast<int64_t>(total_points.size() / 2), 2});
    auto point_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, total_points.data(),
        total_points.size(), point_tensor_shape.data(), point_tensor_shape.size());
    if (!point_tensor.IsTensor() || !point_tensor.HasValue()) {
        LOG(ERROR) << "create point tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(point_tensor));

    std::vector<int64_t> point_labels_tensor_shape({1, static_cast<int64_t>(total_labels.size())});
    auto point_label_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, total_labels.data(),
        total_labels.size(), point_labels_tensor_shape.data(),point_labels_tensor_shape.size());
    if (!point_label_tensor.IsTensor() || !point_label_tensor.HasValue()) {
        LOG(ERROR) << "create point labels tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(point_label_tensor));

    // init mask input tensor
    std::vector<float> mask_tensor_values(1 * 1 * 256 * 256, 0.0);
    std::vector<int64_t> mask_tensor_shape({1, 1, 256, 256});
    auto mask_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, mask_tensor_values.data(),
        mask_tensor_values.size(), mask_tensor_shape.data(),mask_tensor_shape.size());
    if (!mask_tensor.IsTensor() || !mask_tensor.HasValue()) {
        LOG(ERROR) << "create mask tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(mask_tensor));

    // init has mask input tensor
    std::vector<float> has_mask_tensor_values(1, 0.0);
    std::vector<int64_t> has_mask_tensor_shape({1});
    auto has_mask_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, has_mask_tensor_values.data(),
        has_mask_tensor_values.size(), has_mask_tensor_shape.data(),has_mask_tensor_shape.size());
    if (!has_mask_tensor.IsTensor() || !has_mask_tensor.HasValue()) {
        LOG(ERROR) << "create has mask tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(has_mask_tensor));

    // init ori image size input tensor
    std::vector<float> ori_image_size_tensor_values = {
        static_cast<float>(_m_ori_image_size.height), static_cast<float>(_m_ori_image_size.width)};
    std::vector<int64_t> ori_image_size_tensor_shape({2});
    auto ori_img_size_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, ori_image_size_tensor_values.data(),
        ori_image_size_tensor_values.size(), ori_image_size_tensor_shape.data(),
        ori_image_size_tensor_shape.size());
    if (!ori_img_size_tensor.IsTensor() || !ori_img_size_tensor.HasValue()) {
        LOG(ERROR) << "create ori image size tensor for decoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    decoder_input_tensor.push_back(std::move(ori_img_size_tensor));

    // run decoder
    auto output_tensors = _m_decoder_sess->Run(
        Ort::RunOptions{nullptr}, _m_input_names.data(), decoder_input_tensor.data(),
        decoder_input_tensor.size(), _m_output_names.data(), _m_output_names.size());
    auto masks_preds_value = output_tensors[0].GetTensorMutableData<float>();

    auto output_mask_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int output_mask_h = static_cast<int>(output_mask_shape[2]);
    int output_mask_w = static_cast<int>(output_mask_shape[3]);
    cv::Mat mask(cv::Size(output_mask_w, output_mask_h), CV_8UC1);
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            mask.at<uchar>(row, col) = masks_preds_value[row * mask.cols + col] > 0 ? 255 : 0;
        }
    }
    mask.copyTo(out_mask);

    return StatusCode::OJBK;
}

/***
 *
 * @param image_embeddings
 * @param prompt_points
 * @param out_mask
 * @return
 */
StatusCode SamPromptDecoder::Impl::trt_get_mask(
    const std::vector<float> &image_embeddings,
    const std::vector<cv::Point2f> &prompt_points,
    cv::Mat &out_mask) {
    // init image embedding cuda memo copy
    auto input_mem_size = static_cast<int32_t>(image_embeddings.size() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_image_embedding_binding.index()), image_embeddings.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    // init point coords and point labels cuda memo
    std::vector<float> total_points;
    std::vector<float> total_labels;
    for (auto& pt : prompt_points) {
        total_points.push_back(pt.x);
        total_points.push_back(pt.y);
        total_labels.push_back(1.0f);
    }
    total_points.push_back(0.0f);
    total_points.push_back(0.0f);
    total_labels.push_back(-1.0);
    nvinfer1::Dims3 points_shape(1, static_cast<int>(total_points.size() / 2), 2);
    nvinfer1::Dims2 labels_shape(1, static_cast<int>(total_labels.size()));
    _m_point_coords_binding.set_dims(points_shape);
    _m_point_labels_binding.set_dims(labels_shape);
    _m_trt_execution_context->setInputShape("point_coords", points_shape);
    _m_trt_execution_context->setInputShape("point_labels", labels_shape);
    input_mem_size = static_cast<int32_t >(total_points.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_point_coords_binding.index()), total_points.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    input_mem_size = static_cast<int32_t >(total_labels.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_point_labels_binding.index()), total_labels.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // init masks cuda memo
    std::vector<float> mask_tensor_values(1 * 1 * 256 * 256, 0.0);
    input_mem_size = static_cast<int32_t >(mask_tensor_values.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_mask_input_binding.index()), mask_tensor_values.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // init has mask input tensor
    std::vector<float> has_mask_tensor_values(1, 0.0);
    input_mem_size = static_cast<int32_t >(has_mask_tensor_values.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_has_mask_input_binding.index()), has_mask_tensor_values.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // do inference
    _m_trt_execution_context->setInputTensorAddress(
        "image_embeddings", _m_device_memory.at(_m_image_embedding_binding.index()));
    _m_trt_execution_context->setInputTensorAddress(
        "point_coords", _m_device_memory.at(_m_point_coords_binding.index()));
    _m_trt_execution_context->setInputTensorAddress(
        "point_labels", _m_device_memory.at(_m_point_labels_binding.index()));
    _m_trt_execution_context->setInputTensorAddress(
        "mask_input", _m_device_memory.at(_m_mask_input_binding.index()));
    _m_trt_execution_context->setInputTensorAddress(
        "has_mask_input", _m_device_memory.at(_m_has_mask_input_binding.index()));
    _m_trt_execution_context->setTensorAddress(
        "low_res_masks", _m_device_memory.at(_m_low_res_masks_output_binding.index()));
    _m_trt_execution_context->setTensorAddress(
        "iou_predictions", _m_device_memory.at(_m_iou_predictions_output_binding.index()));
    if (!_m_trt_execution_context->enqueueV3(_m_cuda_stream)) {
        LOG(ERROR) << "excute input data for inference failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    std::vector<float> low_res_mask_data;
    low_res_mask_data.resize(_m_low_res_masks_output_binding.volume());
    cuda_status = cudaMemcpyAsync(low_res_mask_data.data(),
                                  _m_device_memory.at(_m_low_res_masks_output_binding.index()),
                                  _m_low_res_masks_output_binding.volume() * sizeof(float),
                                  cudaMemcpyDeviceToHost, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    std::vector<float> iou_preds_data;
    iou_preds_data.resize(_m_iou_predictions_output_binding.volume());
    cuda_status = cudaMemcpyAsync(iou_preds_data.data(),
                                  _m_device_memory.at(_m_iou_predictions_output_binding.index()),
                                  _m_iou_predictions_output_binding.volume() * sizeof(float),
                                  cudaMemcpyDeviceToHost, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    cudaStreamSynchronize(_m_cuda_stream);

    // parse output mask
    int best_mask_idx = static_cast<int>(
        std::distance(iou_preds_data.begin(), std::max_element(iou_preds_data.begin(), iou_preds_data.end())));
    trt_decode_output_mask(low_res_mask_data, best_mask_idx, out_mask);

    return StatusCode::OJBK;
}

/***
 *
 * @param low_res_mask_value
 * @param mask_idx
 * @param out_mask
 */
void SamPromptDecoder::Impl::trt_decode_output_mask(const std::vector<float> &low_res_mask_value, const int mask_idx, cv::Mat &out_mask) {
    // select best low res mask
    cv::Mat mask(cv::Size(256, 256), CV_32FC1);
    for (auto row = 0; row < 256; ++row) {
        auto row_data = mask.ptr<float>(row);
        for (auto col = 0; col < 256; ++col) {
            row_data[col] = low_res_mask_value[mask_idx * 256 * 256 + row * 256 + col];
        }
    }
    // resize low res mask into large res
    cv::resize(mask, mask, _m_encoder_input_size);
    // crop out padded part
    auto ori_img_width = static_cast<float>(_m_ori_image_size.width);
    auto ori_img_height = static_cast<float>(_m_ori_image_size.height);
    auto long_side = std::max(_m_ori_image_size.height, _m_ori_image_size.width);
    float scale = static_cast<float>(_m_encoder_input_size.height) / static_cast<float>(long_side);
    cv::Size target_size = cv::Size(
        static_cast<int>(scale * ori_img_width), static_cast<int>(scale * ori_img_height));
    auto pad_h = _m_encoder_input_size.height - target_size.height;
    auto pad_w = _m_encoder_input_size.width - target_size.width;
    cv::Rect cropped_roi(0, 0, _m_encoder_input_size.width - pad_w, _m_encoder_input_size.height - pad_h);
    mask = mask(cropped_roi);
    // resize mask into ori image size
    cv::resize(mask, mask, _m_ori_image_size);
    // fill in mask value
    cv::Mat o_mask(_m_ori_image_size, CV_8UC1);
    for (int row = 0; row < mask.rows; ++row) {
        auto row_data = o_mask.ptr(row);
        auto mask_data = mask.ptr<float>(row);
        for (int col = 0; col < mask.cols; ++col) {
            row_data[col] = mask_data[col] > 0.0 ? 255 : 0;
        }
    }
    o_mask.copyTo(out_mask);
}

/***
 *
 */
SamPromptDecoder::SamPromptDecoder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
SamPromptDecoder::~SamPromptDecoder() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamPromptDecoder::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param image_embeddings
 * @param bboxes
 * @param predicted_masks
 * @return
 */
StatusCode SamPromptDecoder::decode(
    const std::vector<float>& image_embeddings,
    const std::vector<cv::Rect2f>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    return _m_pimpl->decode(image_embeddings, bboxes, predicted_masks);
}

/***
 *
 * @param image_embeddings
 * @param points
 * @param predicted_masks
 * @return
 */
StatusCode SamPromptDecoder::decode(
    const std::vector<float> &image_embeddings,
    const std::vector<std::vector<cv::Point2f>> &points,
    std::vector<cv::Mat> &predicted_masks) {
    return _m_pimpl->decode(image_embeddings, points, predicted_masks);
}

/***
 *
 * @param ori_img_size
 */
void SamPromptDecoder::set_ori_image_size(const cv::Size &ori_img_size) {
    return _m_pimpl->set_ori_image_size(ori_img_size);
}

/***
 *
 * @param input_node_size
 */
void SamPromptDecoder::set_encoder_input_size(const cv::Size &input_node_size) {
    return _m_pimpl->set_encoder_input_size(input_node_size);
}

/***
 *
 * @return
 */
bool SamPromptDecoder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}