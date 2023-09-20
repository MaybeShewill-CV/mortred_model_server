/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_trt_everything_decoder.cpp
 * Date: 23-9-20
 ************************************************/

#include "sam_trt_amg_decoder.h"

#include "glog/logging.h"
#include "stl_container/concurrentqueue.h"
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

class SamTrtEverythingDecoder::Impl {
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
     * @param points
     * @param predicted_masks
     * @return
     */
    StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<std::vector<cv::Point2f> >& points,
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
        return _m_successfully_initialized;
    }

  private:
    // model file path
    std::string _m_model_file_path;

    // model input/output names
    std::vector<const char*> _m_input_names;
    std::vector<const char*> _m_output_names;

    // tensorrt engine
    std::unique_ptr<nvinfer1::IRuntime> _m_trt_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> _m_trt_engine;
    std::unique_ptr<TrtLogger> _m_trt_logger;

    // decoder thread executor
    struct SamDecodeInput {
        // bindings
        EngineBinding image_embedding_binding;
        EngineBinding point_coords_binding;
        EngineBinding point_labels_binding;
        EngineBinding mask_input_binding;
        EngineBinding has_mask_input_binding;
        EngineBinding low_res_masks_output_binding;
        EngineBinding iou_predictions_output_binding;

        // device memory
        DeviceMemory device_memory;
        cudaStream_t cuda_stream = nullptr;
    };
    // thread executor
    using ThreadExecutor = std::pair<nvinfer1::IExecutionContext*, SamDecodeInput>;
    // worker queue
    moodycamel::ConcurrentQueue<ThreadExecutor> _m_decoder_queue;
    // worker queue size
    int _m_decoder_queue_size = 4;
    // parallel decode context
    struct parallel_decode_seriex_ctx {
        std::vector<cv::Mat> decoded_masks;
    };

    // origin image size
    cv::Size _m_ori_image_size;
    // vit encoder input node size
    cv::Size _m_encoder_input_size = cv::Size(1024, 1024);

    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @param input_file_path
     * @param file_content
     * @return
     */
    static bool read_model_file(const std::string& input_file_path, std::vector<unsigned char>& file_content);

    /***
     *
     * @param image_embeddings
     * @param point_coords
     * @param point_labels
     * @param out_mask
     * @return
     */
    StatusCode get_mask(const std::vector<cv::Point2f>& point_coords,
                        const std::vector<float>& point_labels,
                        cv::Mat& out_mask);

    /***
     *
     * @param low_res_mask_value
     * @param mask_idx
     * @param out_mask
     * @param encoder_input_size
     * @return
     */
    void decode_output_mask(
        const std::vector<float>& low_res_mask_value,
        int mask_idx,
        cv::Mat& out_mask);

    /***
     *
     * @param decoder_input
     * @return
     */
    StatusCode init_thread_executor(ThreadExecutor& executor);

    /***
     *
     * @param ctx
     */
    void thread_decode_mask_proc(parallel_decode_seriex_ctx* ctx);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamTrtEverythingDecoder::Impl::init(const decltype(toml::parse("")) &cfg) {
    // init sam vit trt config section
    if (!cfg.contains("SAM_VIT_TRT_DECODER")) {
        LOG(ERROR) << "Config file does not contain SAM_VIT_TRT_DECODER section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    toml::value cfg_content = cfg.at("SAM_VIT_TRT_DECODER");

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

    // init mask decoder executor queue
    _m_decoder_queue_size = static_cast<int>(cfg_content.at("worker_queue_size").as_integer());
    for (auto idx = 0; idx < _m_decoder_queue_size; ++idx) {
        ThreadExecutor executor;
        auto* context = _m_trt_engine->createExecutionContext();
        executor.first = context;
        auto init_status = init_thread_executor(executor);
        if (init_status != StatusCode::OK) {
            LOG(ERROR) << "init thread mask decode executor failed, status code: " << init_status;
            _m_successfully_initialized = false;
            return StatusCode::MODEL_INIT_FAILED;
        }
        _m_decoder_queue.enqueue(std::move(executor));
    }
    LOG(INFO) << "init mask decoder queue complete";

    _m_successfully_initialized = true;
    LOG(INFO) << "Sam trt segmentation model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
    return StatusCode::OK;
}

/***
 *
 * @param image_embeddings
 * @param bboxes
 * @param predicted_masks
 * @return
 */
StatusCode SamTrtEverythingDecoder::Impl::decode(
    const std::vector<float> &image_embeddings,
    const std::vector<std::vector<cv::Point2f> > &points,
    std::vector<cv::Mat> &predicted_masks) {
    // init image embedding cuda memo copy
    auto input_mem_size = static_cast<int32_t>(image_embeddings.size() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
        _m_device_memory.at(_m_image_embedding_binding.index()), image_embeddings.data(), input_mem_size,
        cudaMemcpyHostToDevice, _m_cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy input image memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    // decode masks
    for (auto& point_coords : points) {
        std::vector<float> point_labels;
        point_labels.resize(point_coords.size(), 1.0);
        cv::Mat out_mask;
        auto status = get_mask(point_coords, point_labels, out_mask);
        if (status != StatusCode::OK) {
            LOG(ERROR) << "decode mask from image failed, status code: " << status;
            return status;
        }
        predicted_masks.push_back(out_mask);
    }
    return StatusCode::OJBK;
}

/***
 *
 * @param input_file_path
 * @param file_content
 * @return
 */
bool SamTrtEverythingDecoder::Impl::read_model_file(const std::string &input_file_path, std::vector<unsigned char> &file_content) {
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
 * @param image_embeddings
 * @param point_coords
 * @param point_labels
 * @param out_mask
 * @return
 */
StatusCode SamTrtEverythingDecoder::Impl::get_mask(
    const std::vector<cv::Point2f> &point_coords,
    const std::vector<float> &point_labels,
    cv::Mat &out_mask) {
    auto t_start = std::chrono::high_resolution_clock::now();
    // init point coords and point labels cuda memo
    std::vector<float> total_points;
    for (auto& pt : point_coords) {
        total_points.push_back(pt.x);
        total_points.push_back(pt.y);
    }
    total_points.push_back(0.0f);
    total_points.push_back(0.0f);
    std::vector<float> total_labels = point_labels;
    total_labels.push_back(-1.0);
    nvinfer1::Dims3 points_shape(1, static_cast<int>(total_points.size() / 2), 2);
    nvinfer1::Dims2 labels_shape(1, static_cast<int>(total_labels.size()));
    _m_point_coords_binding.set_dims(points_shape);
    _m_point_labels_binding.set_dims(labels_shape);
    _m_trt_execution_context->setInputShape("point_coords", points_shape);
    _m_trt_execution_context->setInputShape("point_labels", labels_shape);
    auto input_mem_size = static_cast<int32_t >(total_points.size() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
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
    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    DLOG(INFO) << "      ---- mask decode memcpy cpu to gpu cost time: " << t_cost << " ms";

    // do inference
    t_start = std::chrono::high_resolution_clock::now();
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
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    DLOG(INFO) << "      ---- mask decode inference cost time: " << t_cost << " ms";

    // parse output mask
    t_start = std::chrono::high_resolution_clock::now();
    int best_mask_idx = static_cast<int>(
        std::distance(iou_preds_data.begin(), std::max_element(iou_preds_data.begin(), iou_preds_data.end())));
    decode_output_mask(low_res_mask_data, best_mask_idx, out_mask);
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    DLOG(INFO) << "      ---- mask decode output mask cost time: " << t_cost << " ms";

    return StatusCode::OJBK;
}

/***
 *
 * @param low_res_mask_value
 * @param mask_idx
 * @param out_mask
 * @param encoder_input_size
 * @return
 */
void SamTrtEverythingDecoder::Impl::decode_output_mask(
    const std::vector<float> &low_res_mask_value, const int mask_idx, cv::Mat &out_mask) {
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
 * @param executor
 * @return
 */
StatusCode SamTrtEverythingDecoder::Impl::init_thread_executor(ThreadExecutor& executor) {
    auto& context = executor.first;
    auto& decoder_input = executor.second;

    // bind image embedding tensor
    auto& image_embedding_binding = decoder_input.image_embedding_binding;
    std::string input_node_name = "image_embeddings";
    auto successfully_bind = TrtHelper::setup_engine_binding(
        _m_trt_engine, input_node_name, image_embedding_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor image_embeddings failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (image_embedding_binding.dims().nbDims != 4) {
        std::string input_shape_str;
        for (auto idx = 0; idx < image_embedding_binding.dims().nbDims; ++idx) {
            input_shape_str += std::to_string(image_embedding_binding.dims().d[idx]) + ",";
        }
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (image_embedding_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind point coords tensor
    input_node_name = "point_coords";
    auto& point_coords_binding = decoder_input.point_coords_binding;
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, point_coords_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor point_coords failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (point_coords_binding.dims().nbDims != 3) {
        auto input_shape_str = TrtHelper::dims_to_string(point_coords_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [B, N, 2]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (point_coords_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind point labels tensor
    input_node_name = "point_labels";
    auto& point_labels_binding = decoder_input.point_labels_binding;
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, point_labels_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor point_labels failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (point_labels_binding.dims().nbDims != 2) {
        std::string input_shape_str = TrtHelper::dims_to_string(point_labels_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [B, N]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (point_labels_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind mask input tensor
    input_node_name = "mask_input";
    auto& mask_input_binding = decoder_input.mask_input_binding;
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, mask_input_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor mask_input failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (mask_input_binding.dims().nbDims != 4) {
        std::string input_shape_str = TrtHelper::dims_to_string(mask_input_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [B, N, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (mask_input_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind has mask input tensor
    input_node_name = "has_mask_input";
    auto& has_mask_input_binding = decoder_input.has_mask_input_binding;
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, input_node_name, has_mask_input_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind input tensor mask_input failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (has_mask_input_binding.dims().nbDims != 1) {
        std::string input_shape_str = TrtHelper::dims_to_string(has_mask_input_binding.dims());
        LOG(ERROR) << "wrong input tensor shape: " << input_shape_str << " expected: [N,]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (has_mask_input_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic input tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind low res masks output tensor
    std::string output_node_name = "low_res_masks";
    auto& low_res_masks_output_binding = decoder_input.low_res_masks_output_binding;
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, output_node_name, low_res_masks_output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (low_res_masks_output_binding.dims().nbDims != 4) {
        std::string output_shape_str = TrtHelper::dims_to_string(low_res_masks_output_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C, H, W]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (low_res_masks_output_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic output tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // bind iou predictions output tensor
    output_node_name = "iou_predictions";
    auto& iou_predictions_output_binding = decoder_input.iou_predictions_output_binding;
    successfully_bind = TrtHelper::setup_engine_binding(_m_trt_engine, output_node_name, iou_predictions_output_binding);
    if (!successfully_bind) {
        LOG(ERROR) << "bind output tensor failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (iou_predictions_output_binding.dims().nbDims != 2) {
        std::string output_shape_str = TrtHelper::dims_to_string(iou_predictions_output_binding.dims());
        LOG(ERROR) << "wrong output tensor shape: " << output_shape_str << " expected: [N, C]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (iou_predictions_output_binding.is_dynamic()) {
        LOG(ERROR) << "trt not support dynamic output tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }

    // setup device memory
    auto& device_memory = decoder_input.device_memory;
    auto context_ptr = std::unique_ptr<nvinfer1::IExecutionContext>(context);
    auto set_device_memo_status = TrtHelper::setup_device_memory(
        _m_trt_engine, context_ptr, device_memory);
    if (set_device_memo_status != StatusCode::OK) {
        LOG(ERROR) << "setup device memory for model failed, status code: " << set_device_memo_status;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init cuda stream
    auto& cuda_stream = decoder_input.cuda_stream;
    if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
        LOG(ERROR) << "ERROR: cuda stream creation failed." << std::endl;
        return StatusCode::MODEL_INIT_FAILED;
    }

    return StatusCode::OK;
}

/***
 *
 */
SamTrtEverythingDecoder::SamTrtEverythingDecoder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
SamTrtEverythingDecoder::~SamTrtEverythingDecoder() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamTrtEverythingDecoder::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param image_embeddings
 * @param points
 * @param predicted_masks
 * @return
 */
StatusCode SamTrtEverythingDecoder::decode(
    const std::vector<float> &image_embeddings,
    const std::vector<std::vector<cv::Point2f> > &points,
    std::vector<cv::Mat> &predicted_masks) {
    return _m_pimpl->decode(image_embeddings, points, predicted_masks);
}

/***
 *
 * @param ori_img_size
 */
void SamTrtEverythingDecoder::set_ori_image_size(const cv::Size &ori_img_size) {
    return _m_pimpl->set_ori_image_size(ori_img_size);
}

/***
 *
 * @param ori_img_size
 */
void SamTrtEverythingDecoder::set_encoder_input_size(const cv::Size &input_node_size){
    return _m_pimpl->set_encoder_input_size(input_node_size);
}

/***
 *
 * @return
 */
bool SamTrtEverythingDecoder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}