/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_trt_everything_decoder.cpp
 * Date: 23-9-20
 ************************************************/

#include "sam_amg_decoder.h"

#include "glog/logging.h"
#include "stl_container/concurrentqueue.h"
#include "TensorRT-8.6.1.6/NvInferRuntime.h"
#include "workflow/WFFacilities.h"
#include "workflow/Workflow.h"

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

class SamAmgDecoder::Impl {
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
     * @param amg_output
     * @param points_per_side
     * @param pred_iou_thresh
     * @param stability_score_thresh
     * @param box_nms_thresh
     * @param min_mask_region_area
     * @return
     */
    StatusCode decode_everything(
        const std::vector<float> &image_embeddings,
        AmgMaskOutput& amg_output, int points_per_side = 32, float pred_iou_thresh = 0.88,
        float stability_score_thresh = 0.95, float box_nms_thresh = 0.7, int min_mask_region_area = 0);

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
    struct ThreadExecutor {
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        std::unique_ptr<SamDecodeInput> input;
    };
    // worker queue
    moodycamel::ConcurrentQueue<ThreadExecutor> _m_decoder_queue;
    // worker queue size
    int _m_decoder_queue_size = 4;
    // parallel compute thread nums
    int _m_compute_thread_nums = 1;
    // parallel decode context
    struct thread_decode_seriex_ctx {
        // output params
        cv::Mat decoded_masks;
        StatusCode model_run_status = StatusCode::OK;
        float pred_iou = 0.0f;
        float stability_score = 0.0f;
        cv::Point2f point_coord;
        // time consuming measurement
        long dequeue_thread_executor_time_consuming = 0;
        long gpu_memo_cpy_time_consuming = 0;
        long model_inference_consuming = 0;
        long decode_mask_time_consuming = 0;
        long enqueue_thread_executor_time_consuming = 0;
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
     * @param points
     * @param predicted_masks
     * @param predicted_iou
     * @param stability_scores
     * @param point_coords
     * @return
     */
    StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<std::vector<cv::Point2f> >& points,
        std::vector<cv::Mat>& predicted_masks,
        std::vector<float>& predicted_iou,
        std::vector<float>& stability_scores,
        std::vector<cv::Point2f>& point_coords);

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
     * @param image_embeddings
     * @param point
     * @param ctx
     */
    void thread_decode_mask_proc(
        const std::vector<float>& image_embeddings,
        const cv::Point2f& point,
        thread_decode_seriex_ctx* ctx);

    /***
     *
     * @param input_image_size
     * @param n_points_per_side
     * @return
     */
    static std::vector<std::vector<cv::Point2f> > generate_prompt_points(const cv::Size& input_image_size, int n_points_per_side);

    /***
     *
     * @param mask
     * @param mask_threshold
     * @param threshold_offset
     * @return
     */
    static float calculate_stability_score(const cv::Mat& mask);

    /***
     *
     * @param pred_masks
     * @param pred_ious
     * @param pred_stability_scores
     * @param point_coords
     * @param pred_iou_thresh
     * @param stability_score_thresh
     * @param box_nms_thresh
     * @param min_mask_region_area
     * @param amg_output
     */
    static void filter_output_masks(
        const std::vector<cv::Mat>& pred_masks, const std::vector<float>& pred_ious, const std::vector<float>& pred_stability_scores,
        const std::vector<cv::Point2f>& point_coords, float pred_iou_thresh,
        float stability_score_thresh, float box_nms_thresh, int min_mask_region_area,
        AmgMaskOutput& amg_output);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamAmgDecoder::Impl::init(const decltype(toml::parse("")) &cfg) {
    // init sam vit trt config section
    if (!cfg.contains("SAM_AMG_DECODER")) {
        LOG(ERROR) << "Config file does not contain SAM_AMG_DECODER section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    toml::value cfg_content = cfg.at("SAM_AMG_DECODER");

    // init trt runtime
    _m_trt_logger = std::make_unique<TrtLogger>();
    auto* trt_runtime = nvinfer1::createInferRuntime(*_m_trt_logger);
    if(trt_runtime == nullptr) {
        LOG(ERROR) << "init tensorrt runtime failed";
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
        LOG(ERROR) << "sam amg decoder model file: " << _m_model_file_path << " not exist";
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
        // init trt context
        auto context = std::unique_ptr<nvinfer1::IExecutionContext>(_m_trt_engine->createExecutionContext());
        // init decoder input
        auto decoder_input = std::make_unique<SamDecodeInput>();

        ThreadExecutor executor;
        executor.context = std::move(context);
        executor.input = std::move(decoder_input);
        auto init_status = init_thread_executor(executor);
        if (init_status != StatusCode::OK) {
            LOG(ERROR) << "init thread mask decode executor failed, status code: " << init_status;
            _m_successfully_initialized = false;
            return StatusCode::MODEL_INIT_FAILED;
        }
        _m_decoder_queue.enqueue(std::move(executor));
    }

    // init compute thread pool
    _m_compute_thread_nums = static_cast<int>(cfg_content.at("compute_threads").as_integer());
    WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.compute_threads = _m_compute_thread_nums;
    WORKFLOW_library_init(&settings);

    _m_successfully_initialized = true;
    LOG(INFO) << "Successfully load sam amg decoder from: " << FilePathUtil::get_file_name(_m_model_file_path);

    return StatusCode::OK;
}

/***
 *
 * @param image_embeddings
 * @param amg_output
 * @param points_per_side
 * @param pred_iou_thresh
 * @param stability_score_thresh
 * @param stability_score_offset
 * @param box_nms_thresh
 * @param min_mask_region_area
 * @return
 */
StatusCode SamAmgDecoder::Impl::decode_everything(
    const std::vector<float>& image_embeddings,
    AmgMaskOutput& amg_output, const int points_per_side, const float pred_iou_thresh,
    const float stability_score_thresh, const float box_nms_thresh, const int min_mask_region_area) {
    // generate decoding prompt points
    auto prompt_pts = generate_prompt_points(_m_ori_image_size, points_per_side);

    // decode masks
    std::vector<cv::Mat> pred_masks;
    std::vector<float> pred_ious;
    std::vector<float> pred_stability_scores;
    std::vector<cv::Point2f> point_coords;
    auto status = decode(
        image_embeddings, prompt_pts, pred_masks, pred_ious, pred_stability_scores, point_coords);
    if (status != StatusCode::OK) {
        LOG(INFO) << "decode mask from prompt points failed, status code: " << status;
        return status;
    }

    // filter output masks
    filter_output_masks(pred_masks, pred_ious, pred_stability_scores, point_coords, pred_iou_thresh,
                        stability_score_thresh, box_nms_thresh, min_mask_region_area,amg_output);

    return status;
}

/***
 *
 * @param input_file_path
 * @param file_content
 * @return
 */
bool SamAmgDecoder::Impl::read_model_file(const std::string &input_file_path, std::vector<unsigned char> &file_content) {
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
 * @param points
 * @param predicted_masks
 * @param predicted_iou
 * @param point_coords
 * @return
 */
StatusCode SamAmgDecoder::Impl::decode(
    const std::vector<float> &image_embeddings,
    const std::vector<std::vector<cv::Point2f> > &points,
    std::vector<cv::Mat> &predicted_masks,
    std::vector<float>& predicted_iou,
    std::vector<float>& stability_scores,
    std::vector<cv::Point2f>& point_coords) {

    WFFacilities::WaitGroup wait_group(1);
    StatusCode status = StatusCode::OK;
    // create workflow parallel series
    auto* p_series = Workflow::create_parallel_work([&](const ParallelWork* pwork) -> void {
        for (auto idx = 0; idx < pwork->size(); ++idx) {
            auto* series_ctx = (thread_decode_seriex_ctx*)pwork->series_at(idx)->get_context();
            if (series_ctx->model_run_status != StatusCode::OK) {
                status = series_ctx->model_run_status;
            } else {
                predicted_masks.push_back(series_ctx->decoded_masks);
                predicted_iou.push_back(series_ctx->pred_iou);
                stability_scores.push_back(series_ctx->stability_score);
                point_coords.push_back(series_ctx->point_coord);
            }
//            LOG(INFO) << "      -- series: " << idx << " decode time profile";
//            LOG(INFO) << "      -- dequeue thread executor cost time: " << series_ctx->dequeue_thread_executor_time_consuming << " ms";
//            LOG(INFO) << "      -- copy inputs to gpu memory cost time: " << series_ctx->gpu_memo_cpy_time_consuming << " ms";
//            LOG(INFO) << "      -- decoding model inference cost time: " << series_ctx->model_inference_consuming << " ms";
//            LOG(INFO) << "      -- decode output mask cost time: " << series_ctx->decode_mask_time_consuming << " ms";
//            LOG(INFO) << "      -- enqueue thread executor cost time: " << series_ctx->enqueue_thread_executor_time_consuming << " ms";
            delete series_ctx;
        }
        wait_group.done();
    });

    // add multiple decode task into parallel series
    for (auto& pts : points) {
        auto* ctx = new thread_decode_seriex_ctx;
        auto&& decode_proc = [this](auto && PH1, auto && PH2, auto && PH3) {
            thread_decode_mask_proc(std::forward<decltype(PH1)>(PH1),
                                    std::forward<decltype(PH2)>(PH2),std::forward<decltype(PH3)>(PH3)); };
        auto* decode_task = WFTaskFactory::create_go_task(
            "parallel_decode_mask", decode_proc, image_embeddings, pts[0], ctx);
        auto* series = Workflow::create_series_work(decode_task, nullptr);
        series->set_context(ctx);
        p_series->add_series(series);
    }
    p_series->start();
    wait_group.wait();

    return status;
}

/***
 *
 * @param low_res_mask_value
 * @param mask_idx
 * @param out_mask
 * @param encoder_input_size
 * @return
 */
void SamAmgDecoder::Impl::decode_output_mask(
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
    
    mask.copyTo(out_mask);
}

/***
 *
 * @param executor
 * @return
 */
StatusCode SamAmgDecoder::Impl::init_thread_executor(ThreadExecutor& executor) {
//    auto& engine = executor.trt_engine;
    auto& context = executor.context;
    auto& decoder_input = executor.input;

    // bind image embedding tensor
    auto& image_embedding_binding = decoder_input->image_embedding_binding;
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
    auto& point_coords_binding = decoder_input->point_coords_binding;
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
    auto& point_labels_binding = decoder_input->point_labels_binding;
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
    auto& mask_input_binding = decoder_input->mask_input_binding;
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
    auto& has_mask_input_binding = decoder_input->has_mask_input_binding;
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
    auto& low_res_masks_output_binding = decoder_input->low_res_masks_output_binding;
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
    auto& iou_predictions_output_binding = decoder_input->iou_predictions_output_binding;
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
    auto set_device_memo_status = TrtHelper::setup_device_memory(
        _m_trt_engine, context, decoder_input->device_memory);
    if (set_device_memo_status != StatusCode::OK) {
        LOG(ERROR) << "setup device memory for model failed, status code: " << set_device_memo_status;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init cuda stream
    auto& cuda_stream = decoder_input->cuda_stream;
    if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
        LOG(ERROR) << "ERROR: cuda stream creation failed." << std::endl;
        return StatusCode::MODEL_INIT_FAILED;
    }

    return StatusCode::OK;
}

/***
 *
 * @param image_embeddings
 * @param point
 * @param ctx
 */
void SamAmgDecoder::Impl::thread_decode_mask_proc(
    const std::vector<float> &image_embeddings,
    const cv::Point2f &point,
    thread_decode_seriex_ctx *ctx) {
    // get decoder
    auto t_start = std::chrono::high_resolution_clock::now();
    ThreadExecutor decode_executor;
    while (!_m_decoder_queue.try_dequeue(decode_executor)) {}
    auto& context = decode_executor.context;
    auto& decoder_input = decode_executor.input;
    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds >(t_end - t_start).count();
    ctx->dequeue_thread_executor_time_consuming = t_cost;

    // prepare input bindings
    auto& image_embedding_binding = decoder_input->image_embedding_binding;
    auto& point_coords_binding = decoder_input->point_coords_binding;
    auto& point_labels_binding = decoder_input->point_labels_binding;
    auto& mask_input_binding = decoder_input->mask_input_binding;
    auto& has_mask_input_binding = decoder_input->has_mask_input_binding;
    auto& low_res_masks_output_binding = decoder_input->low_res_masks_output_binding;
    auto& iou_predictions_output_binding = decoder_input->iou_predictions_output_binding;

    auto& device_memory = decoder_input->device_memory;
    auto& cuda_stream = decoder_input->cuda_stream;

    // init image embedding cuda memo copy
    t_start = std::chrono::high_resolution_clock::now();
    auto input_mem_size = static_cast<int32_t>(image_embeddings.size() * sizeof(float));
    auto cuda_status = cudaMemcpyAsync(
        device_memory.at(image_embedding_binding.index()), image_embeddings.data(), input_mem_size,
        cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy image embedding memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        return;
    }

    // init point coords/labels memo
    std::vector<float> total_points;
    total_points.push_back(point.x);
    total_points.push_back(point.y);
    total_points.push_back(0.0f);
    total_points.push_back(0.0f);
    std::vector<float> total_labels = {1.0, -1.0};
    input_mem_size = static_cast<int32_t >(total_points.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        device_memory.at(point_coords_binding.index()), total_points.data(), input_mem_size,
        cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy point coords memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        return;
    }
    input_mem_size = static_cast<int32_t >(total_labels.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        device_memory.at(point_labels_binding.index()), total_labels.data(), input_mem_size,
        cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy point labels memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        return;
    }

    // init masks cuda memo
    std::vector<float> mask_tensor_values(1 * 1 * 256 * 256, 0.0);
    input_mem_size = static_cast<int32_t >(mask_tensor_values.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        device_memory.at(mask_input_binding.index()), mask_tensor_values.data(), input_mem_size,
        cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy mask tensor memo to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        return;
    }

    // init has mask input tensor
    std::vector<float> has_mask_tensor_values(1, 0.0);
    input_mem_size = static_cast<int32_t >(has_mask_tensor_values.size() * sizeof(float));
    cuda_status = cudaMemcpyAsync(
        device_memory.at(has_mask_input_binding.index()), has_mask_tensor_values.data(), input_mem_size,
        cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "copy has mask tensor value to gpu failed, error str: " << cudaGetErrorString(cuda_status);
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        return;
    }
    cudaStreamSynchronize(cuda_stream);
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds >(t_end - t_start).count();
    ctx->gpu_memo_cpy_time_consuming = t_cost;

    // do inference
    t_start = std::chrono::high_resolution_clock::now();
    context->setInputTensorAddress("image_embeddings", device_memory.at(image_embedding_binding.index()));
    context->setInputTensorAddress("point_coords", device_memory.at(point_coords_binding.index()));
    context->setInputTensorAddress("point_labels", device_memory.at(point_labels_binding.index()));
    context->setInputTensorAddress("mask_input", device_memory.at(mask_input_binding.index()));
    context->setInputTensorAddress("has_mask_input", device_memory.at(has_mask_input_binding.index()));
    context->setTensorAddress("low_res_masks", device_memory.at(low_res_masks_output_binding.index()));
    context->setTensorAddress("iou_predictions", device_memory.at(iou_predictions_output_binding.index()));
    if (!context->enqueueV3(cuda_stream)) {
        LOG(ERROR) << "excute input data for inference failed";
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        return;
    }

    std::vector<float> low_res_mask_data;
    low_res_mask_data.resize(low_res_masks_output_binding.volume());
    cuda_status = cudaMemcpyAsync(
        low_res_mask_data.data(),device_memory.at(low_res_masks_output_binding.index()),
        low_res_masks_output_binding.volume() * sizeof(float),cudaMemcpyDeviceToHost, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        return;
    }
    std::vector<float> iou_preds_data;
    iou_preds_data.resize(iou_predictions_output_binding.volume());
    cuda_status = cudaMemcpyAsync(
        iou_preds_data.data(), device_memory.at(iou_predictions_output_binding.index()),
        iou_predictions_output_binding.volume() * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "async copy output tensor back from device memory to host memory failed, error str: "
                   << cudaGetErrorString(cuda_status);
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        return;
    }
    cudaStreamSynchronize(cuda_stream);
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds >(t_end - t_start).count();
    ctx->model_inference_consuming = t_cost;

    // parse output mask
    t_start = std::chrono::high_resolution_clock::now();
    int best_mask_idx = static_cast<int>(
        std::distance(iou_preds_data.begin(), std::max_element(iou_preds_data.begin(), iou_preds_data.end())));
    decode_output_mask(low_res_mask_data, best_mask_idx, ctx->decoded_masks);
    ctx->pred_iou = iou_preds_data[best_mask_idx];
    ctx->stability_score = calculate_stability_score(ctx->decoded_masks);
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds >(t_end - t_start).count();
    ctx->decode_mask_time_consuming = t_cost;

    // restore worker queue
    t_start = std::chrono::high_resolution_clock::now();
    _m_decoder_queue.enqueue(std::move(decode_executor));
    t_end = std::chrono::high_resolution_clock::now();
    t_cost = std::chrono::duration_cast<std::chrono::milliseconds >(t_end - t_start).count();
    ctx->enqueue_thread_executor_time_consuming = t_cost;
}

/***
 *
 * @param input_image_size
 * @param n_points_per_side
 * @return
 */
std::vector<std::vector<cv::Point2f> > SamAmgDecoder::Impl::generate_prompt_points(
    const cv::Size &input_image_size, int n_points_per_side) {
    std::vector<std::vector<cv::Point2f> > prompt_points;
    auto w_step = static_cast<float>(input_image_size.width) / static_cast<float>(n_points_per_side);
    auto h_step = static_cast<float>(input_image_size.height) / static_cast<float>(n_points_per_side);
    for (auto start_y = h_step / 2.0f; start_y < static_cast<float>(input_image_size.height);) {
        for (auto start_x = w_step / 2.0f; start_x < static_cast<float>(input_image_size.width);) {
            prompt_points.push_back({cv::Point2f(start_x, start_y)});
            start_x += w_step;
        }
        start_y += h_step;
    }
    return prompt_points;
}

/***
 *
 * @param mask
 * @param mask_threshold
 * @param threshold_offset
 * @return
 */
float SamAmgDecoder::Impl::calculate_stability_score(const cv::Mat &mask) {
    float intersections = 0.0f;
    float unions = 0.0f;
    for (auto row = 0; row < mask.rows; ++row) {
        auto row_data = mask.ptr<float>(row);
        for (auto col = 0; col < mask.cols; ++col) {
            auto value = row_data[col];
            if (value > 0.0 + 1.0f) {
                intersections += 1.0f;
            }
            if (value > 0.0 - 1.0f) {
                unions += 1.0f;
            }
        }
    }
    return intersections / unions;
}

/***
 *
 * @param pred_masks
 * @param pred_ious
 * @param pred_stability_scores
 * @param point_coords
 * @param pred_iou_thresh
 * @param stability_score_thresh
 * @param stability_score_offset
 * @param box_nms_thresh
 * @param min_mask_region_area
 * @param amg_output
 */
void SamAmgDecoder::Impl::filter_output_masks(
    const std::vector<cv::Mat> &pred_masks, const std::vector<float> &pred_ious, const std::vector<float> &pred_stability_scores,
    const std::vector<cv::Point2f> &point_coords, const float pred_iou_thresh, const float stability_score_thresh,
    const float box_nms_thresh, const int min_mask_region_area,
    AmgMaskOutput &amg_output) {

    std::vector<cv::Mat> iou_threshed_masks;
    std::vector<float> iou_threshed_ious;
    std::vector<float> iou_threshed_stability_scores;
    std::vector<cv::Point2f> iou_threshed_point_coords;
    for (auto idx = 0; idx < pred_ious.size(); ++idx) {
        if (pred_ious[idx] >= pred_iou_thresh) {
            iou_threshed_masks.push_back(pred_masks[idx]);
            iou_threshed_ious.push_back(pred_ious[idx]);
            iou_threshed_stability_scores.push_back(pred_stability_scores[idx]);
            iou_threshed_point_coords.push_back(point_coords[idx]);
        }
    }

    // filter by stability score
    std::vector<cv::Mat> stability_threshed_masks;
    std::vector<float> stability_threshed_ious;
    std::vector<float> stability_scores;
    std::vector<cv::Point2f> stability_threshed_point_coords;
    for (auto idx = 0; idx < iou_threshed_stability_scores.size(); ++idx) {
        auto stability_score = iou_threshed_stability_scores[idx];
        if (stability_score >= stability_score_thresh) {
            stability_threshed_masks.push_back(iou_threshed_masks[idx]);
            stability_threshed_ious.push_back(iou_threshed_ious[idx]);
            stability_scores.push_back(stability_score);
            stability_threshed_point_coords.push_back(iou_threshed_point_coords[idx]);
        }
    }
    iou_threshed_masks.clear();
    iou_threshed_masks.shrink_to_fit();
    iou_threshed_ious.clear();
    iou_threshed_ious.shrink_to_fit();
    iou_threshed_stability_scores.clear();
    iou_threshed_stability_scores.shrink_to_fit();
    iou_threshed_point_coords.clear();
    iou_threshed_point_coords.shrink_to_fit();

    // threshold masks generate mask bboxes
    std::vector<cv::Rect> mask_bboxes;
    std::vector<int32_t> mask_areas;
    for (auto &mask : stability_threshed_masks) {
        int32_t tl_x = INT32_MAX;
        int32_t tl_y = INT32_MAX;
        int32_t rb_x = INT32_MIN;
        int32_t rb_y = INT32_MIN;
        int32_t mask_area = 0;
        for (int row = 0; row < mask.rows; ++row) {
            auto mask_data = mask.ptr<float>(row);
            for (int col = 0; col < mask.cols; ++col) {
                mask_data[col] = mask_data[col] > 0.0 ? 255.0 : 0.0;
                if (mask_data[col] == 255.0f) {
                    mask_area += 1;
                    if (row < tl_y) {
                        tl_y = row;
                    }
                    if (col < tl_x) {
                        tl_x = col;
                    }
                    if (row > rb_y) {
                        rb_y = row;
                    }
                    if (col > rb_x) {
                        rb_x = col;
                    }
                }
            }
        }
        mask_areas.push_back(mask_area);
        if (tl_x < rb_x && tl_y < rb_y) {
            auto mask_bbox = cv::Rect(tl_x, tl_y, rb_x - tl_x, rb_y - tl_y);
            mask_bboxes.push_back(mask_bbox);
        } else {
            mask_bboxes.emplace_back(0, 0, 0, 0);
        }
    }

    // nms mask bboxes
    std::vector<int> nms_keep_indices;
    cv::dnn::NMSBoxes(mask_bboxes, stability_threshed_ious, 0.0, box_nms_thresh, nms_keep_indices);
    std::vector<cv::Mat> nms_threshed_masks;
    std::vector<float> nms_threshed_ious;
    std::vector<cv::Rect> nms_threshed_mask_bboxes;
    std::vector<int32_t> nms_threshed_mask_areas;
    std::vector<float> nms_threshed_stability_scores;
    std::vector<cv::Point2f> nms_threshed_point_coords;
    for (auto &idx : nms_keep_indices) {
        nms_threshed_masks.push_back(stability_threshed_masks[idx]);
        nms_threshed_ious.push_back(stability_threshed_ious[idx]);
        nms_threshed_mask_bboxes.push_back(mask_bboxes[idx]);
        nms_threshed_mask_areas.push_back(mask_areas[idx]);
        nms_threshed_stability_scores.push_back(stability_scores[idx]);
        nms_threshed_point_coords.push_back(stability_threshed_point_coords[idx]);
    }
    mask_bboxes.clear();
    mask_bboxes.shrink_to_fit();
    mask_areas.clear();
    mask_areas.shrink_to_fit();
    stability_threshed_masks.clear();
    stability_threshed_masks.shrink_to_fit();
    stability_threshed_ious.clear();
    stability_threshed_ious.shrink_to_fit();
    stability_scores.clear();
    stability_scores.shrink_to_fit();
    stability_threshed_point_coords.clear();
    stability_threshed_point_coords.shrink_to_fit();

    // filter small region mask
    std::vector<cv::Mat> region_threshed_masks;
    std::vector<float> region_threshed_ious;
    std::vector<cv::Rect> region_threshed_mask_bboxes;
    std::vector<int32_t> region_threshed_mask_areas;
    std::vector<float> region_threshed_stability_scores;
    std::vector<cv::Point2f> region_threshed_point_coords;
    if (min_mask_region_area > 0) {
        for (auto idx = 0; idx < nms_threshed_masks.size(); ++idx) {
            cv::Mat labels;
            cv::Mat stats;
            cv::Mat centroids;
            auto mask = nms_threshed_masks[idx];
            auto components_count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
            for (auto i = 1; i < components_count; ++i) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area < min_mask_region_area) {
                    cv::Mat component_mask = (labels == i);
                    mask.setTo(0, component_mask);
                }
            }
            components_count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
            if (components_count > 0) {
                region_threshed_masks.push_back(mask);
                region_threshed_ious.push_back(nms_threshed_ious[idx]);
                region_threshed_mask_bboxes.push_back(nms_threshed_mask_bboxes[idx]);
                region_threshed_mask_areas.push_back(nms_threshed_mask_areas[idx]);
                region_threshed_stability_scores.push_back(nms_threshed_stability_scores[idx]);
                region_threshed_point_coords.push_back(nms_threshed_point_coords[idx]);
            }
        }
    } else {
        region_threshed_masks = nms_threshed_masks;
        region_threshed_ious = nms_threshed_ious;
        region_threshed_mask_bboxes = nms_threshed_mask_bboxes;
        region_threshed_mask_areas = nms_threshed_mask_areas;
        region_threshed_stability_scores = nms_threshed_stability_scores;
        region_threshed_point_coords = nms_threshed_point_coords;
    }

    // sort filter result according to mask area
    std::vector<size_t> sort_index(region_threshed_mask_areas.size());
    for (size_t i = 0; i < region_threshed_mask_areas.size(); ++i) {
        sort_index[i] = i;
    }
    std::sort(sort_index.begin(), sort_index.end(), [&region_threshed_mask_areas](int i, int j) {
        return region_threshed_mask_areas[i] > region_threshed_mask_areas[j];});

    std::vector<cv::Mat> sorted_masks(sort_index.size());
    std::vector<float> sorted_ious(sort_index.size());
    std::vector<cv::Rect> sorted_mask_bboxes(sort_index.size());
    std::vector<int32_t> sorted_mask_areas(sort_index.size());
    std::vector<float> sorted_stability_scores(sort_index.size());
    std::vector<cv::Point2f> sorted_point_coords(sort_index.size());

    for (size_t i = 0; i < sort_index.size(); ++i) {
        sorted_masks[i] = region_threshed_masks[sort_index[i]];
        sorted_ious[i] = region_threshed_ious[sort_index[i]];
        sorted_mask_bboxes[i] = region_threshed_mask_bboxes[sort_index[i]];
        sorted_mask_areas[i] = region_threshed_mask_areas[sort_index[i]];
        sorted_stability_scores[i] = region_threshed_stability_scores[sort_index[i]];
        sorted_point_coords[i] = region_threshed_point_coords[sort_index[i]];
    }

    amg_output.segmentations = sorted_masks;
    amg_output.bboxes = sorted_mask_bboxes;
    amg_output.preds_ious = sorted_ious;
    amg_output.areas = sorted_mask_areas;
    amg_output.preds_stability_scores = sorted_stability_scores;
    amg_output.point_coords = sorted_point_coords;
}

/***
 *
 */
SamAmgDecoder::SamAmgDecoder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
SamAmgDecoder::~SamAmgDecoder() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamAmgDecoder::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param image_embeddings
 * @param amg_output
 * @param points_per_side
 * @param pred_iou_thresh
 * @param stability_score_thresh
 * @param stability_score_offset
 * @param box_nms_thresh
 * @param min_mask_region_area
 * @return
 */
StatusCode SamAmgDecoder::decode_everything(
    const std::vector<float> &image_embeddings,
    AmgMaskOutput& amg_output, const int points_per_side, const float pred_iou_thresh, const float stability_score_thresh,
    const float box_nms_thresh, const int min_mask_region_area) {
    return _m_pimpl->decode_everything(
        image_embeddings, amg_output, points_per_side, pred_iou_thresh, stability_score_thresh,
        box_nms_thresh, min_mask_region_area);
}

/***
 *
 * @param ori_img_size
 */
void SamAmgDecoder::set_ori_image_size(const cv::Size &ori_img_size) {
    return _m_pimpl->set_ori_image_size(ori_img_size);
}

/***
 *
 * @param ori_img_size
 */
void SamAmgDecoder::set_encoder_input_size(const cv::Size &input_node_size){
    return _m_pimpl->set_encoder_input_size(input_node_size);
}

/***
 *
 * @return
 */
bool SamAmgDecoder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}