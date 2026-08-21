/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_amg_decoder.cpp
 * Date: 23-9-20
 ************************************************/

#include "sam_amg_decoder.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

#include "glog/logging.h"
#include "stl_container/blockingconcurrentqueue.h"
#include "workflow/WFFacilities.h"
#include "workflow/Workflow.h"

#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "models/backend/backend_config.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"

namespace jinq {
namespace models {

using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace segment_anything {

using jinq::models::backend::BackendConfig;
using jinq::models::backend::InferenceSession;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorInfo;

class SamAmgDecoder::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        // The queue stores non-owning session pointers; clear it before the
        // owning unique_ptr vector is destroyed.
        InferenceSession* session = nullptr;
        while (_m_decoder_queue.try_dequeue(session)) {
            // No explicit deletion: ownership remains in _m_sessions.
        }
        _m_sessions.clear();
    }

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init(const toml::table& cfg);

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
    // One session per parallel executor. InferenceSession is not thread
    // safe, so execution contexts are never shared across Workflow tasks.
    std::vector<std::unique_ptr<InferenceSession>> _m_sessions;

    // worker queue of non-owning session pointers
    moodycamel::BlockingConcurrentQueue<InferenceSession*> _m_decoder_queue;
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

    bool validate_session(const InferenceSession& session) const;

    const TensorInfo* find_info(
        const InferenceSession& session, const std::string& name) const;

    const TensorInfo* find_output(
        const InferenceSession& session, const std::string& name) const;

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
StatusCode SamAmgDecoder::Impl::init(const toml::table &cfg) {
    const toml::table* backend_table = cfg["amg_decoder_backend"].as_table();
    if (backend_table == nullptr) {
        LOG(ERROR) << "config section [SAM_AMG.amg_decoder_backend] missing";
        return StatusCode::MODEL_INIT_FAILED;
    }
    BackendConfig backend_config;
    std::string backend_err;
    if (!jinq::models::backend::parse_backend_table(
            *backend_table, &backend_config, &backend_err)) {
        LOG(ERROR) << "invalid sam amg decoder backend config: " << backend_err;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* params = cfg["params"].as_table();
    if (params == nullptr) {
        LOG(ERROR) << "config section [SAM_AMG.params] missing";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_decoder_queue_size = static_cast<int>((*params)["worker_queue_size"].value_or<int64_t>(4));
    _m_compute_thread_nums = static_cast<int>((*params)["compute_threads"].value_or<int64_t>(1));
    if (_m_decoder_queue_size <= 0 || _m_compute_thread_nums == 0) {
        LOG(ERROR) << "invalid sam amg decoder worker/compute thread counts";
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_sessions.reserve(static_cast<size_t>(_m_decoder_queue_size));
    for (int idx = 0; idx < _m_decoder_queue_size; ++idx) {
        std::string session_err;
        auto session = InferenceSession::create(backend_config, &session_err);
        if (session == nullptr) {
            LOG(ERROR) << "create sam amg decoder session failed: " << session_err;
            _m_sessions.clear();
            return StatusCode::MODEL_INIT_FAILED;
        }
        if (!validate_session(*session)) {
            _m_sessions.clear();
            return StatusCode::MODEL_INIT_FAILED;
        }
        _m_sessions.push_back(std::move(session));
    }
    for (auto& session : _m_sessions) {
        _m_decoder_queue.enqueue(session.get());
    }

    struct WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.compute_threads = _m_compute_thread_nums;
    WORKFLOW_library_init(&settings);

    _m_successfully_initialized = true;
    LOG(INFO) << "Successfully load sam amg decoder with "
              << _m_sessions.size() << " unified sessions";
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
        for (size_t idx = 0; idx < pwork->size(); ++idx) {
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
bool SamAmgDecoder::Impl::validate_session(const InferenceSession& session) const {
    const std::vector<std::string> required_inputs = {
        "image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input"};
    for (const auto& name : required_inputs) {
        const auto* info = find_info(session, name);
        if (info == nullptr || info->dtype != jinq::models::backend::DType::F32) {
            LOG(ERROR) << "sam amg decoder input missing or invalid: " << name;
            return false;
        }
        if (name != "point_coords" && name != "point_labels" && info->dynamic) {
            LOG(ERROR) << "sam amg decoder input '" << name << "' must be static";
            return false;
        }
    }
    const auto* low_res = find_output(session, "low_res_masks");
    const auto* iou = find_output(session, "iou_predictions");
    if (low_res == nullptr || iou == nullptr ||
        low_res->dtype != jinq::models::backend::DType::F32 ||
        iou->dtype != jinq::models::backend::DType::F32) {
        LOG(ERROR) << "sam amg decoder outputs low_res_masks/iou_predictions are invalid";
        return false;
    }
    return true;
}

const TensorInfo* SamAmgDecoder::Impl::find_info(
    const InferenceSession& session, const std::string& name) const {
    const auto iter = std::find_if(
        session.inputs().begin(), session.inputs().end(),
        [&name](const TensorInfo& info) { return info.name == name; });
    return iter == session.inputs().end() ? nullptr : &*iter;
}

const TensorInfo* SamAmgDecoder::Impl::find_output(
    const InferenceSession& session, const std::string& name) const {
    const auto iter = std::find_if(
        session.outputs().begin(), session.outputs().end(),
        [&name](const TensorInfo& info) { return info.name == name; });
    return iter == session.outputs().end() ? nullptr : &*iter;
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
    InferenceSession* session = nullptr;
    _m_decoder_queue.wait_dequeue(session);
    if (session == nullptr) {
        ctx->model_run_status = StatusCode::MODEL_INIT_FAILED;
        return;
    }

    auto fill = [](std::vector<NamedTensor>& inputs, const std::string& name,
                   const std::vector<float>& values,
                   const std::vector<int64_t>& shape) {
        NamedTensor named;
        named.name = name;
        named.tensor = Tensor::make<float>(shape);
        if (values.size() != static_cast<size_t>(named.tensor.element_count())) {
            return false;
        }
        std::memcpy(named.tensor.buffer.data(), values.data(), named.tensor.byte_size());
        inputs.push_back(std::move(named));
        return true;
    };

    const auto& sess_inputs = session->inputs();
    auto info = [&sess_inputs](const std::string& name) {
        const auto iter = std::find_if(
            sess_inputs.begin(), sess_inputs.end(),
            [&name](const TensorInfo& item) { return item.name == name; });
        return iter == sess_inputs.end() ? nullptr : &*iter;
    };

    std::vector<float> points = {point.x, point.y, 0.0f, 0.0f};
    std::vector<float> labels = {1.0f, -1.0f};
    const auto* point_coords_info = info("point_coords");
    const auto* point_labels_info = info("point_labels");
    if (!point_coords_info->dynamic && !point_labels_info->dynamic) {
        while (static_cast<int64_t>(labels.size()) < point_coords_info->shape[1]) {
            points.push_back(0.0f);
            points.push_back(0.0f);
            labels.push_back(-1.0f);
        }
    }
    std::vector<NamedTensor> inputs;
    const auto* embedding_info = info("image_embeddings");
    const auto* mask_info = info("mask_input");
    const auto* has_mask_info = info("has_mask_input");
    if (!fill(inputs, "image_embeddings", image_embeddings, embedding_info->shape) ||
        !fill(
            inputs, "point_coords", points,
            {1, static_cast<int64_t>(labels.size()), 2}) ||
        !fill(inputs, "point_labels", labels, {1, static_cast<int64_t>(labels.size())}) ||
        !fill(
            inputs, "mask_input",
            std::vector<float>(
                static_cast<size_t>(jinq::models::backend::shape_volume(mask_info->shape))),
            mask_info->shape) ||
        !fill(
            inputs, "has_mask_input",
            std::vector<float>(
                static_cast<size_t>(
                    jinq::models::backend::shape_volume(has_mask_info->shape))),
            has_mask_info->shape)) {
        LOG(ERROR) << "create sam amg decoder input tensors failed";
        ctx->model_run_status = StatusCode::MODEL_RUN_SESSION_FAILED;
        _m_decoder_queue.enqueue(session);
        return;
    }

    std::vector<NamedTensor> outputs;
    ctx->model_run_status = session->run(inputs, outputs);
    if (ctx->model_run_status != StatusCode::OK) {
        _m_decoder_queue.enqueue(session);
        return;
    }

    const auto find_named = [&outputs](const std::string& name) {
        const auto iter = std::find_if(
            outputs.begin(), outputs.end(),
            [&name](const NamedTensor& item) { return item.name == name; });
        return iter == outputs.end() ? nullptr : &*iter;
    };
    const auto* low_res = find_named("low_res_masks");
    const auto* iou = find_named("iou_predictions");
    if (low_res == nullptr || iou == nullptr ||
        low_res->tensor.shape.size() != 4 ||
        iou->tensor.element_count() <= 0) {
        LOG(ERROR) << "sam amg decoder outputs are invalid";
        ctx->model_run_status = StatusCode::MODEL_EMPTY_OUTPUT;
        _m_decoder_queue.enqueue(session);
        return;
    }

    const auto* iou_data = iou->tensor.template data<float>();
    const int best_idx = static_cast<int>(
        std::distance(iou_data, std::max_element(iou_data, iou_data + iou->tensor.element_count())));
    const auto& mask_tensor = low_res->tensor;
    const auto* mask_data = mask_tensor.template data<float>() +
                             static_cast<int64_t>(best_idx) * mask_tensor.shape[2] * mask_tensor.shape[3];
    std::vector<float> low_res_mask_data(
        mask_data, mask_data + mask_tensor.shape[2] * mask_tensor.shape[3]);
    decode_output_mask(low_res_mask_data, 0, ctx->decoded_masks);
    ctx->pred_iou = iou_data[best_idx];
    ctx->stability_score = calculate_stability_score(ctx->decoded_masks);
    ctx->point_coord = point;
    _m_decoder_queue.enqueue(session);
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
    for (size_t idx = 0; idx < pred_ious.size(); ++idx) {
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
    for (size_t idx = 0; idx < iou_threshed_stability_scores.size(); ++idx) {
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
        for (size_t idx = 0; idx < nms_threshed_masks.size(); ++idx) {
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
StatusCode SamAmgDecoder::init(const toml::table &cfg) {
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
