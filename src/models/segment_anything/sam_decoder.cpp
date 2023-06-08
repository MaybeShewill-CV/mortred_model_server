/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamDecoder.cpp
 * Date: 23-6-7
 ************************************************/

#include "sam_decoder.h"

#include "glog/logging.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "common/time_stamp.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace segment_anything {

class SamDecoder::Impl {
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
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param image_embeddings
     * @param bboxes
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<cv::Rect2f>& bboxes,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param ori_image_size
     */
    void set_ori_image_size(const cv::Size& ori_image_size) {
        _m_ori_image_size = ori_image_size;
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

    // origin image size
    cv::Size _m_ori_image_size;

    // init flag
    bool _m_successfully_init_model = false;

  private:
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
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode SamDecoder::Impl::init(const decltype(toml::parse("")) &cfg) {
    // ort env and memo info
    _m_env = {ORT_LOGGING_LEVEL_WARNING, ""};

    // init sam decoder configs
    auto sam_decoder_cfg = cfg.at("SAM_VIT_DECODER");
    _m_model_path = sam_decoder_cfg["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "sam decoder model file path: " << _m_model_path << " not exists";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_model_device = sam_decoder_cfg["compute_backend"].as_string();
    if (std::strcmp(_m_model_device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_device_id = sam_decoder_cfg["gpu_device_id"].as_integer();
    }
    _m_thread_nums = sam_decoder_cfg["model_threads_num"].as_integer();
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

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load sam decoder model";
    return StatusCode::OJBK;
}

/***
 *
 * @param image_embeddings
 * @param bboxes
 * @param predicted_masks
 * @return
 */
jinq::common::StatusCode SamDecoder::Impl::decode(
    const std::vector<float>& image_embeddings,
    const std::vector<cv::Rect2f>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    // decoder masks
    auto t_start = Timestamp::now();
    for (auto& bbox : bboxes) {
        cv::Mat out_mask;
        auto status_code= get_mask(image_embeddings, bbox, out_mask);
        if (status_code != StatusCode::OJBK) {
            return status_code;
        }
        predicted_masks.push_back(out_mask);
    }
    auto t_cost = Timestamp::now() - t_start;
//    LOG(INFO) << "decode finished cost time: " << t_cost;

    return StatusCode::OJBK;
}

/***
 *
 * @param decoder_inputs
 * @param bbox
 * @param points
 * @param out_mask
 * @return
 */
StatusCode SamDecoder::Impl::get_mask(
    const std::vector<float>& image_embeddings,
    const cv::Rect2f &bbox,
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
 */
SamDecoder::SamDecoder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
SamDecoder::~SamDecoder() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode SamDecoder::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param image_embeddings
 * @param bboxes
 * @param predicted_masks
 * @return
 */
jinq::common::StatusCode SamDecoder::decode(
    const std::vector<float>& image_embeddings,
    const std::vector<cv::Rect2f>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    return _m_pimpl->decode(image_embeddings, bboxes, predicted_masks);
}

/***
 *
 * @param ori_img_size
 */
void SamDecoder::set_ori_image_size(const cv::Size &ori_img_size) {
    return _m_pimpl->set_ori_image_size(ori_img_size);
}

/***
 *
 * @return
 */
bool SamDecoder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}