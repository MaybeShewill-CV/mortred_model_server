/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamSegmentor.cpp
 * Date: 23-5-26
 ************************************************/

#include "sam_segmentor.h"

#include "glog/logging.h"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "fmt/format.h"

#include "common/file_path_util.h"
#include "common/cv_utils.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;

namespace segment_anything {

class SamSegmentor::Impl {
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
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<cv::Rect>& bboxes,
        const std::vector<cv::Point>& points,
        const std::vector<int>& point_labels,
        cv::Mat& predicted_mask);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    jinq::common::StatusCode get_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

private:
    // model file path
    std::string _m_encoder_model_path;
    std::string _m_decoder_model_path;

    // model compute thread nums
    uint16_t _m_encoder_thread_nums = 1;
    uint16_t _m_decoder_thread_nums = 1;

    // model backend device
    std::string _m_encoder_model_device;
    std::string _m_decoder_model_device;

    // model backend device id
    uint8_t _m_encoder_device_id = 0;
    uint8_t _m_decoder_device_id = 0;

    // model input/output names
    std::vector<const char*> _m_encoder_input_names;
    std::vector<const char*> _m_encoder_output_names;
    std::vector<const char*> _m_decoder_input_names;
    std::vector<const char*> _m_decoder_output_names;

    // model envs
    Ort::Env _m_env;
    Ort::MemoryInfo _m_memo_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // model session options
    Ort::SessionOptions _m_encoder_sess_options;
    Ort::SessionOptions _m_decoder_sess_options;

    // model session
    std::unique_ptr<Ort::Session> _m_encoder_sess;
    std::unique_ptr<Ort::Session> _m_decoder_sess;

    // model input/output shape info
    std::vector<int64_t> _m_encoder_input_shape;
    std::vector<int64_t> _m_encoder_output_shape;

    // origin image size
    cv::Size _m_ori_image_size;

    // init flag
    bool _m_successfully_init_model = false;

  private:
    /***
     *
     * @param input_image
     * @return
     */
    cv::Mat preprocess_image(const cv::Mat& input_image);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    StatusCode encode_image_embeddings(const cv::Mat& input_image, Ort::Value& image_embeddings);

    /***
     *
     * @param image_embeddings
     * @param bboxes
     * @param points
     * @param point_labels
     * @param out_masks
     * @return
     */
     StatusCode get_masks(
        Ort::Value& image_embeddings,
        const std::vector<cv::Rect>& bboxes,
        const std::vector<cv::Point>& points,
        const std::vector<int>& point_labels,
        std::vector<cv::Mat>& out_masks);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode SamSegmentor::Impl::init(const decltype(toml::parse("")) &cfg) {
    // ort env and memo info
    _m_env = {ORT_LOGGING_LEVEL_WARNING, ""};

    // init sam encoder configs
    auto sam_encoder_cfg = cfg.at("SAM_VIT_ENCODER");
    _m_encoder_model_path = sam_encoder_cfg["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(_m_encoder_model_path)) {
        LOG(ERROR) << "sam encoder model file path: " << _m_encoder_model_path << " not exists";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    bool use_gpu = false;
    _m_encoder_model_device = sam_encoder_cfg["compute_backend"].as_string();
    if (std::strcmp(_m_encoder_model_device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_encoder_device_id = sam_encoder_cfg["gpu_device_id"].as_integer();
    }
    _m_encoder_thread_nums = sam_encoder_cfg["model_threads_num"].as_integer();
    _m_encoder_sess_options.SetIntraOpNumThreads(_m_encoder_thread_nums);
    _m_encoder_sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//    _m_encoder_sess_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_encoder_device_id;
        _m_encoder_sess_options.AppendExecutionProvider_CUDA(cuda_options);
    }
    _m_encoder_sess = std::make_unique<Ort::Session>(
        _m_env, _m_encoder_model_path.c_str(), _m_encoder_sess_options);
    if (_m_encoder_sess->GetInputCount() != 1 || _m_encoder_sess->GetOutputCount() != 1) {
        std::string err_msg = fmt::format(
            "invalid input/output count, input count should be 1 rather than {}, output count should be 1 rather than {}",
            _m_encoder_sess->GetInputCount(), _m_encoder_sess->GetOutputCount());
        LOG(ERROR) << err_msg;
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_encoder_input_names = {"input_image"};
    _m_encoder_output_names = {"image_embeddings"};
    _m_encoder_input_shape = _m_encoder_sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    _m_encoder_output_shape = _m_encoder_sess->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (_m_encoder_input_shape.size() != 4 || _m_encoder_output_shape.size() != 4) {
        LOG(ERROR) << "invalid encoder input/output node shape";
        return StatusCode::MODEL_INIT_FAILED;
    }
    use_gpu = false;
    LOG(INFO) << "... successfully load sam encoder model";

    // init sam decoder configs
    auto sam_decoder_cfg = cfg.at("SAM_VIT_DECODER");
    _m_decoder_model_path = sam_decoder_cfg["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(_m_decoder_model_path)) {
        LOG(ERROR) << "sam decoder model file path: " << _m_decoder_model_path << " not exists";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_decoder_model_device = sam_decoder_cfg["compute_backend"].as_string();
    if (std::strcmp(_m_decoder_model_device.c_str(), "cuda") == 0) {
        use_gpu = true;
        _m_decoder_device_id = sam_decoder_cfg["gpu_device_id"].as_integer();
    }
    _m_decoder_thread_nums = sam_decoder_cfg["model_threads_num"].as_integer();
    _m_decoder_sess_options.SetIntraOpNumThreads(_m_decoder_thread_nums);
    _m_decoder_sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//    _m_decoder_sess_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_decoder_device_id;
        _m_decoder_sess_options.AppendExecutionProvider_CUDA(cuda_options);
    }
    _m_decoder_sess = std::make_unique<Ort::Session>(
        _m_env, _m_decoder_model_path.c_str(), _m_decoder_sess_options);
    _m_decoder_input_names = {
        "image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"};
    _m_decoder_output_names = {"masks", "iou_predictions", "low_res_masks"};

    for (int idx = 0; idx < _m_decoder_input_names.size(); ++idx) {
        auto shape = _m_decoder_sess->GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        auto t_type = _m_decoder_sess->GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetElementType();
        auto tmp_info = fmt::format("decoder input name: {}, data type: {}, shape: ", _m_decoder_input_names[idx], t_type);
        for (auto& s : shape) {
            tmp_info += std::to_string(s) + " ";
        }
        LOG(INFO) << tmp_info;
    }

    for (int idx = 0; idx < _m_decoder_output_names.size(); ++idx) {
        auto shape = _m_decoder_sess->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        auto t_type = _m_decoder_sess->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetElementType();
        auto tmp_info = fmt::format("decoder output name: {}, data type: {}, shape: ", _m_decoder_output_names[idx], t_type);
        for (auto& s : shape) {
            tmp_info += std::to_string(s) + " ";
        }
        LOG(INFO) << tmp_info;
    }

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load sam model";
    return StatusCode::OJBK;
}

/***
 *
 * @param input_image
 * @param bboxes
 * @param points
 * @param point_labels
 * @param predicted_mask
 * @return
 */
jinq::common::StatusCode SamSegmentor::Impl::predict(
    const cv::Mat& input_image,
    const std::vector<cv::Rect>& bboxes,
    const std::vector<cv::Point>& points,
    const std::vector<int>& point_labels,
    cv::Mat& predicted_mask) {
    // fetch origin image size
    _m_ori_image_size = input_image.size();

    // encode image embeddings
    Ort::Value image_embeddings(nullptr);
    auto status= encode_image_embeddings(input_image, image_embeddings);
    if (status != StatusCode::OJBK) {
        LOG(INFO) << "encoding image embeddings failed, status code: " << status;
        return status;
    }
    LOG(INFO) << image_embeddings.GetTensorMutableData<float>()[0] << " "
              << image_embeddings.GetTensorMutableData<float>()[1] << " "
              << image_embeddings.GetTensorMutableData<float>()[2] << " "
              << image_embeddings.GetTensorMutableData<float>()[3] << " ";
    LOG(INFO) << "embedding finished";

    // decoder masks
    std::vector<cv::Mat> masks;
    status = get_masks(image_embeddings, bboxes, points, point_labels, masks);
    if (status != StatusCode::OJBK) {
        LOG(INFO) << "decode sam-masks failed, status code: " << status;
        return status;
    }
    LOG(INFO) << "decode finished";

    cv::imwrite("fuck.png", masks[0]);

    if (_m_encoder_sess->GetInputCount() == 1) {
        return StatusCode::OJBK;
    } else {
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
jinq::common::StatusCode SamSegmentor::Impl::get_embedding(
    const cv::Mat &input_image,
    std::vector<float> &image_embeddings) {
    // encode images
    Ort::Value img_embeds(nullptr);
    auto status= encode_image_embeddings(input_image, img_embeds);
    if (status != StatusCode::OJBK) {
        return status;
    }

    // fetch embedding
    auto embeds_size = std::accumulate(std::begin(_m_encoder_output_shape), std::end(_m_encoder_output_shape), 1, std::multiplies<int64_t>());
    image_embeddings.resize(embeds_size);
    auto img_embeds_val = img_embeds.GetTensorMutableData<float>();
    for (auto idx = 0; idx < embeds_size; ++idx) {
        image_embeddings[idx] = img_embeds_val[idx];
    }
    return StatusCode::OJBK;

}

/***
 *
 * @param input_image
 * @return
 */
cv::Mat SamSegmentor::Impl::preprocess_image(const cv::Mat &input_image) {

    auto input_node_h = static_cast<int>(_m_encoder_input_shape[2]);
    auto input_node_w = static_cast<int>(_m_encoder_input_shape[3]);

    cv::Mat result;
    cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);
    cv::resize(input_image, result,cv::Size(input_node_w, input_node_h));
    result.convertTo(result, CV_32FC3);

    cv::subtract(result, cv::Scalar(123.675, 116.28, 103.53), result);
    cv::divide(result, cv::Scalar(58.395, 57.12, 57.375), result);

    return result;
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode SamSegmentor::Impl::encode_image_embeddings(const cv::Mat &input_image, Ort::Value& image_embeddings) {
    // preprocess image
    auto preprocessed_image = preprocess_image(input_image);
    auto input_tensor_values = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run encoder
    auto input_tensor = Ort::Value::CreateTensor<float>(
        _m_memo_info, input_tensor_values.data(), input_tensor_values.size(),
        _m_encoder_input_shape.data(), _m_encoder_input_shape.size());
    if (!input_tensor.IsTensor() || !input_tensor.HasValue()) {
        LOG(ERROR) << "create input tensor for vit encoder failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    auto output_tensors =
        _m_encoder_sess->Run(
            Ort::RunOptions{nullptr},
            _m_encoder_input_names.data(),
            &input_tensor, 1,
            _m_encoder_output_names.data(), 1
            );
    if (output_tensors.size() != 1 || !output_tensors.front().IsTensor()) {
        LOG(ERROR) << "run vit encoder failed, output tensor size: " << output_tensors.size()
                   << ", front elem is tensor: " << output_tensors.front().IsTensor();
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    image_embeddings = std::move(output_tensors.front());

    return StatusCode::OJBK;
}

/***
 *
 * @param image_embeddings
 * @param bboxes
 * @param points
 * @param point_labels
 * @param out_masks
 * @return
 */
StatusCode SamSegmentor::Impl::get_masks(
    Ort::Value &image_embeddings,
    const std::vector<cv::Rect> &bboxes,
    const std::vector<cv::Point> &points,
    const std::vector<int> &point_labels,
    std::vector<cv::Mat> &out_masks) {
    // init decoder input tensor
    std::vector<Ort::Value> decoder_input_tensor;

    // init embeddings
    decoder_input_tensor.push_back(std::move(image_embeddings));

    // init points tensor and label tensor
    std::vector<float> total_points;
    std::vector<float> total_labels;
    for (auto& pt : points) {
        auto x = static_cast<float>(pt.x);
        auto y = static_cast<float>(pt.y);
        total_points.push_back(x);
        total_points.push_back(y);
        total_labels.push_back(1.0);
    }
    for (auto& rect : bboxes) {
        // top left point
        auto tl_pt = rect.tl();
        auto tl_x = static_cast<float>(tl_pt.x);
        auto tl_y = static_cast<float>(tl_pt.y);
        total_points.push_back(tl_x);
        total_points.push_back(tl_y);
        total_labels.push_back(2.0);
        // bottom right point
        auto br_x = static_cast<float>(tl_x + static_cast<float>(rect.width));
        auto br_y = static_cast<float>(tl_y + static_cast<float>(rect.height));
        total_points.push_back(br_x);
        total_points.push_back(br_y);
        total_labels.push_back(3.0);
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
        static_cast<float>(_m_ori_image_size.width), static_cast<float>(_m_ori_image_size.height)};
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
        Ort::RunOptions{nullptr}, _m_decoder_input_names.data(), decoder_input_tensor.data(),
        decoder_input_tensor.size(), _m_decoder_output_names.data(), _m_decoder_output_names.size());
    auto mask_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    for (auto& s : mask_shape) {
        LOG(INFO) << s;
    }

    auto masks_preds_value = output_tensors[0].GetTensorMutableData<float>();
    auto masks_preds_iou_value = output_tensors[1].GetTensorMutableData<float>();
    for (int idx = 0; idx < 1; ++idx) {
        LOG(INFO) << "preds iou: " << masks_preds_iou_value[idx];
    }
    LOG(INFO) << fmt::format(
        "mask value: {} {} {} {}", masks_preds_value[0], masks_preds_value[1], masks_preds_value[2], masks_preds_value[3]);

    cv::Mat mask(cv::Size(1024, 1024), CV_8UC1);
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            mask.at<uchar>(row, col) = masks_preds_value[row * mask.cols + col] > 0 ? 255 : 0;
        }
    }
    out_masks.push_back(mask);

    return StatusCode::OJBK;
}

/***
 *
 */
SamSegmentor::SamSegmentor() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
SamSegmentor::~SamSegmentor() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode SamSegmentor::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_image
 * @param bboxes
 * @param points
 * @param point_labels
 * @param predicted_mask
 * @return
 */
jinq::common::StatusCode SamSegmentor::predict(
    const cv::Mat& input_image,
    const std::vector<cv::Rect>& bboxes,
    const std::vector<cv::Point>& points,
    const std::vector<int>& point_labels,
    cv::Mat& predicted_mask) {
    return _m_pimpl->predict(input_image, bboxes, points, point_labels, predicted_mask);
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
jinq::common::StatusCode SamSegmentor::get_embedding(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
   return _m_pimpl->get_embedding(input_image, image_embeddings);
}

/***
 *
 * @return
 */
bool SamSegmentor::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}