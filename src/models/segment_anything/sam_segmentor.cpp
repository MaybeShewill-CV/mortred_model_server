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

    // model session options
    Ort::SessionOptions _m_encoder_sess_options;
    Ort::SessionOptions _m_decoder_sess_options;

    // model session
    std::unique_ptr<Ort::Session> _m_encoder_sess;
    std::unique_ptr<Ort::Session> _m_decoder_sess;

    // model input/output shape info
    std::vector<int64_t> _m_encoder_input_shape;
    std::vector<int64_t> _m_encoder_output_shape;

    // init flag
    bool _m_successfully_init_model = false;
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode SamSegmentor::Impl::init(const decltype(toml::parse("")) &cfg) {
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
    _m_encoder_sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_encoder_device_id;
        _m_encoder_sess_options.AppendExecutionProvider_CUDA(cuda_options);
    }
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, ""};
    _m_encoder_sess = std::make_unique<Ort::Session>(
                          env, _m_encoder_model_path.c_str(), _m_encoder_sess_options);
    if (_m_encoder_sess->GetInputCount() != 1 || _m_encoder_sess->GetOutputCount() != 1) {
        std::string err_msg = fmt::format(
                                  "invalid input/output count, input count should be 1 rather than {}, output count should be 1 rather than {}",
                                  _m_encoder_sess->GetInputCount(), _m_encoder_sess->GetOutputCount());
        LOG(ERROR) << err_msg;
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_encoder_input_shape = _m_encoder_sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    _m_encoder_output_shape = _m_encoder_sess->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    use_gpu = false;
    LOG(INFO) << "... successfully load sam encoder model";

    // init sam decoder configs
    auto sam_decoder_cfg = cfg.at("SAM_VIT_ENCODER");
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
    _m_decoder_sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = _m_decoder_device_id;
        _m_decoder_sess_options.AppendExecutionProvider_CUDA(cuda_options);
    }
    _m_decoder_sess = std::make_unique<Ort::Session>(
                          env, _m_decoder_model_path.c_str(), _m_decoder_sess_options);
    LOG(INFO) << "... successfully load sam decoder model";

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

    if (_m_encoder_sess->GetInputCount() == 1) {
        return StatusCode::OJBK;
    } else {
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
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
 * @return
 */
bool SamSegmentor::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}