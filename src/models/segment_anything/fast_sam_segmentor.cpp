/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: FastFastSamSegmentor.cpp
 * Date: 23-6-29
 ************************************************/

#include "fast_sam_segmentor.h"

#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

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

class FastSamSegmentor::Impl {
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
        std::vector<cv::Mat>& predicted_masks);

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

    // model input/output names
    std::string _m_input_name;
    std::string _m_output_0_name;
    std::string _m_output_1_name;

    // model session
    std::unique_ptr<MNN::Interpreter> _m_net;
    MNN::Session* _m_session = nullptr;
    MNN::Tensor* _m_input_tensor = nullptr;
    MNN::Tensor* _m_output_tensor_0 = nullptr;
    MNN::Tensor* _m_output_tensor_1 = nullptr;

    // model input/output shape info
    std::vector<int> _m_input_shape;
    std::vector<int> _m_output_0_shape;
    std::vector<int> _m_output_1_shape;

    // init flag
    bool _m_successfully_init_model = false;

  private:
    /***
     *
     * @param bboxes
     * @return
     */
    std::vector<cv::Rect2f> transform_bboxes(const std::vector<cv::Rect>& bboxes, int target_size=1024) const;

    /***
     *
     * @param input_image
     * @return
     */
    static cv::Mat preprocess_image(const cv::Mat& input_image);

    /***
     *
     */
    void postprocess();
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode FastSamSegmentor::Impl::init(const decltype(toml::parse("")) &cfg) {
    // init sam encoder configs
    auto cfg_content = cfg.at("FAST_SAM");
    _m_model_path = cfg_content["model_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(_m_model_path)) {
        LOG(ERROR) << "fast sam model file path: " << _m_model_path << " not exists";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_path.c_str()));
    if (_m_net == nullptr) {
        LOG(ERROR) << "Create Interpreter failed, model file path: " << _m_model_path;
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << R"(Config file parse error, doesn't not have field "model_threads_nums", use default value 4)";
        _m_thread_nums = 4;
    } else {
        _m_thread_nums = static_cast<int>(cfg_content.at("model_threads_num").as_integer());
    }

    // init session
    MNN::ScheduleConfig mnn_config;
    if (!cfg_content.contains("compute_backend")) {
        LOG(WARNING) << "Config doesn\'t have compute_backend field default cpu";
        mnn_config.type = MNN_FORWARD_CPU;
    } else {
        std::string compute_backend = cfg_content.at("compute_backend").as_string();

        if (std::strcmp(compute_backend.c_str(), "cuda") == 0) {
            mnn_config.type = MNN_FORWARD_CUDA;
        } else if (std::strcmp(compute_backend.c_str(), "cpu") == 0) {
            mnn_config.type = MNN_FORWARD_CPU;
        } else {
            LOG(WARNING) << "not supported compute backend use default cpu instead";
            mnn_config.type = MNN_FORWARD_CPU;
        }
    }
    mnn_config.numThread = _m_thread_nums;
    MNN::BackendConfig backend_config;
    if (!cfg_content.contains("backend_precision_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_precision_mode field default Precision_Normal";
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(cfg_content.at("backend_precision_mode").as_integer());
    }
    if (!cfg_content.contains("backend_power_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_power_mode field default Power_Normal";
        backend_config.power = MNN::BackendConfig::Power_Normal;
    } else {
        backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(cfg_content.at("backend_power_mode").as_integer());
    }
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);
    if (_m_session == nullptr) {
        LOG(ERROR) << "Create Session failed, model file path: " << _m_model_path;
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_name = "images";
    _m_output_0_name = "output0";
    _m_output_1_name = "output1";

    _m_input_tensor = _m_net->getSessionInput(_m_session, _m_input_name.c_str());
    _m_output_tensor_0 = _m_net->getSessionOutput(_m_session, _m_output_0_name.c_str());
    _m_output_tensor_1 = _m_net->getSessionOutput(_m_session, _m_output_1_name.c_str());

    _m_input_shape = _m_input_tensor->shape();
    _m_output_0_shape = _m_output_tensor_0->shape();
    _m_output_1_shape = _m_output_tensor_1->shape();

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load sam vit encoder";
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
jinq::common::StatusCode FastSamSegmentor::Impl::predict(
    const cv::Mat& input_image,
    const std::vector<cv::Rect>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    // check input image
    if (!input_image.data || input_image.empty()) {
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // preprocess image
    auto preprocessed_image = preprocess_image(input_image);
    auto input_image_nchw_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run session
    auto input_tensor_host = MNN::Tensor(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    ::memcpy(input_tensor_host.host<float>(), input_image_nchw_data.data(), input_tensor_host.elementSize());
    _m_net->runSession(_m_session);

    // post process decode mask
    postprocess();


    return StatusCode::OK;
}

/***
 *
 * @param bboxes
 * @param target_size
 * @return
 */
std::vector<cv::Rect2f> FastSamSegmentor::Impl::transform_bboxes(const std::vector<cv::Rect> &bboxes, int target_size) const {
    std::vector<cv::Rect2f> transformed_bboxes;
    return transformed_bboxes;
}

/***
 *
 * @param input_image
 * @return
 */
cv::Mat FastSamSegmentor::Impl::preprocess_image(const cv::Mat &input_image) {
    cv::Mat result;
    cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);

    result.convertTo(result, CV_32FC3);

    cv::resize(result, result, cv::Size(640, 640));

    cv::divide(result, 255.0, result);

    return result;
}

void FastSamSegmentor::Impl::postprocess() {

    auto batch_size = 1;
    auto nc = 1;
    auto nm = 32;

    std::vector<std::vector<float> > threshed_preds;
    LOG(INFO) << _m_output_0_shape.size();
    LOG(INFO) << _m_output_0_shape[0] << " "
              << _m_output_0_shape[1] << " "
              << _m_output_0_shape[2];

    auto output_tensor_0_host = MNN::Tensor(_m_output_tensor_0, _m_output_tensor_0->getDimensionType());
    _m_output_tensor_0->copyToHostTensor(&output_tensor_0_host);
    LOG(INFO) << output_tensor_0_host.shape().size();
    LOG(INFO) << output_tensor_0_host.shape()[0] << " "
              << output_tensor_0_host.shape()[1] << " "
              << output_tensor_0_host.shape()[2];
    auto* output_tensor_0_data = output_tensor_0_host.host<float>();
    LOG(INFO) << output_tensor_0_data[0] << " "
              << output_tensor_0_data[1] << " "
              << output_tensor_0_data[2] << " "
              << output_tensor_0_data[3] << " "
              << output_tensor_0_data[4];

    for (auto bboxes_nums = 0; bboxes_nums < _m_output_0_shape[2]; ++bboxes_nums) {
        std::vector<float> bbox_info;
        bbox_info.resize(_m_output_0_shape[1]);
        for (auto idx = 0; idx < _m_output_0_shape[1]; ++idx) {
            auto info_idx = idx * _m_output_0_shape[1] + bboxes_nums;
            auto info_value = output_tensor_0_data[info_idx];
            bbox_info[idx] = info_value;
        }
        if (bbox_info[4] <= 0.9) {
//            LOG(INFO) << bbox_info[4];
            continue ;
        } else {
//            LOG(INFO) << bbox_info[3];
            threshed_preds.push_back(bbox_info);
        }
    }
    LOG(INFO) << threshed_preds.size();
}

/***
 *
 */
FastSamSegmentor::FastSamSegmentor() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
FastSamSegmentor::~FastSamSegmentor() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode FastSamSegmentor::init(const decltype(toml::parse("")) &cfg) {
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
jinq::common::StatusCode FastSamSegmentor::predict(
    const cv::Mat& input_image,
    const std::vector<cv::Rect>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    return _m_pimpl->predict(input_image, bboxes, predicted_masks);
}

/***
 *
 * @return
 */
bool FastSamSegmentor::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}