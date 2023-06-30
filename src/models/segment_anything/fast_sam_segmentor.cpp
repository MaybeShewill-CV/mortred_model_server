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
    LOG(INFO) << input_image_nchw_data[0] << " "
              << input_image_nchw_data[1] << " "
              << input_image_nchw_data[2] << " "
              << input_image_nchw_data[3] << " "
              << input_image_nchw_data[4];
    LOG(INFO) << input_image_nchw_data.size();

    // run session
    auto input_tensor_host = MNN::Tensor(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    ::memcpy(input_tensor_host.host<float>(), input_image_nchw_data.data(), input_tensor_host.size());
    _m_input_tensor->copyFromHostTensor(&input_tensor_host);

    _m_net->runSession(_m_session);

//    auto output_tensor_0_host = MNN::Tensor(_m_output_tensor_0, _m_output_tensor_0->getDimensionType());
//    _m_output_tensor_0->copyToHostTensor(&output_tensor_0_host);
//    auto* output_tensor_0_data = output_tensor_0_host.host<float>();
//
//    LOG(INFO) << output_tensor_0_data[0] << " "
//              << output_tensor_0_data[1] << " "
//              << output_tensor_0_data[2] << " "
//              << output_tensor_0_data[3] << " "
//              << output_tensor_0_data[4];
//
//    auto output_tensor_1_host = MNN::Tensor(_m_output_tensor_1, _m_output_tensor_0->getDimensionType());
//    _m_output_tensor_1->copyToHostTensor(&output_tensor_1_host);
//    auto* output_tensor_1_data = output_tensor_1_host.host<float>();
//
//    LOG(INFO) << output_tensor_1_data[0] << " "
//              << output_tensor_1_data[1] << " "
//              << output_tensor_1_data[2] << " "
//              << output_tensor_1_data[3] << " "
//              << output_tensor_1_data[4];

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

    cv::resize(input_image, result, cv::Size(640, 640));

    cv::cvtColor(result, result, cv::COLOR_BGR2RGB);

    result.convertTo(result, CV_32FC3);

    cv::divide(result, 255.0, result);

    return result;
}

void FastSamSegmentor::Impl::postprocess() {

    auto output_tensor_0_host = MNN::Tensor(_m_output_tensor_0, _m_output_tensor_0->getDimensionType());
    _m_output_tensor_0->copyToHostTensor(&output_tensor_0_host);
    auto* output_tensor_0_data = output_tensor_0_host.host<float>();

    auto bbox_info_len = _m_output_0_shape[1];
    auto bbox_nums = _m_output_0_shape[2];

    struct bbox_ {
        cv::Rect2f bbox;
        float score = 0.0;
        std::vector<float> masks;
        int class_id = 0;
    };

    std::vector<std::vector<float> > total_preds;
    total_preds.resize(bbox_nums);
    for (auto& bbox : total_preds) {
        bbox.resize(bbox_info_len);
    }

    for (auto idx_0 = 0; idx_0 < bbox_info_len; ++idx_0) {
        for (auto idx_1 = 0; idx_1 < bbox_nums; ++idx_1) {
            auto data_idx = idx_0 * bbox_nums + idx_1;
            total_preds[idx_1][idx_0] = output_tensor_0_data[data_idx];
        }
    }
    LOG(INFO) << total_preds.size();

    std::vector<bbox_> threshed_preds;
    for (auto& bbox : total_preds) {
        auto conf = bbox[4];
        if (conf > 0.25) {
            bbox_ b;
            b.score = bbox[4];
            b.masks = {bbox.begin() + 5, bbox.end()};
            auto cx = bbox[0];
            auto cy = bbox[1];
            auto width = bbox[2];
            auto height = bbox[3];
            auto x = cx - width / 2.0f;
            auto y = cy - height / 2.0f;
            b.bbox = cv::Rect2f(x, y, width, height);
            threshed_preds.push_back(b);
        }
    }
    LOG(INFO) << threshed_preds.size();

    auto nms_result = CvUtils::nms_bboxes(threshed_preds, 0.7);
    LOG(INFO) << nms_result.size();

    auto c = _m_output_1_shape[1];
    auto mh = _m_output_1_shape[2];
    auto mw = _m_output_1_shape[3];
    auto output_tensor_1_host = MNN::Tensor(_m_output_tensor_1, _m_output_tensor_1->getDimensionType());
    _m_output_tensor_1->copyToHostTensor(&output_tensor_1_host);
    auto* output_tensor_1_data = output_tensor_1_host.host<float>();
    std::vector<float> output_tensor_1_data_vec(output_tensor_1_data, output_tensor_1_data + output_tensor_1_host.elementSize());
    auto mask_proto_hwc = CvUtils::convert_to_hwc_vec(output_tensor_1_data_vec, 1, c, mh * mw);
    cv::Mat mask_proto(cv::Size(mh * mw, c), CV_32FC1, mask_proto_hwc.data());

    std::vector<cv::Mat> preds_masks;
    for (auto& bbox : nms_result) {
        cv::Mat mask_in(cv::Size(c, 1), CV_32FC1, bbox.masks.data());
        cv::Mat mask_output = mask_in * mask_proto;
        mask_output = mask_output.reshape(1, {mw, mh});
        cv::Mat tmp_exp(mask_output.size(), CV_32FC1);
        cv::exp(-mask_output, tmp_exp);
        cv::Mat sigmoid_output(mask_output.size(), CV_32FC1);
        sigmoid_output = 1.0f / (1.0f + tmp_exp);

        for (auto row = 0; row < sigmoid_output.rows; ++row) {
            for (auto col = 0; col < sigmoid_output.cols; ++col) {
                if (row > bbox.bbox.y * 0.25 && row < (bbox.bbox.y + bbox.bbox.height) * 0.25 &&
                    col > bbox.bbox.x * 0.25 && col < (bbox.bbox.x + bbox.bbox.width) * 0.25) {
                    continue;
                } else {
                    sigmoid_output.at<float>(row, col) = 0.0;
                }
            }
        }
        cv::resize(
            sigmoid_output, sigmoid_output, cv::Size(640, 640),
            0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat mask(sigmoid_output.size(), CV_8UC1);
        for (auto row = 0; row < sigmoid_output.rows; ++row) {
            for (auto col = 0; col < sigmoid_output.cols; ++col) {
                if (sigmoid_output.at<float>(row, col) >= 0.5) {
                    mask.at<uchar>(row, col) = 255;
                }
            }
        }
        preds_masks.push_back(mask);
    }

    auto comp_area = [](const cv::Mat& a, const cv::Mat& b) -> bool {
        auto a_count = cv::countNonZero(a);
        auto b_count = cv::countNonZero(b);

        return a_count >= b_count;
    };
    std::sort(preds_masks.begin(), preds_masks.end(), comp_area);

    auto color_pool = CvUtils::generate_color_map(static_cast<int>(nms_result.size()));
    cv::Mat color_mask(cv::Size(640, 640), CV_8UC3);
    for (auto idx = 0; idx < preds_masks.size(); ++idx) {
        auto color = color_pool[idx];
        auto mask = preds_masks[idx];
        for (auto row = 0; row < mask.rows; ++row) {
            for (auto col = 0; col < mask.cols; ++col) {
                if (mask.at<uchar>(row, col) == 255) {
                    color_mask.at<cv::Vec3b>(row, col)[0] = static_cast<uchar>(color[0]);
                    color_mask.at<cv::Vec3b>(row, col)[1] = static_cast<uchar>(color[1]);
                    color_mask.at<cv::Vec3b>(row, col)[2] = static_cast<uchar>(color[2]);
                }
            }
        }
    }

    cv::imwrite("fuck_mask.png", color_mask);
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