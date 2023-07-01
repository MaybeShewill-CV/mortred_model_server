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

    // model output shape info
    std::vector<int> _m_output_0_shape;
    std::vector<int> _m_output_1_shape;

    // input image size
    cv::Size _m_input_image_size;
    // input tensor size
    cv::Size _m_input_tensor_size;
    // preds mask shape
    cv::Size _m_preds_mask_size;

    // init flag
    bool _m_successfully_init_model = false;

    // internal bbox
    struct _m_preds_bbox {
        cv::Rect2f bbox;
        float score = 0.0;
        std::vector<float> masks;
        int class_id = 0;
    };

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
    cv::Mat preprocess_image(const cv::Mat& input_image);

    /***
     *
     */
    StatusCode decode_masks(std::vector<cv::Mat>& preds_masks, cv::Mat& merged_mask);
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

    // fetch input tensor
    _m_input_name = "images";
    _m_input_tensor = _m_net->getSessionInput(_m_session, _m_input_name.c_str());
    if (_m_input_tensor == nullptr) {
        LOG(INFO) << "fetch input node \'images\' failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_input_tensor->shape().size() != 4) {
        LOG(INFO) << "Invalid input tensor shape. Input tensor should be with [n, c, h, w] four dims but " << _m_input_tensor->shape().size() << " dims instead";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size = cv::Size(_m_input_tensor->shape()[3], _m_input_tensor->shape()[2]);

    // fetch output tensor 0
    _m_output_0_name = "output0";
    _m_output_tensor_0 = _m_net->getSessionOutput(_m_session, _m_output_0_name.c_str());
    if (_m_output_tensor_0 == nullptr) {
        LOG(INFO) << "fetch output node \'output0\' failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_output_0_shape = _m_output_tensor_0->shape();

    // fetch output tensor 1
    _m_output_1_name = "output1";    
    _m_output_tensor_1 = _m_net->getSessionOutput(_m_session, _m_output_1_name.c_str());
    if (_m_output_tensor_1 == nullptr) {
        LOG(INFO) << "fetch output node \'output1\' failed";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_output_tensor_1->shape().size() != 4) {
        LOG(INFO) << "Invalid output tensor 1 shape. Output tensor 1 should be with [n, c, h, w] four dims but " << _m_output_tensor_1->shape().size() << " dims instead";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_output_1_shape = _m_output_tensor_1->shape();
    _m_preds_mask_size = cv::Size(_m_output_1_shape[3], _m_output_1_shape[2]);

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load fastsam model";
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
    _m_input_image_size = input_image.size();
    auto preprocessed_image = preprocess_image(input_image);
    auto input_image_nchw_data = CvUtils::convert_to_chw_vec(preprocessed_image);
    LOG(INFO) << "input image nchw data: ";
    LOG(INFO) << " ---- "
              << input_image_nchw_data[0] << " "
              << input_image_nchw_data[1] << " "
              << input_image_nchw_data[2] << " "
              << input_image_nchw_data[3] << " "
              << input_image_nchw_data[4];

    // run session
    auto input_tensor_host = MNN::Tensor(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    ::memcpy(input_tensor_host.host<float>(), input_image_nchw_data.data(), input_tensor_host.size());
    _m_input_tensor->copyFromHostTensor(&input_tensor_host);

    _m_net->runSession(_m_session);

    // post process decode mask
    cv::Mat merged_mask;
    auto status = decode_masks(predicted_masks, merged_mask);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "decode masks failed, status code: " << status;
        return status;
    }

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

    cv::resize(input_image, result, _m_input_tensor_size);

    cv::cvtColor(result, result, cv::COLOR_BGR2RGB);

    result.convertTo(result, CV_32FC3);

    cv::divide(result, 255.0, result);

    return result;
}

/***
 *
 * @param preds_masks
 * @param merged_mask
 * @return
 */
StatusCode FastSamSegmentor::Impl::decode_masks(std::vector<cv::Mat>& preds_masks, cv::Mat& merged_mask) {
    // decode output preds info
    auto output_tensor_0_host = MNN::Tensor(_m_output_tensor_0, _m_output_tensor_0->getDimensionType());
    _m_output_tensor_0->copyToHostTensor(&output_tensor_0_host);
    auto* output_tensor_0_data = output_tensor_0_host.host<float>();
    LOG(INFO) << "output tensor 0 data: ";
    LOG(INFO) << " ---- "
              << output_tensor_0_data[0] << " "
              << output_tensor_0_data[1] << " "
              << output_tensor_0_data[2] << " "
              << output_tensor_0_data[3] << " "
              << output_tensor_0_data[4];

    auto bbox_info_len = _m_output_0_shape[1];
    auto bbox_nums = _m_output_0_shape[2];
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
    LOG(INFO) << "total preds bboxes: " << total_preds.size();

    std::vector<_m_preds_bbox> threshed_preds;
    for (auto& bbox : total_preds) {
        auto conf = bbox[4];
        if (conf > 0.25) {
            _m_preds_bbox b;
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
    LOG(INFO) << "remain bbox after conf thresh: " << threshed_preds.size();

    auto nms_result = CvUtils::nms_bboxes(threshed_preds, 0.7);
    LOG(INFO) << "remain bbox after nms thresh: " << nms_result.size();

    auto c = _m_output_1_shape[1];
    auto mh = _m_preds_mask_size.height;
    auto mw = _m_preds_mask_size.width;

    auto output_tensor_1_host = MNN::Tensor(_m_output_tensor_1, _m_output_tensor_1->getDimensionType());
    _m_output_tensor_1->copyToHostTensor(&output_tensor_1_host);
    auto* output_tensor_1_data = output_tensor_1_host.host<float>();
    LOG(INFO) << "output tensor 1 data: ";
    LOG(INFO) << " ---- "
              << output_tensor_1_data[0] << " "
              << output_tensor_1_data[1] << " "
              << output_tensor_1_data[2] << " "
              << output_tensor_1_data[3] << " "
              << output_tensor_1_data[4];
    std::vector<float> output_tensor_1_data_vec(output_tensor_1_data, output_tensor_1_data + output_tensor_1_host.elementSize());
    DLOG(INFO) << "output tensor 1 vec size: " << output_tensor_1_data_vec.size();

    auto mask_proto_hwc = CvUtils::convert_to_hwc_vec(output_tensor_1_data_vec, 1, c, mh * mw);
    DLOG(INFO) << "converted mask proto hwc vector size: " << mask_proto_hwc.size();

    cv::Mat mask_proto(cv::Size(mh * mw, c), CV_32FC1, mask_proto_hwc.data());
    DLOG(INFO) << "mask proto constructed complete, size: " << mask_proto.size();

    float downscale_h = static_cast<float>(mh) / static_cast<float>(_m_input_image_size.height);
    float downscale_w = static_cast<float>(mw) / static_cast<float>(_m_input_image_size.width);
    preds_masks.clear();
    for (auto& bbox : nms_result) {
        // decode mask
        cv::Mat mask_in(cv::Size(c, 1), CV_32FC1, bbox.masks.data());
        cv::Mat mask_output = mask_in * mask_proto;
        mask_output = mask_output.reshape(1, {mw, mh});
        cv::Mat tmp_exp(mask_output.size(), CV_32FC1);
        cv::exp(-mask_output, tmp_exp);
        cv::Mat sigmoid_output(mask_output.size(), CV_32FC1);
        sigmoid_output = 1.0f / (1.0f + tmp_exp);

        // crop mask
        for (auto row = 0; row < sigmoid_output.rows; ++row) {
            for (auto col = 0; col < sigmoid_output.cols; ++col) {
                // downscale preds bounding box
                auto scaled_bbox_tlx = bbox.bbox.x * downscale_w;
                auto scaled_bbox_tly = bbox.bbox.y * downscale_h;
                auto scaled_bbox_rbx = scaled_bbox_tlx + bbox.bbox.width * downscale_w;
                auto scaled_bbox_rby = scaled_bbox_tly + bbox.bbox.height * downscale_h;
                // crop mask via bounding box
                if (row > scaled_bbox_tly && row < scaled_bbox_rby && col > scaled_bbox_tlx && col < scaled_bbox_rbx) {
                    continue;
                } else {
                    sigmoid_output.at<float>(row, col) = 0.0f;
                }
            }
        }

        // thresh mask
        cv::resize(sigmoid_output, sigmoid_output, _m_input_image_size, 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat mask = cv::Mat::zeros(sigmoid_output.size(), CV_8U);
        for (auto row = 0; row < sigmoid_output.rows; ++row) {
            for (auto col = 0; col < sigmoid_output.cols; ++col) {
                if (sigmoid_output.at<float>(row, col) >= 0.5) {
                    mask.at<uchar>(row, col) = 255;
                }
            }
        }
        LOG(INFO) << mask.channels();
        preds_masks.push_back(mask);
    }

    // merge total predicted masks according to mask area order
    auto comp_area = [] (const cv::Mat& a, const cv::Mat& b) -> bool {

        auto a_area = cv::countNonZero(a);
        auto b_area = cv::countNonZero(b);
        
        return a_area >= b_area;
    };
    // std::sort(preds_masks.begin(), preds_masks.end(), comp_area);

    auto color_pool = CvUtils::generate_color_map(static_cast<int>(nms_result.size()));
    cv::Mat color_mask(_m_input_image_size, CV_8UC3);
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
    color_mask.copyTo(merged_mask);

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
