/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: db_text_detector.cpp
* Date: 22-6-6
************************************************/

#include "db_text_detector.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace ocr {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using TextRegion = jinq::models::io_define::ocr::text_region;
using TextRegions = jinq::models::io_define::ocr::std_text_regions_output;

template<typename INPUT, typename OUTPUT>
StatusCode DBTextDetector<INPUT, OUTPUT>::on_init(const toml::table& params) {
    const auto& inputs = this->session().inputs();
    const auto& outputs = this->session().outputs();
    if (inputs.empty() || outputs.empty()) {
        LOG(ERROR) << "db text model exposes no io tensors";
        return StatusCode::MODEL_INIT_FAILED;
    }
    const auto& input_info = inputs.front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected db text input shape: " << input_info.to_string()
                   << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_name = input_info.name;
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.width <= 0 || _m_input_size_host.height <= 0) {
        LOG(ERROR) << "invalid db text input size: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto output_iter = std::find_if(
        outputs.begin(), outputs.end(), [](const auto& item) {
            return item.name == "sigmoid_0.tmp_0";
        });
    if (output_iter == outputs.end()) {
        LOG(ERROR) << "db text output tensor 'sigmoid_0.tmp_0' is missing";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_output_name = output_iter->name;

    if (params.contains("model_input_image_size")) {
        const toml::array* size = params["model_input_image_size"].as_array();
        if (size == nullptr || size->size() != 2) {
            LOG(ERROR) << "params key 'model_input_image_size' must be [height, width]";
            return StatusCode::MODEL_INIT_FAILED;
        }
        _m_input_size_user.height = static_cast<int>((*size)[0].value_or<int64_t>(0));
        _m_input_size_user.width = static_cast<int>((*size)[1].value_or<int64_t>(0));
    } else {
        _m_input_size_user.width = 640;
        _m_input_size_user.height = 640;
    }
    if (params.contains("model_score_threshold")) {
        _m_score_threshold = params["model_score_threshold"].value_or<double>(0.0);
    }
    if (params.contains("model_keep_top_k")) {
        _m_keep_topk = params["model_keep_top_k"].value_or<int64_t>(0);
    }

    LOG(INFO) << "DB_Text detection model initialization complete!!!";
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> DBTextDetector<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    _m_input_size_user = input_image.size();

    // resize image
    cv::Mat tmp;
    cv::resize(input_image, tmp, _m_input_size_host);

    // normalize
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    tmp /= 255.0;
    cv::subtract(tmp, cv::Scalar(0.485, 0.456, 0.406), tmp);
    cv::divide(tmp, cv::Scalar(0.229, 0.224, 0.225), tmp);

    const auto input_chw_image_data = CvUtils::convert_to_chw_vec(tmp);
    NamedTensor named;
    named.name = _m_input_name;
    named.tensor = Tensor::make<float>(
        {1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (input_chw_image_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed db text image size mismatches input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), input_chw_image_data.data(),
                named.tensor.byte_size());
    return {std::move(named)};
}

template<typename INPUT, typename OUTPUT>
StatusCode DBTextDetector<INPUT, OUTPUT>::postprocess(
    const std::vector<NamedTensor>& outputs, OUTPUT& output) {
    const auto output_iter = std::find_if(
        outputs.begin(), outputs.end(),
        [this](const NamedTensor& item) { return item.name == _m_output_name; });
    if (output_iter == outputs.end()) {
        LOG(ERROR) << "db text inference result tensor is missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = output_iter->tensor;
    if (tensor.dtype != jinq::models::backend::DType::F32 ||
        tensor.element_count() <= 0) {
        LOG(ERROR) << "invalid db text inference result tensor";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto* output_data = tensor.template data<float>();
    const auto ele_size = tensor.element_count();

    // construct segmentation prob and score maps. The score values below the
    // threshold are zeroed before contours are decoded, exactly as before.
    std::vector<float> score_data(output_data, output_data + ele_size);
    std::vector<uchar> seg_mat_vec(ele_size);
    for (int index = 0; index < ele_size; ++index) {
        if (score_data[index] >= _m_score_threshold) {
            seg_mat_vec[index] = static_cast<uchar>(score_data[index] * 255.0);
        } else {
            seg_mat_vec[index] = static_cast<uchar>(0);
            score_data[index] = 0.0f;
        }
    }
    cv::Mat seg_prob_mat(_m_input_size_host, CV_8UC1, seg_mat_vec.data());
    cv::Mat seg_score_mat(_m_input_size_host, CV_32FC1, score_data.data());

    return get_boxes_from_bitmap(seg_prob_mat, seg_score_mat, output);
}

template<typename INPUT, typename OUTPUT>
StatusCode DBTextDetector<INPUT, OUTPUT>::get_boxes_from_bitmap(
    const cv::Mat& seg_prob_mat, const cv::Mat& seg_score_mat, OUTPUT& output) const {
    TextRegions result;
    const auto host_width = static_cast<float>(_m_input_size_host.width);
    const auto host_height = static_cast<float>(_m_input_size_host.height);
    const auto user_width = static_cast<float>(_m_input_size_user.width);
    const auto user_height = static_cast<float>(_m_input_size_user.height);

    // contours analysis
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(seg_prob_mat, contours, hierarchy, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        const cv::RotatedRect r_bbox = cv::minAreaRect(contour);
        cv::Rect2f r_bounding_box = r_bbox.boundingRect2f();
        cv::Point2f r_vertices[4];
        r_bbox.points(r_vertices);
        const auto sside = std::min(r_bbox.size.height, r_bbox.size.width);

        // thresh those with short sside
        if (sside < _m_sside_threshold) {
            continue;
        }

        // calculate rotated bbox score
        const auto valid_roi = r_bounding_box &
                               cv::Rect2f(0, 0, seg_score_mat.cols, seg_score_mat.rows);
        const float score = static_cast<float>(cv::mean(seg_score_mat(valid_roi))[0]);
        if (score < _m_score_threshold) {
            continue;
        }

        // rescale bbox coords to origin user image size
        for (auto& pt : r_vertices) {
            pt.x = pt.x * user_width / host_width;
            pt.y = pt.y * user_height / host_height;
        }
        r_bounding_box.x = r_bounding_box.x * user_width / host_width;
        r_bounding_box.y = r_bounding_box.y * user_height / host_height;
        r_bounding_box.width = r_bounding_box.width * user_width / host_width;
        r_bounding_box.height = r_bounding_box.height * user_height / host_height;

        TextRegion region;
        region.bbox = r_bounding_box;
        region.polygon = std::vector<cv::Point2f>(r_vertices, r_vertices + 4);
        region.score = score;
        result.push_back(region);
    }

    if (result.size() > static_cast<size_t>(_m_keep_topk)) {
        result.resize(static_cast<size_t>(_m_keep_topk));
    }
    output = std::move(result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
DBTextDetector<INPUT, OUTPUT>::DBTextDetector()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("DB_TEXT") {}

}
}
}
