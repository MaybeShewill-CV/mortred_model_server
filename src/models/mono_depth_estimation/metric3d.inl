/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: metric3d.cpp
 ************************************************/

#include "metric3d.h"

#include <algorithm>
#include <cstring>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace mono_depth_estimation {

using MdeOutput = jinq::models::io_define::mono_depth_estimation::std_mde_output;
using jinq::models::backend::NamedTensor;
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
StatusCode Metric3D<INPUT, OUTPUT>::on_init(const toml::table& params) {
    if (params.contains("focal_length")) {
        _m_focal_length = static_cast<float>(params["focal_length"].value_or<double>(0.0));
    }
    if (params.contains("intrinsic")) {
        const toml::array* intrinsic = params["intrinsic"].as_array();
        if (intrinsic != nullptr && intrinsic->size() == 4) {
            for (size_t idx = 0; idx < 4; ++idx) {
                _m_intrinsic_params[idx] =
                    static_cast<float>((*intrinsic)[idx].value_or<double>(0.0));
            }
        }
    }
    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3 || input_info.dynamic) {
        LOG(ERROR) << "unexpected metric3d input shape: " << input_info.to_string()
                   << ", expected static [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
const NamedTensor* Metric3D<INPUT, OUTPUT>::find_output(
    const std::vector<NamedTensor>& outputs, const std::string& name) const {
    for (const auto& item : outputs) {
        if (item.name == name) {
            return &item;
        }
    }
    return nullptr;
}

template<typename INPUT, typename OUTPUT>
void Metric3D<INPUT, OUTPUT>::calculate_pad_info(int& pad_h, int& pad_w) const {
    const auto src_w = _m_input_size_user.width;
    const auto src_h = _m_input_size_user.height;
    const auto resize_ratio_h =
        static_cast<float>(_m_input_size_host.height) / static_cast<float>(src_h);
    const auto resize_ratio_w =
        static_cast<float>(_m_input_size_host.width) / static_cast<float>(src_w);
    const auto to_scale_ratio = std::min(resize_ratio_h, resize_ratio_w);
    const auto reshape_h = static_cast<int>(to_scale_ratio * static_cast<float>(src_h));
    const auto reshape_w = static_cast<int>(to_scale_ratio * static_cast<float>(src_w));
    pad_h = std::max(_m_input_size_host.height - reshape_h, 0);
    pad_w = std::max(_m_input_size_host.width - reshape_w, 0);
}

template<typename INPUT, typename OUTPUT>
float Metric3D<INPUT, OUTPUT>::calculate_label_scale_factor() const {
    const auto ori_focal = (_m_intrinsic_params[0] + _m_intrinsic_params[1]) / 2.0f;
    const auto canonical_focal = _m_focal_length;
    const auto src_w = _m_input_size_user.width;
    const auto src_h = _m_input_size_user.height;
    const auto resize_ratio_h =
        static_cast<float>(_m_input_size_host.height) / static_cast<float>(src_h);
    const auto resize_ratio_w =
        static_cast<float>(_m_input_size_host.width) / static_cast<float>(src_w);
    const auto to_scale_ratio = std::min(resize_ratio_h, resize_ratio_w);
    const auto resize_label_scale_factor = 1.0f / to_scale_ratio;
    const auto cano_label_scale_ratio = canonical_focal / ori_focal;
    return cano_label_scale_ratio * resize_label_scale_factor;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> Metric3D<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    // bgr -> rgb -> keep-ratio resize -> center pad -> mean/std (f32 nchw)
    _m_input_size_user = input_image.size();
    cv::Mat tmp;
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    const auto src_w = _m_input_size_user.width;
    const auto src_h = _m_input_size_user.height;
    const auto resize_ratio_h =
        static_cast<float>(_m_input_size_host.height) / static_cast<float>(src_h);
    const auto resize_ratio_w =
        static_cast<float>(_m_input_size_host.width) / static_cast<float>(src_w);
    const auto to_scale_ratio = std::min(resize_ratio_h, resize_ratio_w);
    const auto reshape_h = static_cast<int>(to_scale_ratio * static_cast<float>(src_h));
    const auto reshape_w = static_cast<int>(to_scale_ratio * static_cast<float>(src_w));
    const auto pad_h = std::max(_m_input_size_host.height - reshape_h, 0);
    const auto pad_w = std::max(_m_input_size_host.width - reshape_w, 0);
    const auto pad_h_half = pad_h / 2;
    const auto pad_w_half = pad_w / 2;

    cv::resize(tmp, tmp, cv::Size(reshape_w, reshape_h));
    cv::copyMakeBorder(tmp, tmp, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                       cv::BORDER_CONSTANT, cv::Scalar(123.675, 116.28, 103.53));
    cv::subtract(tmp, cv::Scalar(123.675, 116.28, 103.53), tmp);
    cv::divide(tmp, cv::Scalar(58.395, 57.12, 57.375), tmp);

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template<typename INPUT, typename OUTPUT>
StatusCode Metric3D<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                OUTPUT& output) {
    const auto* depth_tensor = find_output(outputs, "prediction");
    const auto* confidence_tensor = find_output(outputs, "confidence");
    if (depth_tensor == nullptr || confidence_tensor == nullptr) {
        LOG(ERROR) << "metric3d outputs 'prediction'/'confidence' missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto* depth_data = depth_tensor->tensor.template data<float>();
    const auto* conf_data = confidence_tensor->tensor.template data<float>();
    if (depth_tensor->tensor.element_count() <
            static_cast<int64_t>(_m_input_size_host.area()) ||
        confidence_tensor->tensor.element_count() <
            static_cast<int64_t>(_m_input_size_host.area())) {
        LOG(ERROR) << "metric3d maps smaller than the input map";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    cv::Mat depth_map = cv::Mat::zeros(_m_input_size_host, CV_32FC1);
    cv::Mat confidence_map = cv::Mat::zeros(_m_input_size_host, CV_32FC1);
    for (auto row = 0; row < _m_input_size_host.height; ++row) {
        auto* depth_row = depth_map.ptr<float>(row);
        auto* conf_row = confidence_map.ptr<float>(row);
        for (auto col = 0; col < _m_input_size_host.width; ++col) {
            const auto idx = row * _m_input_size_host.width + col;
            depth_row[col] = depth_data[idx] < 0 ? 0 : depth_data[idx];
            conf_row[col] = conf_data[idx];
        }
    }

    // crop the center padding, rescale and undo the canonical focal scaling
    int pad_h = 0;
    int pad_w = 0;
    calculate_pad_info(pad_h, pad_w);
    auto crop_roi = cv::Rect(pad_w / 2, pad_h / 2, depth_map.cols - pad_w, depth_map.rows - pad_h);
    crop_roi = crop_roi & cv::Rect(0, 0, depth_map.cols, depth_map.rows);
    depth_map(crop_roi).copyTo(depth_map);
    confidence_map(crop_roi).copyTo(confidence_map);
    cv::resize(depth_map, depth_map, _m_input_size_user);
    cv::resize(confidence_map, confidence_map, _m_input_size_user);
    cv::divide(depth_map, calculate_label_scale_factor(), depth_map);

    MdeOutput internal_out;
    internal_out.depth_map = depth_map.clone();
    internal_out.confidence_map = confidence_map.clone();
    jinq::common::CvUtils::colorize_depth_map(depth_map, internal_out.colorized_depth_map);
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
Metric3D<INPUT, OUTPUT>::Metric3D()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("METRIC3D") {}

}
}
}
