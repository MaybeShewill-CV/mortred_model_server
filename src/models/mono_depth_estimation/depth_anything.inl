/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: depth_anything.inl
 * Date: 24-1-25
 ************************************************/

#include "depth_anything.h"

#include <cstring>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace mono_depth_estimation {

using MdeOutput = jinq::models::io_define::mono_depth_estimation::std_mde_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode DepthAnything<INPUT, OUTPUT>::on_init(const toml::table &params) {
    // focal_length / intrinsic are accepted for config parity with the metric
    // models; the relative-depth head does not consume them
    (void)params;
    const auto &input_info = this->session().inputs().front();
    // dynamic batch (shape[0] == -1) is fine: only the spatial dims must be
    // concrete for preprocessing; a batch-profile engine reports input_info.dynamic
    // because dim0 is -1, but dims 1..3 are always concrete
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3 || input_info.shape[2] <= 0 || input_info.shape[3] <= 0) {
        LOG(ERROR) << "unexpected depth anything input shape: " << input_info.to_string() << ", expected static [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> DepthAnything<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // keep-ratio rescale, right/bottom zero pad, imagenet normalize (f32 nchw)
    int rescale_w = 0;
    int rescale_h = 0;
    const float aspect_ratio = static_cast<float>(input_image.cols) / static_cast<float>(input_image.rows);
    if (aspect_ratio >= 1) {
        rescale_w = _m_input_size_host.width;
        rescale_h = static_cast<int>(_m_input_size_host.height / aspect_ratio);
    } else {
        rescale_w = static_cast<int>(_m_input_size_host.width * aspect_ratio);
        rescale_h = _m_input_size_host.height;
    }

    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(rescale_w, rescale_h), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat out = cv::Mat::zeros(_m_input_size_host, CV_8UC3);
    resized_image.copyTo(out(cv::Rect(0, 0, resized_image.cols, resized_image.rows)));
    out.convertTo(out, CV_32FC3);
    cv::divide(out, 255, out);
    cv::subtract(out, cv::Scalar(0.406f, 0.456f, 0.485f), out);
    cv::divide(out, cv::Scalar(0.225f, 0.224f, 0.229f), out);

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(out);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>({1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
StatusCode DepthAnything<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                     const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "depth anything output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto &tensor = outputs.front().tensor;
    const auto *depth_data = tensor.template data<float>();
    if (tensor.element_count() < static_cast<int64_t>(_m_input_size_host.area())) {
        LOG(ERROR) << "depth map smaller than the input map: " << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    cv::Mat depth_map(_m_input_size_host, CV_32FC1, const_cast<float *>(depth_data));

    // undo the keep-ratio padding by cropping the valid region
    int crop_w = 0;
    int crop_h = 0;
    if (context.source_size.width > context.source_size.height) {
        crop_w = _m_input_size_host.width;
        crop_h = _m_input_size_host.height * context.source_size.height / context.source_size.width;
    } else {
        crop_w = _m_input_size_host.width * context.source_size.width / context.source_size.height;
        crop_h = _m_input_size_host.height;
    }
    cv::resize(depth_map(cv::Rect(0, 0, crop_w, crop_h)), depth_map, context.source_size);

    MdeOutput internal_out;
    internal_out.depth_map = depth_map.clone();
    jinq::common::CvUtils::colorize_depth_map(depth_map, internal_out.colorized_depth_map);
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
DepthAnything<INPUT, OUTPUT>::DepthAnything() : jinq::models::BackendCvModel<INPUT, OUTPUT>("DEPTH_ANYTHING") {}

} // namespace mono_depth_estimation
} // namespace models
} // namespace jinq
