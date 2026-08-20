/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: attentive_gan_derain_net.inl
* Date: 22-6-14
************************************************/

#include "attentive_gan_derain_net.h"

#include <algorithm>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "glog/logging.h"

namespace jinq {
namespace models {
namespace enhancement {

using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;
using jinq::models::io_define::enhancement::std_enhancement_output;

template <typename INPUT, typename OUTPUT>
StatusCode AttentiveGanDerain<INPUT, OUTPUT>::on_init(const toml::table& params) {
    (void)params;
    const auto& inputs = this->session().inputs();
    if (inputs.size() != 1 || inputs.front().shape.size() != 4 ||
            inputs.front().shape[3] != 3 || inputs.front().dynamic) {
        LOG(ERROR) << "unexpected attentive gan input io: "
                   << (inputs.empty() ? std::string("empty") : inputs.front().to_string())
                   << ", expected one static [N,H,W,3] tensor";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(inputs.front().shape[1]);
    _m_input_size_host.width = static_cast<int>(inputs.front().shape[2]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "invalid attentive gan input size: " << _m_input_size_host;
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
std::vector<NamedTensor> AttentiveGanDerain<INPUT, OUTPUT>::preprocess(
    const cv::Mat& input_image) {
    _m_input_size_user = input_image.size();

    cv::Mat tmp;
    if (input_image.size() != _m_input_size_host) {
        cv::resize(input_image, tmp, _m_input_size_host);
    } else {
        tmp = input_image;
    }
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    tmp /= 127.5;
    cv::subtract(tmp, cv::Scalar(1.0, 1.0, 1.0), tmp);

    std::vector<NamedTensor> tensors;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, _m_input_size_host.height, _m_input_size_host.width, 3});
    const auto bytes = tmp.total() * tmp.elemSize();
    if (bytes != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed attentive gan image byte size mismatches tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), tmp.data, bytes);
    tensors.push_back(std::move(named));
    return tensors;
}

template <typename INPUT, typename OUTPUT>
StatusCode AttentiveGanDerain<INPUT, OUTPUT>::postprocess(
    const std::vector<NamedTensor>& outputs, OUTPUT& output) {
    if (outputs.empty()) {
        LOG(ERROR) << "attentive gan output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = outputs.front().tensor;
    // the exported mnn output is plain hwc ([H,W,3] or [1,H,W,3])
    if ((tensor.shape.size() != 3 || tensor.shape[2] != 3) &&
        (tensor.shape.size() != 4 || tensor.shape[3] != 3)) {
        LOG(ERROR) << "unexpected attentive gan output shape: "
                   << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    if (tensor.shape.size() == 3) {
        _m_input_size_host.height = static_cast<int>(tensor.shape[0]);
        _m_input_size_host.width = static_cast<int>(tensor.shape[1]);
    } else {
        _m_input_size_host.height = static_cast<int>(tensor.shape[1]);
        _m_input_size_host.width = static_cast<int>(tensor.shape[2]);
    }
    if (_m_input_size_host.area() <= 0 ||
            tensor.element_count() != 3 * _m_input_size_host.area()) {
        LOG(ERROR) << "invalid attentive gan output shape: "
                   << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto* host_data = tensor.template data<float>();
    cv::Mat output_feats(
        _m_input_size_host, CV_32FC3, const_cast<float*>(host_data));
    std::vector<cv::Mat> output_feats_split;
    cv::split(output_feats, output_feats_split);
    const auto b_max_value = *std::max_element(
        output_feats_split[0].begin<float>(), output_feats_split[0].end<float>());
    const auto b_min_value = *std::min_element(
        output_feats_split[0].begin<float>(), output_feats_split[0].end<float>());
    const auto g_max_value = *std::max_element(
        output_feats_split[1].begin<float>(), output_feats_split[1].end<float>());
    const auto g_min_value = *std::min_element(
        output_feats_split[1].begin<float>(), output_feats_split[1].end<float>());
    const auto r_max_value = *std::max_element(
        output_feats_split[2].begin<float>(), output_feats_split[2].end<float>());
    const auto r_min_value = *std::min_element(
        output_feats_split[2].begin<float>(), output_feats_split[2].end<float>());

    cv::Mat output_image(_m_input_size_host, CV_8UC3);
    for (auto row = 0; row < output_image.size().height; ++row) {
        for (auto col = 0; col < output_image.size().width; ++col) {
            const float b_feats_val = output_feats.at<cv::Vec3f>(row, col)[0];
            const float g_feats_val = output_feats.at<cv::Vec3f>(row, col)[1];
            const float r_feats_val = output_feats.at<cv::Vec3f>(row, col)[2];

            const auto b_scale_val = static_cast<float>(
                (b_feats_val - b_min_value) * 255.0 / (b_max_value - b_min_value));
            const auto g_scale_val = static_cast<float>(
                (g_feats_val - g_min_value) * 255.0 / (g_max_value - g_min_value));
            const auto r_scale_val = static_cast<float>(
                (r_feats_val - r_min_value) * 255.0 / (r_max_value - r_min_value));

            output_image.at<cv::Vec3b>(row, col)[0] = static_cast<uint8_t>(b_scale_val);
            output_image.at<cv::Vec3b>(row, col)[1] = static_cast<uint8_t>(g_scale_val);
            output_image.at<cv::Vec3b>(row, col)[2] = static_cast<uint8_t>(r_scale_val);
        }
    }
    if (output_image.size() != _m_input_size_user) {
        cv::resize(output_image, output_image, _m_input_size_user);
    }

    std_enhancement_output internal_out;
    output_image.copyTo(internal_out.enhancement_result);
    output = std::move(internal_out);
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
AttentiveGanDerain<INPUT, OUTPUT>::AttentiveGanDerain()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("ATTENTIVEGANDERAIN") {}

} // namespace enhancement
} // namespace models
} // namespace jinq
