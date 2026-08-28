/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: pp_humanseg.inl
 * Date: 22-7-20
 ************************************************/

#include "pp_humanseg.h"

#include <cstring>
#include <utility>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

using SegmentationOutput = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode PPHumanSeg<INPUT, OUTPUT>::on_init(const toml::table &params) {
    (void)params;
    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected pphumanseg input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "invalid pphumanseg input tensor size: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> PPHumanSeg<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {

    cv::Mat tmp;
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);
    if (tmp.size() != _m_input_size_host) {
        cv::resize(tmp, tmp, _m_input_size_host);
    }
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    tmp /= 255.0;
    cv::subtract(tmp, cv::Scalar(0.5, 0.5, 0.5), tmp);
    cv::divide(tmp, cv::Scalar(0.5, 0.5, 0.5), tmp);

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>({1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data size mismatches input tensor size";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());

    std::vector<NamedTensor> inputs;
    inputs.push_back(std::move(named));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
StatusCode PPHumanSeg<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                  const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "pphumanseg output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto &tensor = outputs.front().tensor;
    if (tensor.shape.size() < 3) {
        LOG(ERROR) << "unexpected pphumanseg output shape: " << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    const auto channels = tensor.shape[tensor.shape.size() - 3];
    if (channels != 2) {
        LOG(ERROR) << "unexpected pphumanseg output channel count: " << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    const auto *host_data = tensor.template data<float>();
    const auto plane_size = static_cast<int64_t>(_m_input_size_host.area());
    std::vector<float> hwc_host_data(static_cast<size_t>(plane_size * channels));
    for (auto row = 0; row < _m_input_size_host.height; ++row) {
        for (auto col = 0; col < _m_input_size_host.width; ++col) {
            for (auto channel = 0; channel < channels; ++channel) {
                hwc_host_data[static_cast<size_t>(row * _m_input_size_host.width * channels + col * channels + channel)] =
                    host_data[static_cast<size_t>(channel * plane_size + row * _m_input_size_host.width + col)];
            }
        }
    }

    cv::Mat logits(_m_input_size_host, CV_32FC2, hwc_host_data.data());
    cv::resize(logits, logits, context.source_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat result_image(context.source_size, CV_32SC1, cv::Scalar(0));
    for (auto row = 0; row < logits.rows; ++row) {
        for (auto col = 0; col < logits.cols; ++col) {
            const auto logit_val = logits.at<cv::Vec2f>(row, col);
            if (logit_val[0] < logit_val[1]) {
                result_image.at<int32_t>(row, col) = 1;
            }
        }
    }

    SegmentationOutput internal_out;
    internal_out.segmentation_result = std::move(result_image);
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
PPHumanSeg<INPUT, OUTPUT>::PPHumanSeg() : jinq::models::BackendCvModel<INPUT, OUTPUT>("PP_HUMANSEG") {}

} // namespace scene_segmentation
} // namespace models
} // namespace jinq
