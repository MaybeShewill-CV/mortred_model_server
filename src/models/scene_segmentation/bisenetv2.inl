/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: bisenetv2.inl
 * Date: 22-6-9
 ************************************************/

#include "bisenetv2.h"

#include <cstring>
#include <utility>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "models/backend/f32_output.h"
#include "models/backend/model_runtime.h"
#include "models/backend/request_geometry.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

using SegmentationOutput = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode BiseNetV2<INPUT, OUTPUT>::on_init(const toml::table &params) {
    (void)params;
    const auto input_info =
        jinq::models::backend::SessionIoValidator(this->session()).input().f32().rank(4).nhwc().channels(3).static_shape().validate();
    if (!input_info.ok()) {
        LOG(ERROR) << "unexpected bisenetv2 input shape: " << input_info.error << ", expected static [N,H,W,3] (nhwc)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.value.shape[1]);
    _m_input_size_host.width = static_cast<int>(input_info.value.shape[2]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "invalid bisenetv2 input tensor size: " << input_info.error;
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> BiseNetV2<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {

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

    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>({1, _m_input_size_host.height, _m_input_size_host.width, 3});
    const auto bytes = tmp.total() * tmp.elemSize();
    if (bytes != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed image byte size " << bytes << " mismatches input tensor byte size " << named.tensor.byte_size();
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), tmp.data, bytes);

    std::vector<NamedTensor> inputs;
    inputs.push_back(std::move(named));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
StatusCode BiseNetV2<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                 const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "bisenetv2 output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto source_status = jinq::models::backend::validated_source_size(context, "bisenetv2");
    if (source_status != StatusCode::OK) {
        return source_status;
    }
    jinq::models::backend::F32OutputView output_view;
    const auto output_status = jinq::models::backend::validated_f32_first_output(
        outputs, {jinq::models::backend::DType::F32, 3, {context.network_size.height, context.network_size.width, -1}}, "bisenetv2",
        &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto &tensor = *output_view.tensor;
    const auto *host_data = output_view.data;
    const auto cls_nums = tensor.shape.back();

    // model output is float [H, W, cls_nums], compute argmax per pixel
    cv::Mat result_image(context.network_size, CV_32SC1, cv::Scalar(0));
    for (auto row = 0; row < result_image.rows; ++row) {
        for (auto col = 0; col < result_image.cols; ++col) {
            const float *logit = host_data + (row * result_image.cols + col) * cls_nums;
            int best_cls = 0;
            float best_val = logit[0];
            for (auto cls = 1; cls < cls_nums; ++cls) {
                if (logit[cls] > best_val) {
                    best_val = logit[cls];
                    best_cls = cls;
                }
            }
            result_image.at<int32_t>(row, col) = best_cls;
        }
    }
    cv::resize(result_image, result_image, context.source_size, 0.0, 0.0, cv::INTER_NEAREST);

    SegmentationOutput internal_out;
    internal_out.segmentation_result = std::move(result_image);
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
BiseNetV2<INPUT, OUTPUT>::BiseNetV2() : jinq::models::BackendCvModel<INPUT, OUTPUT>("BISENETV2") {}

} // namespace scene_segmentation
} // namespace models
} // namespace jinq
