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
#include "models/backend/f32_output.h"
#include "models/backend/model_runtime.h"
#include "models/backend/request_geometry.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

using SegmentationOutput = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode PPHumanSeg<INPUT, OUTPUT>::on_init(const toml::table &params) {
    const auto input_info =
        jinq::models::backend::SessionIoValidator(this->session()).input().f32().allow_fp16().rank(4).nchw().channels(3).static_shape().validate();
    if (!input_info.ok()) {
        LOG(ERROR) << "unexpected pphumanseg input shape: " << input_info.error << ", expected static [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.value.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.value.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "invalid pphumanseg input tensor size: " << input_info.error;
        return StatusCode::MODEL_INIT_FAILED;
    }
    this->set_image_decode_hint(_m_input_size_host, jinq::models::cv_input::parse_image_decode_upscale(params, "pphumanseg"));
    this->set_gpu_preprocess({
        .resize = jinq::models::backend::GpuPreprocessDescriptor::Resize::DIRECT_RESIZE,
        .norm = {.scale = 1.0f / 255.0f, .mean = {0.5f, 0.5f, 0.5f}, .std = {0.5f, 0.5f, 0.5f}},
        .color = jinq::models::backend::GpuPreprocessDescriptor::Color::RGB,
        .pad_value = 114,
        .output_dtype = input_info.value.dtype,
        .output_nhwc = false,
    });

    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> PPHumanSeg<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {

    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .bgr_to_rgb()
                      .resize(_m_input_size_host)
                      .to_float()
                      .scale(1.0f / 255.0f)
                      .subtract({0.5f, 0.5f, 0.5f})
                      .divide({0.5f, 0.5f, 0.5f})
                      .nchw(this->session().inputs().front());
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode PPHumanSeg<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                  const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "pphumanseg output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto source_status = jinq::models::backend::validated_source_size(context, "pphumanseg");
    if (source_status != StatusCode::OK) {
        return source_status;
    }
    jinq::models::backend::F32OutputView output_view;
    const auto output_status = jinq::models::backend::validated_f32_first_output(
        outputs, {jinq::models::backend::DType::F32, 4, {1, 2, context.network_size.height, context.network_size.width},
                  jinq::models::backend::TensorLayout::Nchw}, "pphumanseg",
        &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto *host_data = output_view.data;

    const auto plane_size = static_cast<int64_t>(context.network_size.area());
    // family decode order (bisenetv2 / hrnet / msocrnet): argmax at network
    // resolution, then NEAREST upsample to source. The old path bilinearly
    // resized logits first, which yields different edge labels and no other
    // family member does it. Binary output: channel 1 wins when its logit
    // exceeds channel 0's.
    cv::Mat result_image(context.network_size, CV_32SC1, cv::Scalar(0));
    for (auto row = 0; row < context.network_size.height; ++row) {
        for (auto col = 0; col < context.network_size.width; ++col) {
            const auto offset = static_cast<int64_t>(row) * context.network_size.width + col;
            if (host_data[plane_size + offset] > host_data[offset]) {
                result_image.at<int32_t>(row, col) = 1;
            }
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
PPHumanSeg<INPUT, OUTPUT>::PPHumanSeg() : jinq::models::BackendCvModel<INPUT, OUTPUT>("PP_HUMANSEG") {}

} // namespace scene_segmentation
} // namespace models
} // namespace jinq
