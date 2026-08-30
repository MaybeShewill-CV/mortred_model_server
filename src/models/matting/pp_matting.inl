/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: pp_matting.inl
 * Date: 22-7-19
 ************************************************/

#include "pp_matting.h"

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
namespace matting {

using MattingOutput = jinq::models::io_define::matting::std_matting_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode PPMatting<INPUT, OUTPUT>::on_init(const toml::table &params) {
    (void)params;
    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected ppmatting input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "invalid ppmatting input tensor size: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (this->session().outputs().empty()) {
        LOG(ERROR) << "ppmatting model exposes no output tensor";
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> PPMatting<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {

    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .bgr_to_rgb()
                      .resize(_m_input_size_host)
                      .to_float()
                      .scale(1.0f / 255.0f)
                      .subtract({0.5f, 0.5f, 0.5f})
                      .divide({0.5f, 0.5f, 0.5f})
                      .nchw(this->session().inputs().front().name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode PPMatting<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                 const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "ppmatting output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto source_status = jinq::models::backend::validated_source_size(context, "ppmatting");
    if (source_status != StatusCode::OK) {
        return source_status;
    }
    jinq::models::backend::F32OutputView output_view;
    const auto output_status = jinq::models::backend::validated_f32_first_output(
        outputs, {jinq::models::backend::DType::F32, 4, {1, 1, context.network_size.height, context.network_size.width}}, "ppmatting",
        &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto *host_data = output_view.data;

    cv::Mat result_image(context.network_size, CV_32FC1, const_cast<float *>(host_data));
    cv::resize(result_image, result_image, context.source_size, 0.0, 0.0, cv::INTER_NEAREST);
    result_image *= 255.0;
    result_image.convertTo(result_image, CV_32SC1);

    MattingOutput internal_out;
    internal_out.matting_result = std::move(result_image);
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
PPMatting<INPUT, OUTPUT>::PPMatting() : jinq::models::BackendCvModel<INPUT, OUTPUT>("PP_MATTING") {}

} // namespace matting
} // namespace models
} // namespace jinq
