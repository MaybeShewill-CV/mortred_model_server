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
StatusCode PPMatting<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                 const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "ppmatting output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto &tensor = outputs.front().tensor;
    const auto *host_data = tensor.template data<float>();
    if (tensor.element_count() < static_cast<int64_t>(_m_input_size_host.area())) {
        LOG(ERROR) << "unexpected ppmatting output shape: " << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    cv::Mat result_image(_m_input_size_host, CV_32FC1, const_cast<float *>(host_data));
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
