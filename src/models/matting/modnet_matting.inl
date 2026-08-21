/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: modnet_matting.inl
 * Date: 22-7-19
 ************************************************/

#include "modnet_matting.h"

#include <cstring>
#include <utility>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace matting {

using MattingOutput = jinq::models::io_define::matting::std_matting_output;
using jinq::models::backend::NamedTensor;
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
StatusCode ModNetMatting<INPUT, OUTPUT>::on_init(const toml::table& params) {
    (void)params;
    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected modnet input shape: " << input_info.to_string()
                   << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "invalid modnet input tensor size: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> ModNetMatting<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    _m_input_size_user = input_image.size();

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
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data size mismatches input tensor size";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());

    std::vector<NamedTensor> inputs;
    inputs.push_back(std::move(named));
    return inputs;
}

template<typename INPUT, typename OUTPUT>
StatusCode ModNetMatting<INPUT, OUTPUT>::postprocess(
    const std::vector<NamedTensor>& outputs, OUTPUT& output) {
    if (outputs.empty()) {
        LOG(ERROR) << "modnet output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = outputs.front().tensor;
    const auto* host_data = tensor.template data<float>();
    if (tensor.element_count() < static_cast<int64_t>(_m_input_size_host.area())) {
        LOG(ERROR) << "unexpected modnet output shape: "
                   << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    cv::Mat result_image(_m_input_size_host, CV_32FC1, const_cast<float*>(host_data));
    cv::resize(result_image, result_image, _m_input_size_user, 0.0, 0.0, cv::INTER_LINEAR);
    result_image *= 255.0;
    result_image.convertTo(result_image, CV_8UC1);

    MattingOutput internal_out;
    internal_out.matting_result = std::move(result_image);
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
ModNetMatting<INPUT, OUTPUT>::ModNetMatting()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("MODNET_MATTING") {}

}
}
}
