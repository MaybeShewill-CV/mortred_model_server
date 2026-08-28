/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: autoencoder_kl.inl
 * Date: 26-8-17
 ************************************************/

#include "autoencoder_kl.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace diffusion {

using VaeInput = jinq::models::io_define::diffusion::std_vae_decode_input;
using VaeOutput = jinq::models::io_define::diffusion::std_vae_decode_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> jinq::models::PreparedInput AutoEncoderKL<INPUT, OUTPUT>::prepare_inputs(const INPUT &input) {
    const auto &input_info = this->session().inputs().front();
    if (input_info.dynamic) {
        LOG(ERROR) << "vae decoder input must be static, got " << input_info.to_string();
        return {};
    }
    jinq::models::PreparedInput prepared;
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = input_info.name;
    named.tensor = jinq::models::backend::Tensor::make<float>(input_info.shape);
    if (input.decode_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "vae decode data element count " << input.decode_data.size() << " mismatches session input "
                   << input_info.to_string();
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), input.decode_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    prepared.inputs = std::move(inputs);
    return prepared;
}

template <typename INPUT, typename OUTPUT>
StatusCode AutoEncoderKL<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                     const jinq::models::InferenceContext & /*context*/, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "vae decoder output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto &tensor = outputs.front().tensor;
    if (tensor.shape.size() != 4) {
        LOG(ERROR) << "unexpected vae decoder output shape: " << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto *data = tensor.data<float>();
    const auto channels = tensor.shape[1];
    const auto height = tensor.shape[2];
    const auto width = tensor.shape[3];
    const auto element_count = tensor.element_count();

    // [-1,1] -> [0,1] -> uint8, chw -> hwc -> rgb2bgr
    std::vector<uint8_t> decode_out;
    decode_out.reserve(static_cast<size_t>(element_count));
    for (int64_t idx = 0; idx < element_count; ++idx) {
        auto pix_value = data[idx] / 2.0f + 0.5f;
        pix_value = pix_value < 0.0f ? 0.0f : pix_value;
        pix_value = pix_value > 1.0f ? 1.0f : pix_value;
        decode_out.push_back(static_cast<uint8_t>(pix_value * 255.0f));
    }
    auto hwc_data = jinq::common::CvUtils::convert_to_hwc_vec<uint8_t>(decode_out, static_cast<int>(channels), static_cast<int>(height),
                                                                       static_cast<int>(width));

    auto mat_dtype = CV_8UC3;
    if (channels == 1) {
        mat_dtype = CV_8UC1;
    } else if (channels == 4) {
        mat_dtype = CV_8UC4;
    }
    VaeOutput internal_out;
    const cv::Mat decode_image(cv::Size(static_cast<int>(width), static_cast<int>(height)), mat_dtype, hwc_data.data());
    if (channels == 3) {
        cv::cvtColor(decode_image, internal_out.decode_output, cv::COLOR_RGB2BGR);
    } else {
        internal_out.decode_output = decode_image.clone();
    }
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
AutoEncoderKL<INPUT, OUTPUT>::AutoEncoderKL() : jinq::models::BackendCvModel<INPUT, OUTPUT>("AUTOENCODER_KL") {}

} // namespace diffusion
} // namespace models
} // namespace jinq
