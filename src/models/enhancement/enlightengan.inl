/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: enlightengan.inl
 * Date: 22-6-13
 ************************************************/

#include "enlightengan.h"

#include <cmath>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "glog/logging.h"

#include "models/backend/f32_output.h"
#include "models/backend/request_geometry.h"

namespace jinq {
namespace models {
namespace enhancement {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;
using jinq::models::io_define::enhancement::std_enhancement_output;

namespace {

bool copy_mat_to_buffer(const cv::Mat &image, std::vector<uint8_t> &buffer) {
    const auto bytes = image.total() * image.elemSize();
    if (bytes != buffer.size()) {
        return false;
    }
    if (image.isContinuous()) {
        std::memcpy(buffer.data(), image.data, bytes);
    } else {
        uint8_t *dst = buffer.data();
        for (int row = 0; row < image.rows; ++row) {
            const auto row_bytes = static_cast<size_t>(image.cols) * image.elemSize();
            std::memcpy(dst, image.ptr(row), row_bytes);
            dst += row_bytes;
        }
    }
    return true;
}

/*** round a dimension up to a multiple of 16 with a real ceil (the network
 * downsamples by 16, so input dims must stay 16-aligned). Integer division
 * would truncate and silently round DOWN, so divide in floating point. */
inline int align_up_16(int value) {
    return static_cast<int>(std::ceil(static_cast<float>(value) / 16.0f)) * 16;
}

} // namespace

template <typename INPUT, typename OUTPUT> StatusCode EnlightenGan<INPUT, OUTPUT>::on_init(const toml::table &params) {
    const auto &inputs = this->session().inputs();
    if (inputs.size() != 2 || this->session().outputs().size() != 1) {
        LOG(ERROR) << "unexpected enlighten gan io count, expected input_src/input_gray and output";
        return StatusCode::MODEL_INIT_FAILED;
    }
    const auto *input_src_info = &inputs.front();
    const auto *input_gray_info = &inputs.back();
    if (input_src_info->name != "input_src" || input_gray_info->name != "input_gray") {
        for (const auto &info : inputs) {
            if (info.name == "input_src") {
                input_src_info = &info;
            } else if (info.name == "input_gray") {
                input_gray_info = &info;
            }
        }
    }
    if (input_src_info->name != "input_src" || input_gray_info->name != "input_gray" || input_src_info->shape.size() != 4 ||
        input_src_info->shape[1] != 3 || input_gray_info->shape.size() != 4 || input_gray_info->shape[1] != 1) {
        LOG(ERROR) << "unexpected enlighten gan input io, expected input_src [N,3,H,W] and "
                   << "input_gray [N,1,H,W]";
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_dynamic = false;
    _m_input_size_host = {};
    const auto src_height = input_src_info->shape[2];
    const auto src_width = input_src_info->shape[3];
    if (src_height > 0 && src_width > 0) {
        // fixed-shape session: every request runs at the declared input size
        _m_input_size_host.height = static_cast<int>(src_height);
        _m_input_size_host.width = static_cast<int>(src_width);
    } else if (params.contains("model_input_image_size")) {
        // dynamic session input (unset dims): fall back to the size declared
        // in the model config, mirroring the msocrnet/yolov* handling
        const toml::array *size = params["model_input_image_size"].as_array();
        if (size != nullptr && size->size() == 2) {
            _m_input_size_host.height = static_cast<int>((*size)[0].value_or<int64_t>(0));
            _m_input_size_host.width = static_cast<int>((*size)[1].value_or<int64_t>(0));
        }
        if (_m_input_size_host.area() <= 0) {
            LOG(ERROR) << "invalid params.model_input_image_size for enlighten gan";
            return StatusCode::MODEL_INIT_FAILED;
        }
    } else {
        // dynamic session input with no declared size: the input size follows
        // each request in preprocess (aligned up to a multiple of 16)
        _m_input_dynamic = true;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> EnlightenGan<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    if (input_image.size().height < 10 || input_image.size().width < 10 || (input_image.channels() != 3 && input_image.channels() != 4)) {
        LOG(ERROR) << "invalid enlighten gan image size or channels: " << input_image.size() << ", channels=" << input_image.channels();
        return {};
    }

    // fixed-shape sessions always run at the declared size; only sessions
    // without one follow the request, padded up to a multiple of 16
    cv::Size network_size = _m_input_size_host;
    if (_m_input_dynamic) {
        network_size.height = align_up_16(input_image.size().height);
        network_size.width = align_up_16(input_image.size().width);
    }
    if (network_size.area() <= 0) {
        LOG(ERROR) << "enlighten gan input size unresolved: " << network_size;
        return {};
    }

    // resize FIRST, convert colors LAST: cv::resize rebuilds the destination
    // with the source type, so resizing the already-converted output_src from
    // the RAW input silently discards the RGB conversion (3-channel requests
    // were fed BGR) and a 4-channel request smuggled its alpha plane into the
    // normalized tensor
    cv::Mat output_src;
    input_image.copyTo(output_src);
    if (output_src.size() != network_size) {
        cv::resize(output_src, output_src, network_size);
    }
    if (output_src.channels() == 4) {
        cv::cvtColor(output_src, output_src, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(output_src, output_src, cv::COLOR_BGR2RGB);
    }
    if (output_src.type() != CV_32FC3) {
        output_src.convertTo(output_src, CV_32FC3);
    }
    output_src /= 255.0;
    cv::subtract(output_src, cv::Scalar(0.5, 0.5, 0.5), output_src);
    cv::divide(output_src, cv::Scalar(0.5, 0.5, 0.5), output_src);

    std::vector<cv::Mat> src_split;
    cv::split(output_src, src_split);
    cv::Mat output_gray = 1.0 - (0.299 * (src_split[0] + 1.0) + 0.587 * (src_split[1] + 1.0) + 0.114 * (src_split[2] + 1.0)) * 0.5;

    std::vector<NamedTensor> tensors;
    NamedTensor input_src;
    input_src.name = "input_src";
    input_src.tensor = jinq::models::backend::Tensor::make<float>({1, 3, network_size.height, network_size.width});
    const auto input_src_chw_data = CvUtils::convert_to_chw_vec(output_src);
    if (input_src_chw_data.size() * sizeof(float) != input_src.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed enlighten gan src tensor size mismatch";
        return {};
    }
    std::memcpy(input_src.tensor.buffer.data(), input_src_chw_data.data(), input_src.tensor.byte_size());

    NamedTensor input_gray;
    input_gray.name = "input_gray";
    input_gray.tensor = jinq::models::backend::Tensor::make<float>({1, 1, network_size.height, network_size.width});
    if (!copy_mat_to_buffer(output_gray, input_gray.tensor.buffer)) {
        LOG(ERROR) << "preprocessed enlighten gan gray tensor size mismatch";
        return {};
    }
    tensors.push_back(std::move(input_src));
    tensors.push_back(std::move(input_gray));
    return tensors;
}

template <typename INPUT, typename OUTPUT>
StatusCode EnlightenGan<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                    const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "enlighten gan output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto source_status = jinq::models::backend::validated_source_size(context, "enlighten gan");
    if (source_status != StatusCode::OK) {
        return source_status;
    }
    // the network geometry travels per request in the context; the model is
    // shared across requests and batch items interleave pre/postprocess calls
    const auto network_size = context.network_size;
    if (network_size.area() <= 0) {
        LOG(ERROR) << "enlighten gan network size unresolved: " << network_size;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }
    jinq::models::backend::F32OutputView output_view;
    const auto output_status = jinq::models::backend::validated_f32_first_output(
        outputs, {jinq::models::backend::DType::F32, 4, {1, 3, network_size.height, network_size.width}}, "enlighten gan",
        &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto *host_data = output_view.data;
    const auto pixel_count = static_cast<size_t>(network_size.height) * network_size.width;
    std::vector<uchar> output_img_data(pixel_count * 3);
    for (auto row = 0; row < network_size.height; ++row) {
        for (auto col = 0; col < network_size.width; ++col) {
            for (auto c = 0; c < 3; ++c) {
                const auto hwc_idx = row * network_size.width * 3 + col * 3 + c;
                const auto chw_idx = c * pixel_count + row * network_size.width + col;
                auto pix_val_f = (host_data[chw_idx] + 1.0) * 255.0 / 2.0;
                if (pix_val_f < 0.0) {
                    pix_val_f = 0.0;
                }
                if (pix_val_f >= 255) {
                    pix_val_f = 255.0;
                }
                output_img_data[hwc_idx] = static_cast<uchar>(pix_val_f);
            }
        }
    }

    std_enhancement_output internal_out;
    cv::Mat result_image(network_size, CV_8UC3, output_img_data.data());
    cv::cvtColor(result_image, internal_out.enhancement_result, cv::COLOR_RGB2BGR);
    if (internal_out.enhancement_result.size() != context.source_size) {
        cv::resize(internal_out.enhancement_result, internal_out.enhancement_result, context.source_size);
    }
    if (!context.source_image.empty() && context.source_image.channels() == 4) {
        std::vector<cv::Mat> input_image_split;
        cv::split(context.source_image, input_image_split);
        std::vector<cv::Mat> output_image_split;
        cv::split(internal_out.enhancement_result, output_image_split);
        output_image_split.push_back(input_image_split[3]);
        cv::merge(output_image_split, internal_out.enhancement_result);
    }

    output = std::move(internal_out);
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
EnlightenGan<INPUT, OUTPUT>::EnlightenGan() : jinq::models::BackendCvModel<INPUT, OUTPUT>("ENLIGHTENGAN") {}

} // namespace enhancement
} // namespace models
} // namespace jinq
