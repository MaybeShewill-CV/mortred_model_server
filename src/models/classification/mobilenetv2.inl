/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: mobilenetv2.inl
 * Date: 22-6-13
 ************************************************/

#include "mobilenetv2.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "glog/logging.h"

#include "models/backend/f32_output.h"
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {
namespace classification {
using jinq::common::StatusCode;

using ClassificationOutput = jinq::models::io_define::classification::std_classification_output;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode MobileNetv2<INPUT, OUTPUT>::on_init(const toml::table &params) {
    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[3] != 3) {
        LOG(ERROR) << "unexpected classification input shape: " << input_info.to_string() << ", expected [N,H,W,3] (nhwc)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size.height = static_cast<int>(input_info.shape[1]);
    _m_input_tensor_size.width = static_cast<int>(input_info.shape[2]);

    if (params.contains("model_input_image_size")) {
        const toml::array *size = params["model_input_image_size"].as_array();
        if (size == nullptr || size->size() != 2) {
            LOG(ERROR) << "params key 'model_input_image_size' must be [height, width]";
            return StatusCode::MODEL_INIT_FAILED;
        }
        _m_input_tensor_size.height = static_cast<int>((*size)[0].value_or<int64_t>(0));
        _m_input_tensor_size.width = static_cast<int>((*size)[1].value_or<int64_t>(0));
        if (_m_input_tensor_size.width <= 0 || _m_input_tensor_size.height <= 0) {
            LOG(ERROR) << "invalid model_input_image_size";
            return StatusCode::MODEL_INIT_FAILED;
        }
    }

    if (params.contains("class_name_file")) {
        const std::string file_path = params["class_name_file"].value_or<std::string>("");
        if (!jinq::common::FilePathUtil::is_file_exist(file_path)) {
            LOG(WARNING) << "class name file: " << file_path << " not exist";
        } else {
            std::ifstream file(file_path, std::ios::in);
            std::string line;
            uint16_t line_num = 0;
            while (std::getline(file, line)) {
                _m_class_id2names[line_num] = line;
                ++line_num;
            }
        }
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> cv::Mat MobileNetv2<INPUT, OUTPUT>::preprocess_mat(const cv::Mat &input_image) {
    // resize -> center crop -> rgb -> per channel normalize (f32 nhwc)
    cv::Mat tmp;
    cv::resize(input_image, tmp, cv::Size(256, 256));
    const auto dw = static_cast<int>(std::floor((256 - _m_input_tensor_size.width) / 2));
    const auto dh = static_cast<int>(std::floor((256 - _m_input_tensor_size.height) / 2));
    tmp = tmp(cv::Rect(dw, dh, _m_input_tensor_size.width, _m_input_tensor_size.height));

    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
    tmp.convertTo(tmp, CV_32FC3);
    cv::subtract(tmp, cv::Scalar(123.68f, 116.78f, 103.94f), tmp);
    cv::divide(tmp, cv::Scalar(58.395f, 57.12f, 57.375f), tmp);
    return tmp;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> MobileNetv2<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    const cv::Mat tmp = preprocess_mat(input_image);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>({1, _m_input_tensor_size.height, _m_input_tensor_size.width, 3});
    const auto bytes = tmp.total() * tmp.elemSize();
    if (bytes != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed image byte size " << bytes << " mismatches tensor byte size " << named.tensor.byte_size();
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), tmp.data, bytes);
    inputs.push_back(std::move(named));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
StatusCode MobileNetv2<INPUT, OUTPUT>::run_batch(const std::vector<INPUT> &in, std::vector<OUTPUT> &out,
                                                 std::vector<StatusCode> &item_status) {
    out.clear();
    item_status.assign(in.size(), StatusCode::OK);
    if (in.empty()) {
        LOG(ERROR) << "batch input is empty";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    if (!this->is_successfully_initialized()) {
        LOG(ERROR) << "model is not successfully initialized, refuse to run batch";
        out.assign(in.size(), OUTPUT{});
        item_status.assign(in.size(), StatusCode::MODEL_INIT_FAILED);
        return StatusCode::MODEL_INIT_FAILED;
    }

    // per-item preprocess: a failing item is isolated and skipped from the
    // packed run; its batch mates still get their results
    std::vector<cv::Mat> mats;
    std::vector<size_t> valid_items;
    mats.reserve(in.size());
    valid_items.reserve(in.size());
    for (size_t idx = 0; idx < in.size(); ++idx) {
        StatusCode image_status = StatusCode::OK;
        std::string image_error;
        const cv::Mat image = this->load_model_image(in[idx], &image_status, &image_error);
        if (image.empty()) {
            LOG(ERROR) << "batch item " << idx << ": " << (image_error.empty() ? "image is empty" : image_error);
            item_status[idx] = image_status == StatusCode::OK ? StatusCode::MODEL_EMPTY_INPUT_IMAGE : image_status;
            continue;
        }
        mats.push_back(preprocess_mat(image));
        valid_items.push_back(idx);
    }
    if (valid_items.empty()) {
        out.assign(in.size(), OUTPUT{});
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    backend::NamedTensor named;
    if (!this->pack_nhwc_batch(this->session().inputs().front().name, mats, &named)) {
        // packing failure is input-level: attribute it to the valid items
        out.assign(in.size(), OUTPUT{});
        for (size_t idx : valid_items) {
            item_status[idx] = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    std::vector<backend::NamedTensor> inputs;
    inputs.push_back(std::move(named));
    std::vector<backend::NamedTensor> outputs;
    const auto status = this->session().run(inputs, outputs);
    if (status != StatusCode::OK) {
        // session-level failure cannot be attributed to an item: broadcast
        out.assign(in.size(), OUTPUT{});
        for (size_t idx : valid_items) {
            item_status[idx] = status;
        }
        return status;
    }
    if (outputs.empty()) {
        LOG(ERROR) << "batched classification output is empty";
        out.assign(in.size(), OUTPUT{});
        for (size_t idx : valid_items) {
            item_status[idx] = StatusCode::MODEL_EMPTY_OUTPUT;
        }
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto items = this->split_batch_output(outputs.front().tensor, static_cast<int64_t>(in.size()));
    if (items.size() != in.size()) {
        out.assign(in.size(), OUTPUT{});
        for (size_t idx : valid_items) {
            item_status[idx] = StatusCode::MODEL_EMPTY_OUTPUT;
        }
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    out.resize(in.size());
    StatusCode aggregate = StatusCode::OK;
    for (size_t pos = 0; pos < valid_items.size(); ++pos) {
        const size_t idx = valid_items[pos];
        std::vector<backend::NamedTensor> item_outputs;
        item_outputs.push_back({outputs.front().name, items[idx]});
        const auto post_status = postprocess(item_outputs, jinq::models::backend::InferenceContext{}, out[idx]);
        item_status[idx] = post_status;
        if (post_status != StatusCode::OK) {
            aggregate = post_status;
        }
    }
    return aggregate;
}

template <typename INPUT, typename OUTPUT>
StatusCode MobileNetv2<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                   const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "classification model output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto output_rank = outputs.front().tensor.shape.size();
    jinq::models::backend::TensorContract output_contract;
    output_contract.dtype = jinq::models::backend::DType::F32;
    output_contract.rank = output_rank == 1 ? 1 : 2;
    output_contract.shape = output_rank == 1 ? std::vector<int64_t>{-1} : std::vector<int64_t>{1, -1};
    jinq::models::backend::F32OutputView output_view;
    const auto output_status = jinq::models::backend::validated_f32_first_output(outputs, output_contract, "classification", &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto &tensor = *output_view.tensor;
    const auto *scores = output_view.data;
    const auto score_count = tensor.element_count();

    ClassificationOutput internal_out;
    internal_out.scores.reserve(static_cast<size_t>(score_count));
    for (int64_t idx = 0; idx < score_count; ++idx) {
        internal_out.scores.push_back(scores[idx]);
    }
    const auto max_score = std::max_element(internal_out.scores.begin(), internal_out.scores.end());
    const auto cls_id = static_cast<int>(std::distance(internal_out.scores.begin(), max_score));
    internal_out.class_id = cls_id;
    const auto name_iter = _m_class_id2names.find(static_cast<uint16_t>(cls_id));
    if (name_iter != _m_class_id2names.end()) {
        internal_out.category = name_iter->second;
    }
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
MobileNetv2<INPUT, OUTPUT>::MobileNetv2() : jinq::models::BackendCvModel<INPUT, OUTPUT>("MOBILENETV2") {}

} // namespace classification
} // namespace models
} // namespace jinq
