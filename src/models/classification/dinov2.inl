/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dinov2.cpp
************************************************/

#include "dinov2.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {
namespace classification {

using ClassificationOutput = jinq::models::io_define::classification::std_classification_output;
using jinq::models::backend::NamedTensor;

template<typename INPUT, typename OUTPUT>
jinq::common::StatusCode Dinov2<INPUT, OUTPUT>::on_init(const toml::table& params) {
    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected classification input shape: " << input_info.to_string()
                   << ", expected [N,3,H,W] (nchw)";
        return jinq::common::StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size.height = static_cast<int>(input_info.shape[2]);
    _m_input_tensor_size.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_tensor_size.area() <= 0) {
        LOG(ERROR) << "invalid dinov2 input shape: " << input_info.to_string();
        return jinq::common::StatusCode::MODEL_INIT_FAILED;
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
    return jinq::common::StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> Dinov2<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    // rgb -> resize -> [0,1] -> clip mean/std, emitted as f32 nchw
    cv::Mat tmp;
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);
    cv::resize(tmp, tmp, _m_input_tensor_size);
    tmp.convertTo(tmp, CV_32FC3);
    cv::divide(tmp, 255.0, tmp);
    cv::subtract(tmp, cv::Scalar(0.48145466, 0.4578275, 0.40821073), tmp);
    cv::divide(tmp, cv::Scalar(0.26862954, 0.26130258, 0.27577711), tmp);

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, 3, _m_input_tensor_size.height, _m_input_tensor_size.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data size mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template<typename INPUT, typename OUTPUT>
jinq::common::StatusCode Dinov2<INPUT, OUTPUT>::postprocess(
    const std::vector<NamedTensor>& outputs, OUTPUT& output) {
    if (outputs.empty()) {
        LOG(ERROR) << "classification model output tensor is empty";
        return jinq::common::StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = outputs.front().tensor;
    const auto* scores = tensor.data<float>();
    const auto score_count = tensor.element_count();
    if (score_count <= 0) {
        LOG(ERROR) << "classification model output tensor is empty";
        return jinq::common::StatusCode::MODEL_EMPTY_OUTPUT;
    }

    ClassificationOutput internal_out;
    internal_out.scores.reserve(static_cast<size_t>(score_count));
    for (int64_t idx = 0; idx < score_count; ++idx) {
        internal_out.scores.push_back(scores[idx]);
    }
    // output node "cls_tokens" is the ViT [CLS] embedding; argmax matches the
    // exported head and the existing golden expectations
    const auto max_score = std::max_element(internal_out.scores.begin(), internal_out.scores.end());
    const auto cls_id = static_cast<int>(std::distance(internal_out.scores.begin(), max_score));
    internal_out.class_id = cls_id;
    const auto name_iter = _m_class_id2names.find(static_cast<uint16_t>(cls_id));
    if (name_iter != _m_class_id2names.end()) {
        internal_out.category = name_iter->second;
    }
    output = std::move(internal_out);
    return jinq::common::StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
Dinov2<INPUT, OUTPUT>::Dinov2()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("DINOV2") {}

}
}
}
