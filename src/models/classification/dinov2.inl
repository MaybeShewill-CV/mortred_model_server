/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: dinov2.inl
 * Date: 23-6-12
 ************************************************/

#include "dinov2.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "glog/logging.h"

#include "models/backend/f32_output.h"
#include "models/backend/model_runtime.h"
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {
namespace classification {
using jinq::common::StatusCode;

using ClassificationOutput = jinq::models::io_define::classification::std_classification_output;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode Dinov2<INPUT, OUTPUT>::on_init(const toml::table &params) {
    const auto input_info =
        jinq::models::backend::SessionIoValidator(this->session()).input().f32().rank(4).nchw().channels(3).static_shape().validate();
    if (!input_info.ok()) {
        LOG(ERROR) << "unexpected classification input shape: " << input_info.error << ", expected static [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size.height = static_cast<int>(input_info.value.shape[2]);
    _m_input_tensor_size.width = static_cast<int>(input_info.value.shape[3]);
    if (_m_input_tensor_size.area() <= 0) {
        LOG(ERROR) << "invalid dinov2 input shape: " << input_info.error;
        return StatusCode::MODEL_INIT_FAILED;
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

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> Dinov2<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // rgb -> resize -> [0,1] -> clip mean/std, emitted as f32 nchw
    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .bgr_to_rgb()
                      .resize(_m_input_tensor_size)
                      .to_float()
                      .scale(1.0f / 255.0f)
                      .subtract({0.48145466f, 0.4578275f, 0.40821073f})
                      .divide({0.26862954f, 0.26130258f, 0.27577711f})
                      .nchw(this->session().inputs().front().name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode Dinov2<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                              const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "classification model output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto output_rank = outputs.front().tensor.shape.size();
    auto output_view = jinq::models::backend::OutputReader(outputs, outputs.front().name)
                           .f32()
                           .shape(output_rank == 1 ? std::vector<int64_t>{-1} : std::vector<int64_t>{1, -1})
                           .finite()
                           .read();
    if (!output_view.ok()) {
        return output_view.status;
    }
    const auto &tensor = *output_view.value.tensor;
    const auto *scores = output_view.value.data;
    const auto score_count = tensor.element_count();

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
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT> Dinov2<INPUT, OUTPUT>::Dinov2() : jinq::models::BackendCvModel<INPUT, OUTPUT>("DINOV2") {}

} // namespace classification
} // namespace models
} // namespace jinq
