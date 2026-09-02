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

template <typename INPUT, typename OUTPUT> StatusCode MobileNetv2<INPUT, OUTPUT>::on_init(const toml::table &params) {
    const auto input_info =
        jinq::models::backend::SessionIoValidator(this->session()).input().f32().rank(4).nhwc().channels(3).static_shape().validate();
    if (!input_info.ok()) {
        LOG(ERROR) << "unexpected mobilenetv2 input shape: " << input_info.error << ", expected static [N,H,W,3] (nhwc)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size.height = static_cast<int>(input_info.value.shape[1]);
    _m_input_tensor_size.width = static_cast<int>(input_info.value.shape[2]);

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
    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .resize({256, 256})
                      .center_crop(_m_input_tensor_size)
                      .bgr_to_rgb()
                      .to_float()
                      .subtract({123.68f, 116.78f, 103.94f})
                      .divide({58.395f, 57.12f, 57.375f})
                      .mat();
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return std::move(result.value);
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> MobileNetv2<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    const cv::Mat tmp = preprocess_mat(input_image);
    auto result = jinq::models::backend::ImagePipeline(tmp).nhwc(this->session().inputs().front().name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode MobileNetv2<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                   const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
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
    const auto max_score = std::max_element(internal_out.scores.begin(), internal_out.scores.end());
    const auto cls_id = static_cast<int>(std::distance(internal_out.scores.begin(), max_score));
    internal_out.class_id = cls_id;
    const auto name_iter = _m_class_id2names.find(static_cast<uint16_t>(cls_id));
    if (name_iter != _m_class_id2names.end()) {
        internal_out.category = name_iter->second;
    }
    // request-level top_k: keep the k highest scores (descending) to shrink
    // the payload; the full class-index ordered array is the default
    int keep = static_cast<int>(score_count);
    if (context.params != nullptr) {
        keep = context.params->get_i32("top_k", keep);
    }
    if (keep < static_cast<int>(score_count)) {
        std::vector<int> order(static_cast<size_t>(score_count));
        for (int idx = 0; idx < static_cast<int>(score_count); ++idx) {
            order[static_cast<size_t>(idx)] = idx;
        }
        std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                          [&internal_out](int lhs, int rhs) {
                              return internal_out.scores[static_cast<size_t>(lhs)] >
                                     internal_out.scores[static_cast<size_t>(rhs)];
                          });
        order.resize(static_cast<size_t>(keep));
        std::vector<float> top_scores;
        top_scores.reserve(order.size());
        for (const int idx : order) {
            top_scores.push_back(internal_out.scores[static_cast<size_t>(idx)]);
        }
        internal_out.scores = std::move(top_scores);
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
