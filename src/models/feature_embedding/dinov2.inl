/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: dinov2.inl
 * Date: 23-6-12
 ************************************************/

#include "dinov2.h"

#include <cmath>

#include "glog/logging.h"

#include "models/backend/f32_output.h"
#include "models/backend/model_runtime.h"
#include <opencv2/opencv.hpp>

namespace jinq {
namespace models {
namespace feature_embedding {
using jinq::common::StatusCode;

using FeatureEmbeddingOutput = jinq::models::io_define::feature_embedding::std_feature_embedding_output;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode Dinov2<INPUT, OUTPUT>::on_init(const toml::table &params) {
    const auto input_info =
        jinq::models::backend::SessionIoValidator(this->session()).input().f32().rank(4).nchw().channels(3).static_shape().validate();
    if (!input_info.ok()) {
        LOG(ERROR) << "unexpected feature embedding input shape: " << input_info.error << ", expected static [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size.height = static_cast<int>(input_info.value.shape[2]);
    _m_input_tensor_size.width = static_cast<int>(input_info.value.shape[3]);
    if (_m_input_tensor_size.area() <= 0) {
        LOG(ERROR) << "invalid dinov2 input shape: " << input_info.error;
        return StatusCode::MODEL_INIT_FAILED;
    }
    (void)params;
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
                                              const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "feature embedding model output tensor is empty";
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
    const auto *embedding_data = output_view.value.data;
    const auto embedding_count = tensor.element_count();

    // the exported output node is the ViT [CLS] token embedding; the whole
    // vector is returned as the feature embedding, no argmax / class mapping
    FeatureEmbeddingOutput internal_out;
    internal_out.embedding.assign(embedding_data, embedding_data + embedding_count);

    if (context.params != nullptr && context.params->get_bool("normalize", false)) {
        double norm_sq = 0.0;
        for (const float value : internal_out.embedding) {
            norm_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        const double norm = std::sqrt(norm_sq);
        if (norm > 1e-12) {
            for (float &value : internal_out.embedding) {
                value = static_cast<float>(static_cast<double>(value) / norm);
            }
        }
    }

    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT> Dinov2<INPUT, OUTPUT>::Dinov2() : jinq::models::BackendCvModel<INPUT, OUTPUT>("DINOV2") {}

} // namespace feature_embedding
} // namespace models
} // namespace jinq
