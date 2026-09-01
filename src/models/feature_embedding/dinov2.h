/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: dinov2.h
 * Date: 23-6-12
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DINOV2_H
#define MORTRED_MODEL_SERVER_DINOV2_H

#include <string>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/io/feature_embedding.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace feature_embedding {
using jinq::common::StatusCode;

/***
 * DINOv2 (ViT) image feature extractor. The model is a vision transformer:
 * the exported output tensor is the [CLS] token embedding, NOT a classification
 * score distribution. The request-overridable `normalize` param (declared in
 * the feature_embedding task catalog) L2-normalizes the returned embedding.
 */
template <typename INPUT, typename OUTPUT> class Dinov2 : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    Dinov2();
    ~Dinov2() override = default;

    Dinov2(const Dinov2 &transformer) = delete;
    Dinov2 &operator=(const Dinov2 &transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext &context, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    // network input tensor size
    cv::Size _m_input_tensor_size = cv::Size(224, 224);
};

} // namespace feature_embedding
} // namespace models
} // namespace jinq

#include "dinov2.inl"

#endif // MORTRED_MODEL_SERVER_DINOV2_H
