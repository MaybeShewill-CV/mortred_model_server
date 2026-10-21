/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_vit_encoder.h
 * Date: 23-6-7
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_VIT_ENCODER_H
#define MORTRED_MODEL_SERVER_SAM_VIT_ENCODER_H

#include <memory>
#include <vector>

#include "toml/toml.hpp"
#include <opencv2/opencv.hpp>

#include "common/status_code.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"

namespace mortred {
namespace models {
namespace segment_anything {
using mortred::common::StatusCode;

/***
 * SAM ViT image encoder backed by the unified inference-session layer.
 * The session is owned by the enclosing model (MultiSessionModel); this class
 * only drives it and never frees it.
 */
class SamVitEncoder {
  public:
    explicit SamVitEncoder(mortred::models::backend::InferenceSession *session);
    ~SamVitEncoder();

    SamVitEncoder(const SamVitEncoder &) = delete;
    SamVitEncoder &operator=(const SamVitEncoder &) = delete;

    StatusCode init();

    StatusCode encode(const cv::Mat &input_image, std::vector<float> &image_embeddings);

    std::vector<int> get_encoder_input_shape() const;

    bool is_successfully_initialized() const;

  protected:
    /*** Empty output vector → MODEL_EMPTY_OUTPUT; non-empty dtype/size miss → MODEL_OUTPUT_CONTRACT_FAILED. */
    static StatusCode classify_encoder_outputs(const std::vector<mortred::models::backend::NamedTensor> &outputs);

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

} // namespace segment_anything
} // namespace models
} // namespace mortred

#endif // MORTRED_MODEL_SERVER_SAM_VIT_ENCODER_H
