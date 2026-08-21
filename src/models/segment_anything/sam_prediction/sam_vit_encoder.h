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

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/backend/session.h"

namespace jinq {
namespace models {
namespace segment_anything {
using jinq::common::StatusCode;

/***
 * SAM ViT image encoder backed by the unified inference-session layer.
 */
class SamVitEncoder {
  public:
    explicit SamVitEncoder(
        std::unique_ptr<jinq::models::backend::InferenceSession> session);
    ~SamVitEncoder();

    SamVitEncoder(const SamVitEncoder&) = delete;
    SamVitEncoder& operator=(const SamVitEncoder&) = delete;

    StatusCode init();

    StatusCode encode(
        const cv::Mat& input_image, std::vector<float>& image_embeddings);

    std::vector<int> get_encoder_input_shape() const;

    bool is_successfully_initialized() const;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

} // namespace segment_anything
} // namespace models
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_SAM_VIT_ENCODER_H
