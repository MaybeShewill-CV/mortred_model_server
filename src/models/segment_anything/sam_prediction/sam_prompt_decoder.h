/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: sam_prompt_decoder.h
* Date: 23-6-7
************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_PROMPT_DECODER_H
#define MORTRED_MODEL_SERVER_SAM_PROMPT_DECODER_H

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
 * SAM prompt decoder backed by the unified inference-session layer. ONNX
 * models may expose the full-resolution "masks" output, while TensorRT
 * engines normally expose "low_res_masks" plus IoU predictions.
 */
class SamPromptDecoder {
  public:
    explicit SamPromptDecoder(
        std::unique_ptr<jinq::models::backend::InferenceSession> session);
    ~SamPromptDecoder();

    SamPromptDecoder(const SamPromptDecoder&) = delete;
    SamPromptDecoder& operator=(const SamPromptDecoder&) = delete;

    StatusCode init();

    void set_ori_image_size(const cv::Size& ori_image_size);

    void set_encoder_input_size(const cv::Size& input_node_size);

    StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<cv::Rect2f>& bboxes,
        std::vector<cv::Mat>& predicted_masks);

    StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<std::vector<cv::Point2f>>& points,
        std::vector<cv::Mat>& predicted_masks);

    bool is_successfully_initialized() const;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

} // namespace segment_anything
} // namespace models
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_SAM_PROMPT_DECODER_H
