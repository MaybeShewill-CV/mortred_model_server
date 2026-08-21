/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: openai_clip.h
* Date: 23-6-26
************************************************/

#ifndef MORTRED_MODEL_SERVER_OPENAICLIP_H
#define MORTRED_MODEL_SERVER_OPENAICLIP_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/clip/simple_tokenizer.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace clip {
using jinq::common::StatusCode;

/***
 * OpenAI CLIP multi-engine model. The visual and text encoders are unified
 * inference sessions, while the BPE tokenizer is model-local preprocessing.
 */
template<typename INPUT, typename OUTPUT>
class OpenAiClip : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    OpenAiClip();
    ~OpenAiClip() override = default;

    OpenAiClip(const OpenAiClip& transformer) = delete;
    OpenAiClip& operator=(const OpenAiClip& transformer) = delete;

    StatusCode get_textual_embedding(
        const std::string& input_text, std::vector<float>& text_embeddings);

    StatusCode get_visual_embedding(
        const cv::Mat& input_image, std::vector<float>& image_embeddings);

    StatusCode texts2img(
        const std::vector<std::string>& input_texts, const cv::Mat& input_image,
        std::vector<float>& simi_scores);

    StatusCode imgs2text(
        const std::vector<cv::Mat>& input_images, const std::string& input_text,
        std::vector<float>& simi_scores);

  private:
    using ClipInput = jinq::models::io_define::clip::clip_input;
    using ClipOutput = jinq::models::io_define::clip::clip_output;
    using ClipTaskType = jinq::models::io_define::clip::ClipTaskType;

    StatusCode on_init(const toml::table& params) override;

    StatusCode run_sessions(const INPUT& input, OUTPUT& output) override;

    StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    StatusCode encode_text(
        const std::string& input_text, std::vector<float>& text_embeddings) const;

    StatusCode encode_image(
        const cv::Mat& input_image, std::vector<float>& image_embeddings) const;

    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    void tokenize(const std::string& input_text, std::vector<int32_t>& token_ids) const;

    const jinq::models::backend::NamedTensor* find_output(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const std::string& name) const;

    static const jinq::models::backend::TensorInfo* find_info(
        const jinq::models::backend::InferenceSession& session, const std::string& name);

    static StatusCode validate_visual_io(
        const jinq::models::backend::InferenceSession& session);

    static StatusCode validate_text_io(
        const jinq::models::backend::InferenceSession& session);

    static bool normalize_embedding(std::vector<float>& embedding);

    static bool embeddings_compatible(
        const std::vector<float>& lhs, const std::vector<float>& rhs);

    std::unique_ptr<jinq::models::backend::InferenceSession> _m_visual_encoder;
    std::unique_ptr<jinq::models::backend::InferenceSession> _m_text_encoder;
    SimpleTokenizer _m_tokenizer;
    int _m_context_length = 77;
    bool _m_truncate_context = true;
};

} // namespace clip
} // namespace models
} // namespace jinq

#include "openai_clip.inl"

#endif // MORTRED_MODEL_SERVER_OPENAICLIP_H
