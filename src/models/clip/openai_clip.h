/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: openai_clip.h
 * Date: 23-6-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_OPENAICLIP_H
#define MORTRED_MODEL_SERVER_OPENAICLIP_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "toml/toml.hpp"
#include <opencv2/opencv.hpp>

#include "models/backend/backend_cv_model.h"
#include "models/backend/multi_session_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/clip/simple_tokenizer.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace clip {
using mortred::common::StatusCode;

/***
 * OpenAI CLIP multi-engine model. The visual and text encoders are unified
 * inference sessions, while the BPE tokenizer is model-local preprocessing.
 */
template <typename INPUT, typename OUTPUT>
class OpenAiClip : public mortred::models::backend::MultiSessionModel<OpenAiClip<INPUT, OUTPUT>, INPUT, OUTPUT> {
    using Base = mortred::models::backend::MultiSessionModel<OpenAiClip<INPUT, OUTPUT>, INPUT, OUTPUT>;
    friend Base;

  public:
    OpenAiClip();
    ~OpenAiClip() override = default;

    OpenAiClip(const OpenAiClip &transformer) = delete;
    OpenAiClip &operator=(const OpenAiClip &transformer) = delete;

    StatusCode get_textual_embedding(const std::string &input_text, std::vector<float> &text_embeddings);

    StatusCode get_visual_embedding(const cv::Mat &input_image, std::vector<float> &image_embeddings);

    StatusCode texts2img(const std::vector<std::string> &input_texts, const cv::Mat &input_image, std::vector<float> &simi_scores);

    StatusCode imgs2text(const std::vector<cv::Mat> &input_images, const std::string &input_text, std::vector<float> &simi_scores);

  protected:
    static StatusCode embedding_shape_status(const std::vector<float> &lhs, const std::vector<float> &rhs);

    StatusCode classify_embedding_output(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                                         std::vector<float> &embedding) const;

  private:
    using ClipInput = mortred::models::io_define::clip::clip_input;
    using ClipOutput = mortred::models::io_define::clip::clip_output;
    using ClipTaskType = mortred::models::io_define::clip::ClipTaskType;

    // the two engines this model needs; MultiSessionModel creates, validates
    // and owns them, this class only drives them
    static std::vector<mortred::models::backend::SessionSpec> sessions();

    StatusCode on_init(const toml::table &params) override;

    StatusCode run_sessions(const INPUT &input, OUTPUT &output) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode encode_text(const std::string &input_text, std::vector<float> &text_embeddings) const;

    StatusCode encode_image(const cv::Mat &input_image, std::vector<float> &image_embeddings) const;

    void tokenize(const std::string &input_text, std::vector<int32_t> &token_ids) const;

    static const mortred::models::backend::TensorInfo *find_info(const mortred::models::backend::InferenceSession &session,
                                                              const std::string &name);

    static bool normalize_embedding(std::vector<float> &embedding);

    static bool embeddings_compatible(const std::vector<float> &lhs, const std::vector<float> &rhs);

    SimpleTokenizer _m_tokenizer;
    int _m_context_length = 77;
    bool _m_truncate_context = true;
};

} // namespace clip
} // namespace models
} // namespace mortred

#include "openai_clip.inl"

#endif // MORTRED_MODEL_SERVER_OPENAICLIP_H
