/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: OpenAiClip.h
 * Date: 23-6-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_OPENAICLIP_H
#define MORTRED_MODEL_SERVER_OPENAICLIP_H

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace clip {

namespace openai_clip_impl {
class Impl;
}

/***
 * OpenAI CLIP 多模态模型：文本/图像 embedding 与图文相似度计算。
 * 统一入口为 run(INPUT, OUTPUT)，按 clip_input.task_type 分发到各子能力。
 */
template <typename INPUT, typename OUTPUT>
class OpenAiClip : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
    * constructor
    * @param config
     */
    OpenAiClip();
    
    /***
     *
     */
    ~OpenAiClip() override;

    /***
    * constructor
    * @param transformer
     */
    OpenAiClip(const OpenAiClip& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    OpenAiClip& operator=(const OpenAiClip& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode run(const INPUT& input, OUTPUT& output) override;

    /***
     *
     * @param input_text
     * @param text_embeddings
     * @return
     */
    jinq::common::StatusCode get_textual_embedding(const std::string& input_text, std::vector<float>& text_embeddings);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    jinq::common::StatusCode get_visual_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);

    /***
     *
     * @param input_texts
     * @param input_image
     * @param simi_scores
     * @return
     */
    jinq::common::StatusCode texts2img(
        const std::vector<std::string>& input_texts, const cv::Mat& input_image, std::vector<float>& simi_scores);

    /***
     *
     * @param input_texts
     * @param input_image
     * @param simi_scores
     * @return
     */
    jinq::common::StatusCode imgs2text(
        const std::vector<cv::Mat>& input_images, const std::string& input_text, std::vector<float>& simi_scores);


    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    std::unique_ptr<openai_clip_impl::Impl> _m_pimpl;
};
}
}
}

#include "openai_clip.inl"

#endif // MORTRED_MODEL_SERVER_OPENAICLIP_H
