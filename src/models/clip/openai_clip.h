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

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace clip {

/***
 * 
 */
class OpenAiClip {
  public:
    /***
    * constructor
    * @param config
     */
    OpenAiClip();
    
    /***
     *
     */
    ~OpenAiClip();

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
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

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
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};
}
}
}

#endif // MORTRED_MODEL_SERVER_OPENAICLIP_H
