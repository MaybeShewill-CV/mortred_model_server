/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: OpenAiClipTextEncoder.h
 * Date: 23-6-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_OPENAICLIPTEXTENCODER_H
#define MORTRED_MODEL_SERVER_OPENAICLIPTEXTENCODER_H

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
class OpenAiClipTextEncoder {
  public:
    /***
    * constructor
    * @param config
     */
    OpenAiClipTextEncoder();

    /***
     *
     */
    ~OpenAiClipTextEncoder();

    /***
    * constructor
    * @param transformer
     */
    OpenAiClipTextEncoder(const OpenAiClipTextEncoder& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    OpenAiClipTextEncoder& operator=(const OpenAiClipTextEncoder& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_image
     * @param text_embeddings
     * @return
     */
    jinq::common::StatusCode encode(const cv::Mat& input_image, std::vector<float>& text_embeddings);

    /***
     *
     * @return
     */
    std::vector<int> get_encoder_input_shape() const;

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

#endif // MORTRED_MODEL_SERVER_OPENAICLIPTEXTENCODER_H
