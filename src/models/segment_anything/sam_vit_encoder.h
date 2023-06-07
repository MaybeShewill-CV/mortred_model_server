/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamVitEncoder.h
 * Date: 23-6-7
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAMVITENCODER_H
#define MORTRED_MODEL_SERVER_SAMVITENCODER_H

#include <vector>

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace segment_anything {

/***
 *
 */
class SamVitEncoder {
  public:
    /***
    * constructor
    * @param config
     */
    SamVitEncoder();

    /***
     *
     */
    ~SamVitEncoder();

    /***
    * constructor
    * @param transformer
     */
    SamVitEncoder(const SamVitEncoder& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    SamVitEncoder& operator=(const SamVitEncoder& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    jinq::common::StatusCode get_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);


    /***
     * if db text detector successfully initialized
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

#endif // MORTRED_MODEL_SERVER_SAMVITENCODER_H
