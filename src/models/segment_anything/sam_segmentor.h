/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamSegmentor.h
 * Date: 23-5-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAMSEGMENTOR_H
#define MORTRED_MODEL_SERVER_SAMSEGMENTOR_H

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
class SamSegmentor {
  public:
    /***
    * constructor
    * @param config
     */
    SamSegmentor();
    
    /***
     *
     */
    ~SamSegmentor();

    /***
    * constructor
    * @param transformer
     */
    SamSegmentor(const SamSegmentor& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    SamSegmentor& operator=(const SamSegmentor& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<cv::Rect>& bboxes,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    jinq::common::StatusCode get_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);


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

#endif // MORTRED_MODEL_SERVER_SAMSEGMENTOR_H
