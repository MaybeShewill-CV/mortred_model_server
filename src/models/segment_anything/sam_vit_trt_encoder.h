/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_vit_trt_encoder.h
 * Date: 23-9-7
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_VIT_TRT_ENCODER_H
#define MORTRED_MODEL_SERVER_SAM_VIT_TRT_ENCODER_H

#include <vector>

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace segment_anything {

class SamVitTrtEncoder {
  public:
    /***
    * constructor
    * @param config
     */
    SamVitTrtEncoder();
    
    /***
     *
     */
    ~SamVitTrtEncoder();

    /***
    * constructor
    * @param transformer
     */
    SamVitTrtEncoder(const SamVitTrtEncoder& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    SamVitTrtEncoder& operator=(const SamVitTrtEncoder& transformer) = delete;

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
    jinq::common::StatusCode encode(const cv::Mat& input_image, std::vector<float>& image_embeddings);

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

#endif // MORTRED_MODEL_SERVER_SAM_VIT_TRT_ENCODER_H
