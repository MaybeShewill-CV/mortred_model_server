/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: FastFastSamSegmentor.h
 * Date: 23-6-29
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_FASTFastSamSegmentor_H
#define MORTRED_MODEL_SERVER_FASTFastSamSegmentor_H

#include <memory>
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
class FastSamSegmentor {
  public:
    /***
    * constructor
    * @param config
     */
    FastSamSegmentor();
    
    /***
     *
     */
    ~FastSamSegmentor();

    /***
    * constructor
    * @param transformer
     */
    FastSamSegmentor(const FastSamSegmentor& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    FastSamSegmentor& operator=(const FastSamSegmentor& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_image
     * @param everything_mask
     * @return
     */
    jinq::common::StatusCode everything(const cv::Mat& input_image, cv::Mat& everything_mask);

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

#endif // MORTRED_MODEL_SERVER_FASTFastSamSegmentor_H
