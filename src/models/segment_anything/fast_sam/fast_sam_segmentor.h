/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: fast_sam_segmentor.h
 * Date: 23-9-14
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_FAST_SAM_SEGMENTOR_H
#define MORTRED_MODEL_SERVER_FAST_SAM_SEGMENTOR_H

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace segment_anything {

namespace fast_sam_segmentor_impl {
class Impl;
}

/***
 * FastSAM 分割模型：图像 -> everything mask。
 */
template <typename INPUT, typename OUTPUT>
class FastSamSegmentor : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
    * constructor
    * @param config
     */
    FastSamSegmentor();

    /***
     *
     */
    ~FastSamSegmentor() override;

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
     * @param input_image
     * @param everything_mask
     * @return
     */
    jinq::common::StatusCode everything(const cv::Mat& input_image, cv::Mat& everything_mask);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    std::unique_ptr<fast_sam_segmentor_impl::Impl> _m_pimpl;
};
}
}
}

#include "fast_sam_segmentor.inl"

#endif // MORTRED_MODEL_SERVER_FAST_SAM_SEGMENTOR_H
