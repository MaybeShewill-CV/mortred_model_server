/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamAutoMaskGenerator.h
 * Date: 23-10-13
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_AUTOMASK_GENERATOR_H
#define MORTRED_MODEL_SERVER_SAM_AUTOMASK_GENERATOR_H

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

using AmgMaskOutput = jinq::models::io_define::segment_anything::sam_amg_output;

namespace sam_automask_generator_impl {
class Impl;
}

/***
 * SAM 自动 mask 生成模型：图像 -> 密集点网格自动生成的 mask 集合。
 */
template <typename INPUT, typename OUTPUT>
class SamAutoMaskGenerator : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
    * constructor
    * @param config
     */
    SamAutoMaskGenerator();
    
    /***
     *
     */
    ~SamAutoMaskGenerator() override;

    /***
    * constructor
    * @param transformer
     */
    SamAutoMaskGenerator(const SamAutoMaskGenerator& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    SamAutoMaskGenerator& operator=(const SamAutoMaskGenerator& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const toml::table& cfg) override;

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
     * @param bboxes
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode generate(const cv::Mat& input_image, AmgMaskOutput& amg_output);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    std::unique_ptr<sam_automask_generator_impl::Impl> _m_pimpl;
};
}
}
}

#include "sam_automask_generator.inl"

#endif // MORTRED_MODEL_SERVER_SAM_AUTOMASK_GENERATOR_H
