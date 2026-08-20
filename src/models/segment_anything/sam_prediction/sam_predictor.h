/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamPredictor.h
 * Date: 23-5-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_PREDICTOR_H
#define MORTRED_MODEL_SERVER_SAM_PREDICTOR_H

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

namespace sam_predictor_impl {
class Impl;
}

/***
 * SAM 提示分割模型：图像 + bbox/点提示 -> 分割 mask。
 * 统一入口 run(INPUT, OUTPUT)，按 bboxes/prompt_points 非空自动分发。
 */
template <typename INPUT, typename OUTPUT>
class SamPredictor : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
    * constructor
    * @param config
     */
    SamPredictor();
    
    /***
     *
     */
    ~SamPredictor() override;

    /***
    * constructor
    * @param transformer
     */
    SamPredictor(const SamPredictor& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    SamPredictor& operator=(const SamPredictor& transformer) = delete;

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
    jinq::common::StatusCode run_impl(const INPUT& input, OUTPUT& output) override;

    /***
     *
     * @param input_image
     * @param bboxes
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<cv::Rect>& bboxes,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param input_image
     * @param prompt_points
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<std::vector<cv::Point2f> >& prompt_points,
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
    bool is_successfully_initialized() const override;

  private:
    std::unique_ptr<sam_predictor_impl::Impl> _m_pimpl;
};
}
}
}

#include "sam_predictor.inl"

#endif // MORTRED_MODEL_SERVER_SAM_PREDICTOR_H
