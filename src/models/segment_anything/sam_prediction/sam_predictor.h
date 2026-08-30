/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_predictor.h
 * Date: 23-5-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_PREDICTOR_H
#define MORTRED_MODEL_SERVER_SAM_PREDICTOR_H

#include <memory>
#include <vector>

#include "toml/toml.hpp"
#include <opencv2/opencv.hpp>

#include "models/backend/backend_cv_model.h"
#include "models/backend/multi_session_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace segment_anything {
using jinq::common::StatusCode;

class SamPromptDecoder;
class SamVitEncoder;

/***
 * SAM prompt segmentation model with independent encoder and decoder sessions.
 */
template <typename INPUT, typename OUTPUT>
class SamPredictor : public jinq::models::backend::MultiSessionModel<SamPredictor<INPUT, OUTPUT>, INPUT, OUTPUT> {
    using Base = jinq::models::backend::MultiSessionModel<SamPredictor<INPUT, OUTPUT>, INPUT, OUTPUT>;
    friend Base;

  public:
    SamPredictor();
    ~SamPredictor() override;

    SamPredictor(const SamPredictor &) = delete;
    SamPredictor &operator=(const SamPredictor &) = delete;

    StatusCode predict(const cv::Mat &input_image, const std::vector<cv::Rect> &bboxes, std::vector<cv::Mat> &predicted_masks);

    StatusCode predict(const cv::Mat &input_image, const std::vector<std::vector<cv::Point2f>> &prompt_points,
                       std::vector<cv::Mat> &predicted_masks);

    StatusCode get_embedding(const cv::Mat &input_image, std::vector<float> &image_embeddings);

  private:
    using SamInput = jinq::models::io_define::segment_anything::sam_prompt_input;
    using SamOutput = jinq::models::io_define::segment_anything::std_sam_prompt_output;

    // The encoder has a fixed input/output pair, so its contract lives here.
    // The decoder exposes optional inputs (orig_im_size) and alternative
    // outputs (masks or low_res_masks), which a fixed IoSpec cannot express,
    // so its spec is left empty and SamPromptDecoder validates its own IO.
    static std::vector<jinq::models::backend::SessionSpec> sessions();

    StatusCode on_init(const toml::table &params) override;

    StatusCode run_sessions(const INPUT &input, OUTPUT &output) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    std::vector<cv::Rect2f> transform_bboxes(const std::vector<cv::Rect> &bboxes, int target_size) const;

    std::vector<std::vector<cv::Point2f>> transform_points(const std::vector<std::vector<cv::Point2f>> &points, int target_size) const;

    std::unique_ptr<SamVitEncoder> _m_encoder;
    std::unique_ptr<SamPromptDecoder> _m_decoder;
    cv::Size _m_ori_image_size;
    cv::Size _m_encoder_input_size;
};

} // namespace segment_anything
} // namespace models
} // namespace jinq

#include "sam_predictor.inl"

#endif // MORTRED_MODEL_SERVER_SAM_PREDICTOR_H
