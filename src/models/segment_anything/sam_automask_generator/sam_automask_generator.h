/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_automask_generator.h
 * Date: 23-10-13
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_AUTOMASK_GENERATOR_H
#define MORTRED_MODEL_SERVER_SAM_AUTOMASK_GENERATOR_H

#include <memory>
#include <vector>

#include "toml/toml.hpp"
#include <opencv2/opencv.hpp>

#include "models/backend/backend_cv_model.h"
#include "models/backend/multi_session_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace segment_anything {
using mortred::common::StatusCode;

class SamAmgDecoder;
class SamVitEncoder;

/***
 * SAM automatic mask generator with independent encoder and AMG decoder
 * sessions.
 */
template <typename INPUT, typename OUTPUT> class SamAutoMaskGenerator : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    SamAutoMaskGenerator();
    ~SamAutoMaskGenerator() override;

    SamAutoMaskGenerator(const SamAutoMaskGenerator &) = delete;
    SamAutoMaskGenerator &operator=(const SamAutoMaskGenerator &) = delete;

    StatusCode generate(const cv::Mat &input_image, mortred::models::io_define::segment_anything::sam_amg_output &amg_output,
                        const mortred::models::backend::ParamSet *request_params = nullptr);

  protected:
    StatusCode on_init(const toml::table &params) override;

    StatusCode run_sessions(const INPUT &input, OUTPUT &output) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    // the encoder session is owned here: SamVitEncoder only drives it
    std::unique_ptr<mortred::models::backend::InferenceSession> _m_encoder_session;
    std::unique_ptr<SamVitEncoder> _m_encoder;
    std::unique_ptr<SamAmgDecoder> _m_decoder;
    cv::Size _m_encoder_input_size;
    int _m_points_per_side = 32;
    float _m_pred_iou_thresh = 0.88f;
    float _m_stability_score_thresh = 0.95f;
    float _m_box_nms_thresh = 0.7f;
    int _m_min_mask_region_area = 0;
};

} // namespace segment_anything
} // namespace models
} // namespace mortred

#include "sam_automask_generator.inl"

#endif // MORTRED_MODEL_SERVER_SAM_AUTOMASK_GENERATOR_H
