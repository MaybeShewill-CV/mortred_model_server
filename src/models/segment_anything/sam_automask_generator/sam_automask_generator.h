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

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace segment_anything {

class SamAmgDecoder;
class SamVitEncoder;

/***
 * SAM automatic mask generator with independent encoder and AMG decoder
 * sessions.
 */
template<typename INPUT, typename OUTPUT>
class SamAutoMaskGenerator : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    SamAutoMaskGenerator();
    ~SamAutoMaskGenerator() override;

    SamAutoMaskGenerator(const SamAutoMaskGenerator&) = delete;
    SamAutoMaskGenerator& operator=(const SamAutoMaskGenerator&) = delete;

    jinq::common::StatusCode generate(
        const cv::Mat& input_image,
        jinq::models::io_define::segment_anything::sam_amg_output& amg_output);

  private:
    jinq::common::StatusCode on_init(const toml::table& params) override;

    jinq::common::StatusCode run_sessions(const INPUT& input, OUTPUT& output) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

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
} // namespace jinq

#include "sam_automask_generator.inl"

#endif // MORTRED_MODEL_SERVER_SAM_AUTOMASK_GENERATOR_H
