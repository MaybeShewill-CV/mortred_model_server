/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: fast_sam_segmentor.h
* Date: 23-9-14
************************************************/

#ifndef MORTRED_MODEL_SERVER_FAST_SAM_SEGMENTOR_H
#define MORTRED_MODEL_SERVER_FAST_SAM_SEGMENTOR_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace segment_anything {
using jinq::common::StatusCode;

/***
 * FastSAM segmentation model: image -> everything mask.
 */
template <typename INPUT, typename OUTPUT>
class FastSamSegmentor : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    FastSamSegmentor();
    ~FastSamSegmentor() override = default;

    FastSamSegmentor(const FastSamSegmentor& transformer) = delete;
    FastSamSegmentor& operator=(const FastSamSegmentor& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    StatusCode on_init(const toml::table& params) override;

    const jinq::models::backend::NamedTensor* find_output(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const std::string& name) const;

    cv::Mat upscale_mask_image(const cv::Mat& mask) const;

    // score thresh
    double _m_conf_thresh = 0.25;
    // nms iou threshold
    double _m_iou_thresh = 0.9;
    // user image size of the current run
    cv::Size _m_input_image_size = cv::Size();
    // network input node size
    cv::Size _m_input_tensor_size = cv::Size();
    // mask proto map size
    cv::Size _m_preds_mask_size = cv::Size();
};

}
}
}

#include "fast_sam_segmentor.inl"

#endif //MORTRED_MODEL_SERVER_FAST_SAM_SEGMENTOR_H
