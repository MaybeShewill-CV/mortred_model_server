/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: nano_detector.h
 * Date: 22-6-10
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_NANO_DETECTOR_H
#define MORTRED_MODEL_SERVER_NANO_DETECTOR_H

#include <map>
#include <string>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/backend/tensor_contract.h"
#include "models/model_io_define.h"
#include "models/object_detection/detection_params.h"
#include "models/object_detection/detector_common.h"

namespace jinq {
namespace models {
namespace object_detection {

using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT> class NanoDetector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  private:
    struct CenterPrior {
        int x;
        int y;
        int stride;
    };

  public:
    NanoDetector();
    ~NanoDetector() override = default;

    NanoDetector(const NanoDetector &transformer) = delete;
    NanoDetector &operator=(const NanoDetector &transformer) = delete;

  protected:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    std::vector<float> refine_bbox_coords(const float *preds, int x, int y, int stride,
                                          const jinq::models::backend::InferenceContext &context,
                                          const DetectionGeometryScale &geometry_scale) const;

    void generate_grid_center_priors();

    static inline float fast_exp(float x);

    static void activation_function_softmax(const float *src, float *dst, int length);

    DetectionParams _m_detection_params;
    // model input node size
    cv::Size _m_input_size_host = cv::Size();
    // center priors
    std::vector<CenterPrior> _m_center_priors;
    // strides
    std::vector<int> _m_strides = {8, 16, 32, 64};
    // reg max origin
    int _m_reg_max = 7;
};

} // namespace object_detection
} // namespace models
} // namespace jinq

#include "nano_detector.inl"

#endif // MORTRED_MODEL_SERVER_NANO_DETECTOR_H
