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

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace segment_anything {

struct AmgMaskOutput {
    std::vector<cv::Mat> segmentations;
    std::vector<int32_t> areas;
    std::vector<cv::Rect> bboxes;
    std::vector<float> preds_ious;
    std::vector<float> preds_stability_scores;
    std::vector<cv::Point2f> point_coords;
};

/***
 * 
 */
class SamAutoMaskGenerator {
  public:
    /***
    * constructor
    * @param config
     */
    SamAutoMaskGenerator();
    
    /***
     *
     */
    ~SamAutoMaskGenerator();

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
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

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
    bool is_successfully_initialized() const;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};
}
}
}

#endif // MORTRED_MODEL_SERVER_SAM_AUTOMASK_GENERATOR_H
