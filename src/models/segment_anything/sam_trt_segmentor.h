/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_trt_segmentor.h
 * Date: 23-9-12
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_TRT_SEGMENTOR_H
#define MORTRED_MODEL_SERVER_SAM_TRT_SEGMENTOR_H

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
class SamTrtSegmentor {
  public:
    /***
    * constructor
    * @param config
     */
    SamTrtSegmentor();
    
    /***
     *
     */
    ~SamTrtSegmentor();

    /***
    * constructor
    * @param transformer
     */
    SamTrtSegmentor(const SamTrtSegmentor& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    SamTrtSegmentor& operator=(const SamTrtSegmentor& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<cv::Rect>& bboxes,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param input_image
     * @param points
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<std::vector<cv::Point2f> >& points,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param input_image
     * @param points
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode parallel_predict(
        const cv::Mat& input_image,
        const std::vector<std::vector<cv::Point2f> >& points,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    jinq::common::StatusCode get_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);

    /**
     *
     * @param image_embeddings
     * @param input_image_size
     * @param prompt_points
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode decode_masks(
        const std::vector<float>& image_embeddings,
        const cv::Size& input_image_size,
        const std::vector<std::vector<cv::Point2f> >& prompt_points,
        std::vector<cv::Mat> &predicted_masks);

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

#endif // MORTRED_MODEL_SERVER_SAM_TRT_SEGMENTOR_H
