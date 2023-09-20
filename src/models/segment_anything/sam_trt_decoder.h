/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_trt_decoder.h
 * Date: 23-9-8
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_TRT_DECODER_H
#define MORTRED_MODEL_SERVER_SAM_TRT_DECODER_H

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
class SamTrtDecoder {
  public:
    /***
    * constructor
    * @param config
     */
    SamTrtDecoder();

    /***
     *
     */
    ~SamTrtDecoder();

    /***
    * constructor
    * @param transformer
     */
    SamTrtDecoder(const SamTrtDecoder& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    SamTrtDecoder& operator=(const SamTrtDecoder& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param ori_img_size
     */
    void set_ori_image_size(const cv::Size& ori_img_size);

    /***
     *
     * @param ori_img_size
     */
    void set_encoder_input_size(const cv::Size& input_node_size);

    /***
     *
     * @param image_embeddings
     * @param bboxes
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<cv::Rect2f>& bboxes,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param image_embeddings
     * @param bboxes
     * @param predicted_masks
     * @return
     */
    jinq::common::StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<std::vector<cv::Point2f> >& points,
        std::vector<cv::Mat>& predicted_masks);

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

#endif // MORTRED_MODEL_SERVER_SAM_TRT_DECODER_H
