/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: sam_amg_decoder.h
* Date: 23-9-20
************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_AMG_DECODER_H
#define MORTRED_MODEL_SERVER_SAM_AMG_DECODER_H

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace segment_anything {
using jinq::common::StatusCode;

using AmgMaskOutput = jinq::models::io_define::segment_anything::sam_amg_output;

/***
 *
 */
class SamAmgDecoder {
  public:
    /***
    * constructor
    * @param config
     */
    SamAmgDecoder();

    /***
     *
     */
    ~SamAmgDecoder();

    /***
    * constructor
    * @param transformer
     */
    SamAmgDecoder(const SamAmgDecoder& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    SamAmgDecoder& operator=(const SamAmgDecoder& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    StatusCode init(const toml::table& cfg);

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
     * @param output_mask
     * @param points_per_side
     * @param pred_iou_thresh
     * @param stability_score_thresh
     * @param stability_score_offset
     * @param box_nms_thresh
     * @param min_mask_region_area
     * @return
     */
    StatusCode decode_everything(
        const std::vector<float>& image_embeddings,
        AmgMaskOutput& output, int points_per_side = 32, float pred_iou_thresh = 0.88,
        float stability_score_thresh = 0.95, float box_nms_thresh = 0.7, int min_mask_region_area = 0);

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

#endif // MORTRED_MODEL_SERVER_SAM_AMG_DECODER_H
