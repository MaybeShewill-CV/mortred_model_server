/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SamAutoMaskGenerator.cpp
 * Date: 23-10-13
 ************************************************/

#include "sam_automask_generator.h"

#include "glog/logging.h"

#include "sam_amg_decoder.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/segment_anything/sam_prediction/sam_vit_encoder.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace segment_anything {

using jinq::models::segment_anything::SamVitEncoder;
using jinq::models::segment_anything::SamAmgDecoder;

class SamAutoMaskGenerator::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_image
     * @param amg_output
     * @return
     */
    StatusCode generate(const cv::Mat& input_image, AmgMaskOutput& amg_output);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

  private:
    // model
    std::unique_ptr<SamVitEncoder> _m_sam_encoder;
    std::unique_ptr<SamAmgDecoder> _m_sam_decoder;

    // sam vit input image size
    cv::Size _m_sam_encoder_input_size;

    // mask decode params
    int _m_points_per_side = 32;
    float _m_pred_iou_thresh = 0.88f;
    float _m_stability_score_thresh = 0.95f;
    float _m_box_nms_thresh = 0.7f;
    int _m_min_mask_region_area = 0;

    // init flag
    bool _m_successfully_init_model = false;
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamAutoMaskGenerator::Impl::init(const decltype(toml::parse("")) &cfg) {
    // init sam encoder
    _m_sam_encoder = std::make_unique<SamVitEncoder>();
    _m_sam_encoder->init(cfg);
    if (!_m_sam_encoder->is_successfully_initialized()) {
        LOG(ERROR) << "init sam vit encoder failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_sam_encoder_input_size.height = _m_sam_encoder->get_encoder_input_shape()[2];
    _m_sam_encoder_input_size.width = _m_sam_encoder->get_encoder_input_shape()[3];

    // init sam vit decoder
    _m_sam_decoder = std::make_unique<SamAmgDecoder>();
    _m_sam_decoder->init(cfg);
    if (!_m_sam_decoder->is_successfully_initialized()) {
        LOG(ERROR) << "init sam amg decoder failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_sam_decoder->set_encoder_input_size(_m_sam_encoder_input_size);

    // init decode params
    auto decoder_cfg = cfg.at("SAM_AMG_DECODER");
    _m_points_per_side = static_cast<int>(decoder_cfg.at("points_per_size").as_integer());
    _m_pred_iou_thresh = static_cast<float>(decoder_cfg.at("pred_iou_thresh").as_floating());
    _m_stability_score_thresh = static_cast<float>(decoder_cfg.at("stability_score_thresh").as_floating());
    _m_box_nms_thresh = static_cast<float>(decoder_cfg.at("box_nms_thresh").as_floating());
    _m_min_mask_region_area = static_cast<int>(decoder_cfg.at("min_mask_region_area").as_integer());

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load sam auto mask generator model";
    return StatusCode::OJBK;
}

/***
 *
 * @param input_image
 * @param amg_output
 * @return
 */
StatusCode SamAutoMaskGenerator::Impl::generate(const cv::Mat &input_image, AmgMaskOutput &amg_output) {
    // encode input image
    std::vector<float> img_embeds;
    auto status = _m_sam_encoder->encode(input_image, img_embeds);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "sam encode input image failed status: " << status;
        return status;
    }

    // decode mask from auto-generated prompt points
    _m_sam_decoder->set_ori_image_size(input_image.size());
    status = _m_sam_decoder->decode_everything(
        img_embeds, amg_output, _m_points_per_side, _m_pred_iou_thresh,
        _m_stability_score_thresh,_m_box_nms_thresh, _m_min_mask_region_area);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "sam decode mask from auto-generated prompt points failed status: " << status;
        return status;
    }

    return StatusCode::OK;
}

/***
 *
 */
SamAutoMaskGenerator::SamAutoMaskGenerator() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
SamAutoMaskGenerator::~SamAutoMaskGenerator() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamAutoMaskGenerator::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_image
 * @param amg_output
 * @return
 */
StatusCode SamAutoMaskGenerator::generate(const cv::Mat &input_image, AmgMaskOutput &amg_output) {
    return _m_pimpl->generate(input_image, amg_output);
}

/***
 *
 * @return
 */
bool SamAutoMaskGenerator::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}
