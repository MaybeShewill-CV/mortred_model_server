/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_trt_segmentor.cpp
 * Date: 23-9-12
 ************************************************/

#include "sam_trt_segmentor.h"

#include "glog/logging.h"

#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "models/segment_anything/sam_vit_trt_encoder.h"
#include "models/segment_anything/sam_trt_decoder.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace segment_anything {

using jinq::models::segment_anything::SamVitTrtEncoder;
using jinq::models::segment_anything::SamTrtDecoder;

class SamTrtSegmentor::Impl {
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
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input
     * @param output
     * @return
     */
    StatusCode predict(
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
    StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<std::vector<cv::Point2f> >& points,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    StatusCode get_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);

    /***
     *
     * @param image_embeddings
     * @param input_image_size
     * @param prompt_points
     * @param predicted_masks
     * @return
     */
    StatusCode decode_masks(
        const std::vector<float>& image_embeddings,
        const cv::Size& input_image_size,
        const std::vector<std::vector<cv::Point2f> >& prompt_points,
        std::vector<cv::Mat> &predicted_masks);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

  private:
    // model
    std::unique_ptr<SamVitTrtEncoder> _m_sam_encoder;
    std::unique_ptr<SamTrtDecoder> _m_sam_decoder;

    // origin image size
    cv::Size _m_ori_image_size;

    // sam vit input image size
    cv::Size _m_sam_encoder_input_size;

    // init flag
    bool _m_successfully_init_model = false;

  private:
    /***
     *
     * @param bboxes
     * @return
     */
    cv::Point2f transform_coords(const cv::Point2f& ori_point, int target_size=1024) const;
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode SamTrtSegmentor::Impl::init(const decltype(toml::parse("")) &cfg) {
    // init sam encoder
    _m_sam_encoder = std::make_unique<SamVitTrtEncoder>();
    _m_sam_encoder->init(cfg);
    if (!_m_sam_encoder->is_successfully_initialized()) {
        LOG(ERROR) << "init sam vit trt encoder failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_sam_encoder_input_size.height = _m_sam_encoder->get_encoder_input_shape()[2];
    _m_sam_encoder_input_size.width = _m_sam_encoder->get_encoder_input_shape()[3];

    // init sam vit decoder
    _m_sam_decoder = std::make_unique<SamTrtDecoder>();
    _m_sam_decoder->init(cfg);
    if (!_m_sam_decoder->is_successfully_initialized()) {
        LOG(ERROR) << "init sam decoder failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_sam_decoder->set_encoder_input_size(_m_sam_encoder_input_size);

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load sam trt model";
    return StatusCode::OJBK;
}

/***
 *
 * @param input_image
 * @param bboxes
 * @param points
 * @param point_labels
 * @param predicted_mask
 * @return
 */
StatusCode SamTrtSegmentor::Impl::predict(
    const cv::Mat& input_image,
    const std::vector<cv::Rect>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    // encode image embeddings
    if (!input_image.data || input_image.empty()) {
        LOG(ERROR) << "invalid / empty input image";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_ori_image_size = input_image.size();
    std::vector<float> image_embeddings;
    auto status = _m_sam_encoder->encode(input_image, image_embeddings);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "sam encode image embeddings failed";
        return status;
    }

    // transform bboxes
    std::vector<cv::Rect2f> transformed_bboxes;
    for (auto& bbox : bboxes) {
        cv::Rect2f transformed_bbox;
        auto transformed_tl = transform_coords(bbox.tl(), _m_sam_encoder_input_size.height);
        auto transformed_br = transform_coords(bbox.br(), _m_sam_encoder_input_size.height);
        transformed_bbox.x = transformed_tl.x;
        transformed_bbox.y = transformed_tl.y;
        transformed_bbox.width = transformed_br.x - transformed_tl.x;
        transformed_bbox.height = transformed_br.y - transformed_tl.y;
        transformed_bboxes.push_back(transformed_bbox);
    }

    // decode masks
    _m_sam_decoder->set_ori_image_size(_m_ori_image_size);
    status = _m_sam_decoder->decode(image_embeddings, transformed_bboxes, predicted_masks);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "sam decode masks failed";
        return status;
    }
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @param points
 * @param predicted_masks
 * @return
 */
StatusCode SamTrtSegmentor::Impl::predict(
    const cv::Mat &input_image,
    const std::vector<std::vector<cv::Point2f>> &points,
    std::vector<cv::Mat> &predicted_masks) {
    // encode image embeddings
    if (!input_image.data || input_image.empty()) {
        LOG(ERROR) << "invalid / empty input image";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_ori_image_size = input_image.size();
    std::vector<float> image_embeddings;
    auto status = _m_sam_encoder->encode(input_image, image_embeddings);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "sam encode image embeddings failed";
        return status;
    }

    // transform bboxes
    std::vector<std::vector<cv::Point2f> > transformed_points;
    for (auto& pts : points) {
        std::vector<cv::Point2f> trans_pts;
        for (auto& pt: pts) {
            cv::Point2f transformed_point = transform_coords(pt, _m_sam_encoder_input_size.height);
            trans_pts.push_back(transformed_point);
        }
        transformed_points.push_back(trans_pts);
    }

    // decode masks
    _m_sam_decoder->set_ori_image_size(_m_ori_image_size);
    status = _m_sam_decoder->decode(image_embeddings, transformed_points, predicted_masks);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "sam decode masks failed";
        return status;
    }
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode SamTrtSegmentor::Impl::get_embedding(
    const cv::Mat &input_image,
    std::vector<float> &image_embeddings) {
    return _m_sam_encoder->encode(input_image, image_embeddings);
}

/***
 *
 * @param image_embeddings
 * @param input_image_size
 * @param prompt_points
 * @param predicted_masks
 * @return
 */
StatusCode SamTrtSegmentor::Impl::decode_masks(
    const std::vector<float> &image_embeddings,
    const cv::Size& input_image_size,
    const std::vector<std::vector<cv::Point2f>> &prompt_points,
    std::vector<cv::Mat> &predicted_masks) {
    _m_sam_decoder->set_ori_image_size(input_image_size);
   return _m_sam_decoder->decode(image_embeddings, prompt_points, predicted_masks);
}

/***
 *
 * @param ori_point
 * @param target_size
 * @return
 */
cv::Point2f SamTrtSegmentor::Impl::transform_coords(const cv::Point2f &ori_point, int target_size) const {
    auto ori_img_h = static_cast<float>(_m_ori_image_size.height);
    auto ori_img_w = static_cast<float>(_m_ori_image_size.width);
    auto long_side = std::max(ori_img_h, ori_img_w);

    float scale = static_cast<float>(target_size) / long_side;
    cv::Point2f out_point;
    out_point.x = ori_point.x * scale;
    out_point.y = ori_point.y * scale;
    return out_point;
}

/***
 *
 */
SamTrtSegmentor::SamTrtSegmentor() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
SamTrtSegmentor::~SamTrtSegmentor() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode SamTrtSegmentor::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_image
 * @param bboxes
 * @param points
 * @param point_labels
 * @param predicted_mask
 * @return
 */
StatusCode SamTrtSegmentor::predict(
    const cv::Mat& input_image,
    const std::vector<cv::Rect>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    return _m_pimpl->predict(input_image, bboxes, predicted_masks);
}

/***
 *
 * @param input_image
 * @param points
 * @param predicted_masks
 * @return
 */
StatusCode SamTrtSegmentor::predict(
    const cv::Mat &input_image,
    const std::vector<std::vector<cv::Point2f>> &points,
    std::vector<cv::Mat> &predicted_masks) {
    return _m_pimpl->predict(input_image, points, predicted_masks);
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode SamTrtSegmentor::get_embedding(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    return _m_pimpl->get_embedding(input_image, image_embeddings);
}

/***
 *
 * @param image_embeddings
 * @param input_image_size
 * @param prompt_points
 * @param predicted_masks
 * @return
 */
StatusCode SamTrtSegmentor::decode_masks(
    const std::vector<float> &image_embeddings,
    const cv::Size& input_image_size,
    const std::vector<std::vector<cv::Point2f>> &prompt_points,
    std::vector<cv::Mat> &predicted_masks) {
    return _m_pimpl->decode_masks(image_embeddings, input_image_size, prompt_points, predicted_masks);
}

/***
 *
 * @return
 */
bool SamTrtSegmentor::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}