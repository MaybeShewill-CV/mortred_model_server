/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_predictor.inl
 * Date: 23-5-26
 ************************************************/

#include <algorithm>
#include <type_traits>

#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "sam_prompt_decoder.h"
#include "sam_vit_encoder.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace segment_anything {

using jinq::models::segment_anything::SamVitEncoder;
using jinq::models::segment_anything::SamPromptDecoder;

namespace sam_predictor_impl {

using internal_input = jinq::models::io_define::segment_anything::sam_prompt_input;
using internal_output = jinq::models::io_define::segment_anything::std_sam_prompt_output;

/***
 * 将用户自定义输入转换为模型内部输入
 */
template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<internal_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    return in;
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::mat_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    internal_input result{};
    result.image = in.input_image;
    return result;
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::file_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    internal_input result{};
    if (!FilePathUtil::is_file_exist(in.input_image_path)) {
        DLOG(WARNING) << "input image: " << in.input_image_path << " not exist";
        return result;
    }
    result.image = cv::imread(in.input_image_path, cv::IMREAD_UNCHANGED);
    return result;
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::base64_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    internal_input result{};
    result.image = CvUtils::decode_base64_str_into_cvmat(in.input_image_content);
    return result;
}

/***
 * 将模型内部输出转换为用户自定义输出
 */
template <typename OUTPUT>
typename std::enable_if<
    std::is_same<OUTPUT, std::decay<internal_output>::type>::value, internal_output>::type
transform_output(const internal_output& internal_out) {
    return internal_out;
}

/***
 * SAM 提示分割模型内部实现
 */
class Impl {
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
    StatusCode init(const toml::table& cfg);

    /***
     *
     * @param input_image
     * @param bboxes
     * @param predicted_masks
     * @return
     */
    StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<cv::Rect>& bboxes,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param input_image
     * @param prompt_points
     * @param predicted_masks
     * @return
     */
    StatusCode predict(
        const cv::Mat& input_image,
        const std::vector<std::vector<cv::Point2f> >& prompt_points,
        std::vector<cv::Mat>& predicted_masks);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    StatusCode get_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);

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
    std::unique_ptr<SamPromptDecoder> _m_sam_decoder;

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
    std::vector<cv::Rect2f> transform_bboxes(const std::vector<cv::Rect>& bboxes, int target_size=1024) const;

    /***
     *
     * @param bboxes
     * @return
     */
    std::vector<std::vector<cv::Point2f> > transform_points(
        const std::vector<std::vector<cv::Point2f> >& points, int target_size=1024) const;
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode Impl::init(const toml::table &cfg) {
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

    // init sam prompt decoder
    _m_sam_decoder = std::make_unique<SamPromptDecoder>();
    _m_sam_decoder->init(cfg);
    if (!_m_sam_decoder->is_successfully_initialized()) {
        LOG(ERROR) << "init sam prompt decoder failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_sam_decoder->set_encoder_input_size(_m_sam_encoder_input_size);

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load sam model";
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @param bboxes
 * @param predicted_mask
 * @return
 */
StatusCode Impl::predict(
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
    std::vector<cv::Rect2f> transformed_bboxes = transform_bboxes(bboxes, _m_sam_encoder_input_size.height);

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
 * @param prompt_points
 * @param predicted_masks
 * @return
 */
StatusCode Impl::predict(
    const cv::Mat &input_image,
    const std::vector<std::vector<cv::Point2f>> &prompt_points,
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
    auto transformed_points = transform_points(prompt_points, _m_sam_encoder_input_size.height);

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
StatusCode Impl::get_embedding(
    const cv::Mat &input_image,
    std::vector<float> &image_embeddings) {
    return _m_sam_encoder->encode(input_image, image_embeddings);
}

/***
 *
 * @param bboxes
 * @param target_size
 * @return
 */
std::vector<cv::Rect2f> Impl::transform_bboxes(const std::vector<cv::Rect> &bboxes, int target_size) const {
    auto ori_img_h = static_cast<float>(_m_ori_image_size.height);
    auto ori_img_w = static_cast<float>(_m_ori_image_size.width);
    auto long_side = std::max(ori_img_h, ori_img_w);

    float scale = static_cast<float>(target_size) / long_side;

    std::vector<cv::Rect2f> transformed_bboxes;
    for (auto& box : bboxes) {
        cv::Rect2f new_box = box;
        new_box.x *= scale;
        new_box.y *= scale;
        new_box.width *= scale;
        new_box.height *= scale;
        transformed_bboxes.push_back(new_box);
    }

    return transformed_bboxes;
}

/***
 *
 * @param points
 * @param target_size
 * @return
 */
std::vector<std::vector<cv::Point2f>> Impl::transform_points(
    const std::vector<std::vector<cv::Point2f>> &points, int target_size) const {
    auto ori_img_h = static_cast<float>(_m_ori_image_size.height);
    auto ori_img_w = static_cast<float>(_m_ori_image_size.width);
    auto long_side = std::max(ori_img_h, ori_img_w);
    float scale = static_cast<float>(target_size) / long_side;

    std::vector<std::vector<cv::Point2f> > transformed_points;
    for (auto& pts_per_obj : points) {
        std::vector<cv::Point2f> trans_pts;
        for (auto& pt : pts_per_obj) {
            cv::Point2f trans_pt;
            trans_pt.x = pt.x * scale;
            trans_pt.y = pt.y * scale;
            trans_pts.push_back(trans_pt);
        }
        transformed_points.push_back(trans_pts);
    }

    return transformed_points;
}

} // namespace sam_predictor_impl

/************ Template Implementation ************/

template <typename INPUT, typename OUTPUT>
SamPredictor<INPUT, OUTPUT>::SamPredictor() {
    _m_pimpl = std::make_unique<sam_predictor_impl::Impl>();
}

template <typename INPUT, typename OUTPUT>
SamPredictor<INPUT, OUTPUT>::~SamPredictor() = default;

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode SamPredictor<INPUT, OUTPUT>::init(const toml::table &cfg) {
    return _m_pimpl->init(cfg);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode SamPredictor<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    auto internal_input = sam_predictor_impl::transform_input(input);
    sam_predictor_impl::internal_output internal_output;
    StatusCode status = StatusCode::OK;

    if (!internal_input.bboxes.empty()) {
        status = _m_pimpl->predict(internal_input.image, internal_input.bboxes, internal_output);
    } else {
        status = _m_pimpl->predict(internal_input.image, internal_input.prompt_points, internal_output);
    }

    output = sam_predictor_impl::transform_output<OUTPUT>(internal_output);
    return status;
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode SamPredictor<INPUT, OUTPUT>::predict(
    const cv::Mat& input_image,
    const std::vector<cv::Rect>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    return _m_pimpl->predict(input_image, bboxes, predicted_masks);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode SamPredictor<INPUT, OUTPUT>::predict(
    const cv::Mat &input_image,
    const std::vector<std::vector<cv::Point2f>> &prompt_points,
    std::vector<cv::Mat> &predicted_masks) {
    return _m_pimpl->predict(input_image, prompt_points, predicted_masks);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode SamPredictor<INPUT, OUTPUT>::get_embedding(
    const cv::Mat &input_image, std::vector<float> &image_embeddings) {
   return _m_pimpl->get_embedding(input_image, image_embeddings);
}

template <typename INPUT, typename OUTPUT>
bool SamPredictor<INPUT, OUTPUT>::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

} // namespace segment_anything
} // namespace models
} // namespace jinq
