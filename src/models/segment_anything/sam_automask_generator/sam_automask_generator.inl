/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_automask_generator.inl
 * Date: 23-10-13
 ************************************************/

#include <type_traits>

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

namespace sam_automask_generator_impl {

using internal_input = cv::Mat;
using internal_output = jinq::models::io_define::segment_anything::sam_amg_output;

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
    return in.input_image;
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::file_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    if (!FilePathUtil::is_file_exist(in.input_image_path)) {
        DLOG(WARNING) << "input image: " << in.input_image_path << " not exist";
        return internal_input{};
    }
    return cv::imread(in.input_image_path, cv::IMREAD_UNCHANGED);
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::base64_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    return CvUtils::decode_base64_str_into_cvmat(in.input_image_content);
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
 * SAM 自动 mask 生成模型内部实现
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

    // init sam auto mask generator decoder
    _m_sam_decoder = std::make_unique<SamAmgDecoder>();
    _m_sam_decoder->init(cfg);
    if (!_m_sam_decoder->is_successfully_initialized()) {
        LOG(ERROR) << "init sam amg decoder failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_sam_decoder->set_encoder_input_size(_m_sam_encoder_input_size);

    // init decode params
    auto decoder_cfg = cfg["SAM_AMG_DECODER"];
    _m_points_per_side = static_cast<int>(decoder_cfg["points_per_size"].value_or<int64_t>(0));
    _m_pred_iou_thresh = static_cast<float>(decoder_cfg["pred_iou_thresh"].value_or<double>(0.0));
    _m_stability_score_thresh = static_cast<float>(decoder_cfg["stability_score_thresh"].value_or<double>(0.0));
    _m_box_nms_thresh = static_cast<float>(decoder_cfg["box_nms_thresh"].value_or<double>(0.0));
    _m_min_mask_region_area = static_cast<int>(decoder_cfg["min_mask_region_area"].value_or<int64_t>(0));

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load sam auto mask generator model";
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @param amg_output
 * @return
 */
StatusCode Impl::generate(const cv::Mat &input_image, AmgMaskOutput &amg_output) {
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

} // namespace sam_automask_generator_impl

/************ Template Implementation ************/

template <typename INPUT, typename OUTPUT>
SamAutoMaskGenerator<INPUT, OUTPUT>::SamAutoMaskGenerator() {
    _m_pimpl = std::make_unique<sam_automask_generator_impl::Impl>();
}

template <typename INPUT, typename OUTPUT>
SamAutoMaskGenerator<INPUT, OUTPUT>::~SamAutoMaskGenerator() = default;

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode SamAutoMaskGenerator<INPUT, OUTPUT>::init(const toml::table &cfg) {
    return _m_pimpl->init(cfg);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode SamAutoMaskGenerator<INPUT, OUTPUT>::run_impl(const INPUT& input, OUTPUT& output) {
    auto internal_input = sam_automask_generator_impl::transform_input(input);
    sam_automask_generator_impl::internal_output internal_output;
    auto status = _m_pimpl->generate(internal_input, internal_output);
    output = sam_automask_generator_impl::transform_output<OUTPUT>(internal_output);
    return status;
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode SamAutoMaskGenerator<INPUT, OUTPUT>::generate(
    const cv::Mat &input_image, AmgMaskOutput &amg_output) {
    return _m_pimpl->generate(input_image, amg_output);
}

template <typename INPUT, typename OUTPUT>
bool SamAutoMaskGenerator<INPUT, OUTPUT>::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

} // namespace segment_anything
} // namespace models
} // namespace jinq
