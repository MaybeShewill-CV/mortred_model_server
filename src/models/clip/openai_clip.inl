/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: openai_clip.inl
 * Date: 23-6-26
 ************************************************/

#include <cmath>
#include <functional>
#include <numeric>
#include <type_traits>

#include "glog/logging.h"

#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "models/clip/openai_clip_vit_encoder.h"
#include "models/clip/openai_clip_text_encoder.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace clip {

using jinq::models::clip::OpenAiClipVitEncoder;
using jinq::models::clip::OpenAiClipTextEncoder;

namespace openai_clip_impl {

using internal_input = jinq::models::io_define::clip::clip_input;
using internal_output = jinq::models::io_define::clip::clip_output;

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
    result.task_type = jinq::models::io_define::clip::ClipTaskType::IMAGE_EMBEDDING;
    result.image = in.input_image;
    return result;
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::file_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    internal_input result{};
    result.task_type = jinq::models::io_define::clip::ClipTaskType::IMAGE_EMBEDDING;
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
    result.task_type = jinq::models::io_define::clip::ClipTaskType::IMAGE_EMBEDDING;
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
 * OpenAI CLIP 模型内部实现
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
    StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_text
     * @param textual_embeddings
     * @return
     */
    StatusCode get_textual_embedding(const std::string& input_text, std::vector<float>& textual_embeddings);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    StatusCode get_visual_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);

    /***
     *
     * @param input_texts
     * @param input_image
     * @param simi_scores
     * @return
     */
    StatusCode texts2img(
        const std::vector<std::string>& input_texts, const cv::Mat& input_image, std::vector<float>& simi_scores);

    /***
     *
     * @param input_images
     * @param input_text
     * @param simi_scores
     * @return
     */
    StatusCode imgs2text(
        const std::vector<cv::Mat>& input_images, const std::string& input_text, std::vector<float>& simi_scores);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

  private:
    // model
    std::unique_ptr<OpenAiClipVitEncoder> _m_visual_encoder;
    std::unique_ptr<OpenAiClipTextEncoder> _m_textual_encoder;

    // sam vit input image size
    cv::Size _m_vit_encoder_input_size;

    // init flag
    bool _m_successfully_init_model = false;
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode Impl::init(const decltype(toml::parse("")) &cfg) {
    // init sam encoder
    _m_visual_encoder = std::make_unique<OpenAiClipVitEncoder>();
    _m_visual_encoder->init(cfg);
    if (!_m_visual_encoder->is_successfully_initialized()) {
        LOG(ERROR) << "init openai openai_clip vit encoder failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_vit_encoder_input_size.height = _m_visual_encoder->get_encoder_input_shape()[2];
    _m_vit_encoder_input_size.width = _m_visual_encoder->get_encoder_input_shape()[3];

    // init sam vit decoder
    _m_textual_encoder = std::make_unique<OpenAiClipTextEncoder>();
    _m_textual_encoder->init(cfg);
    if (!_m_textual_encoder->is_successfully_initialized()) {
        LOG(ERROR) << "init openai openai_clip text encoder failed";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load openai openai_clip model";
    return StatusCode::OJBK;
}

/***
 *
 * @param input_text
 * @param textual_embeddings
 * @return
 */
StatusCode Impl::get_textual_embedding(const std::string& input_text, std::vector<float> &textual_embeddings) {
    return _m_textual_encoder->encode(input_text, textual_embeddings);
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode Impl::get_visual_embedding(const cv::Mat &input_image, std::vector<float> &image_embeddings){
    return _m_visual_encoder->encode(input_image, image_embeddings);
}

/***
 *
 * @param input_texts
 * @param input_image
 * @param simi_scores
 * @return
 */
StatusCode Impl::texts2img(
    const std::vector<std::string> &input_texts, const cv::Mat &input_image, std::vector<float> &simi_scores) {
    // get visual features
    std::vector<float> vis_feats;
    auto status = get_visual_embedding(input_image, vis_feats);
    if (status != StatusCode::OJBK) {
        LOG(ERROR) << "get visual features failed, status: " << status;
        return status;
    }
    auto vis_feats_norm = std::inner_product(vis_feats.begin(), vis_feats.end(), vis_feats.begin(), 0.0f);
    vis_feats_norm = std::sqrt(vis_feats_norm);
    for (auto& val : vis_feats) {
        val /= vis_feats_norm;
    }

    // get text features
    std::vector<std::vector<float> > texts_feats;
    for (auto& text : input_texts) {
        std::vector<float> text_feats;
        status = get_textual_embedding(text, text_feats);
        if (status != StatusCode::OK) {
            LOG(ERROR) << "get textual features failed, status: " << status;
            return status;
        } else {
            auto text_feats_norm = std::inner_product(
                text_feats.begin(), text_feats.end(), text_feats.begin(), 0.0f);
            text_feats_norm = std::sqrt(text_feats_norm);
            for (auto& val : text_feats) {
                val /= text_feats_norm;
            }
            texts_feats.push_back(text_feats);
        }
    }

    // calculate simi scores
    std::vector<float> simi_values;
    for (auto& text_feats : texts_feats) {
        auto inner_product = std::inner_product(vis_feats.begin(), vis_feats.end(), text_feats.begin(), 0.0f);
        auto simi_value = std::exp(100.0f * inner_product);
        simi_values.push_back(simi_value);
    }
    auto sum_values = std::accumulate(simi_values.begin(), simi_values.end(), 0.0f, std::plus{});
    simi_scores.resize(0);
    for (auto val : simi_values) {
        auto score = val / sum_values;
        simi_scores.push_back(score);
    }

    return StatusCode::OK;
}

/***
 *
 * @param input_images
 * @param input_text
 * @param simi_scores
 * @return
 */
StatusCode Impl::imgs2text(
    const std::vector<cv::Mat> &input_images, const std::string &input_text, std::vector<float> &simi_scores) {
    // get textual features
    std::vector<float> text_feats;
    auto status = get_textual_embedding(input_text, text_feats);
    if (status != StatusCode::OJBK) {
        LOG(ERROR) << "get textual features failed, status: " << status;
        return status;
    }
    auto text_feats_norm = std::inner_product(text_feats.begin(), text_feats.end(), text_feats.begin(), 0.0f);
    text_feats_norm = std::sqrt(text_feats_norm);
    for (auto& val : text_feats) {
        val /= text_feats_norm;
    }

    // get visual features
    std::vector<std::vector<float> > visuals_feats;
    for (auto& image : input_images) {
        std::vector<float> vis_feats;
        status = get_visual_embedding(image, vis_feats);
        if (status != StatusCode::OK) {
            LOG(ERROR) << "get visual features failed, status: " << status;
            return status;
        } else {
            auto vis_feats_norm = std::inner_product(
                vis_feats.begin(), vis_feats.end(), vis_feats.begin(), 0.0f);
            vis_feats_norm = std::sqrt(vis_feats_norm);
            for (auto& val : vis_feats) {
                val /= vis_feats_norm;
            }
            visuals_feats.push_back(vis_feats);
        }
    }

    // calculate simi scores
    std::vector<float> simi_values;
    for (auto& vis_feats : visuals_feats) {
        auto inner_product = std::inner_product(text_feats.begin(), text_feats.end(), vis_feats.begin(), 0.0f);
        auto simi_value = std::exp(100.0f * inner_product);
        simi_values.push_back(simi_value);
    }
    auto sum_values = std::accumulate(simi_values.begin(), simi_values.end(), 0.0f, std::plus{});
    simi_scores.resize(0);
    for (auto val : simi_values) {
        auto score = val / sum_values;
        simi_scores.push_back(score);
    }

    return StatusCode::OK;
}

} // namespace openai_clip_impl

/************ Template Implementation ************/

template <typename INPUT, typename OUTPUT>
OpenAiClip<INPUT, OUTPUT>::OpenAiClip() {
    _m_pimpl = std::make_unique<openai_clip_impl::Impl>();
}

template <typename INPUT, typename OUTPUT>
OpenAiClip<INPUT, OUTPUT>::~OpenAiClip() = default;

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode OpenAiClip<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode OpenAiClip<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    auto internal_input = openai_clip_impl::transform_input(input);
    openai_clip_impl::internal_output internal_output;
    auto status = StatusCode::OK;

    switch (internal_input.task_type) {
        case jinq::models::io_define::clip::ClipTaskType::TEXT_EMBEDDING:
            status = _m_pimpl->get_textual_embedding(internal_input.text, internal_output.embeddings);
            break;
        case jinq::models::io_define::clip::ClipTaskType::IMAGE_EMBEDDING:
            status = _m_pimpl->get_visual_embedding(internal_input.image, internal_output.embeddings);
            break;
        case jinq::models::io_define::clip::ClipTaskType::TEXTS_TO_IMAGE:
            status = _m_pimpl->texts2img(internal_input.texts, internal_input.image, internal_output.simi_scores);
            break;
        case jinq::models::io_define::clip::ClipTaskType::IMAGES_TO_TEXT:
            status = _m_pimpl->imgs2text(internal_input.images, internal_input.text, internal_output.simi_scores);
            break;
        default:
            status = StatusCode::MODEL_RUN_SESSION_FAILED;
            break;
    }

    output = openai_clip_impl::transform_output<OUTPUT>(internal_output);
    return status;
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode OpenAiClip<INPUT, OUTPUT>::get_textual_embedding(
    const std::string &input_text, std::vector<float> &text_embeddings) {
    return _m_pimpl->get_textual_embedding(input_text, text_embeddings);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode OpenAiClip<INPUT, OUTPUT>::get_visual_embedding(
    const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    return _m_pimpl->get_visual_embedding(input_image, image_embeddings);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode OpenAiClip<INPUT, OUTPUT>::texts2img(
    const std::vector<std::string> &input_texts, const cv::Mat &input_image, std::vector<float> &simi_scores) {
   return _m_pimpl->texts2img(input_texts, input_image, simi_scores);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode OpenAiClip<INPUT, OUTPUT>::imgs2text(
    const std::vector<cv::Mat> &input_images, const std::string &input_text, std::vector<float> &simi_scores) {
   return _m_pimpl->imgs2text(input_images, input_text, simi_scores);
}

template <typename INPUT, typename OUTPUT>
bool OpenAiClip<INPUT, OUTPUT>::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

} // namespace clip
} // namespace models
} // namespace jinq
