/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: OpenAiClip.cpp
 * Date: 23-6-26
 ************************************************/

#include "openai_clip.h"

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

class OpenAiClip::Impl {
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
     * @param input_text
     * @param textual_embeddings
     * @return
     */
    jinq::common::StatusCode get_textual_embedding(const std::string& input_text, std::vector<float>& textual_embeddings);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    jinq::common::StatusCode get_visual_embedding(const cv::Mat& input_image, std::vector<float>& image_embeddings);

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

    // origin image size
    cv::Size _m_ori_image_size;

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
jinq::common::StatusCode OpenAiClip::Impl::init(const decltype(toml::parse("")) &cfg) {
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
 * @param input_image
 * @param textual_embeddings
 * @return
 */
StatusCode OpenAiClip::Impl::get_textual_embedding(const std::string& input_text, std::vector<float> &textual_embeddings) {
    return _m_textual_encoder->encode(input_text, textual_embeddings);
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode OpenAiClip::Impl::get_visual_embedding(const cv::Mat &input_image, std::vector<float> &image_embeddings){
    return _m_visual_encoder->encode(input_image, image_embeddings);
}


/***
 *
 */
OpenAiClip::OpenAiClip() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
OpenAiClip::~OpenAiClip() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode OpenAiClip::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_image
 * @param text_embeddings
 * @return
 */
StatusCode OpenAiClip::get_textual_embedding(const std::string &input_text, std::vector<float> &text_embeddings) {
    return _m_pimpl->get_textual_embedding(input_text, text_embeddings);
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
StatusCode OpenAiClip::get_visual_embedding(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    return _m_pimpl->get_visual_embedding(input_image, image_embeddings);
}

/***
 *
 * @return
 */
bool OpenAiClip::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}