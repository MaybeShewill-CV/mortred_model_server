/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: OpenAiVitEncoder.cpp
 * Date: 23-6-26
 ************************************************/

#include "openai_clip_vit_encoder.h"

#include <functional>
#include <iterator>
#include <numeric>

#include <chrono>

#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace clip {

class OpenAiClipVitEncoder::Impl {
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
    jinq::common::StatusCode init(const toml::table& cfg);

    /***
     *
     * @param input_image
     * @param image_embeddings
     * @return
     */
    jinq::common::StatusCode encode(const cv::Mat& input_image, std::vector<float>& image_embeddings);

    /***
     *
     * @return
     */
    std::vector<int> get_encoder_input_shape() const {
        return _m_input_shape;
    }

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

  private:
    // MNN runtime (owns interpreter/session/tensors)
    jinq::models::MnnNet _m_net;

    // model input/output shape info
    std::vector<int> _m_input_shape;
    std::vector<int> _m_output_shape;

    // init flag
    bool _m_successfully_init_model = false;

  private:
    /***
     *
     * @param input_image
     * @return
     */
    cv::Mat preprocess_image(const cv::Mat& input_image);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode OpenAiClipVitEncoder::Impl::init(const toml::table &cfg) {
    // init vit encoder configs
    const toml::table* cfg_content_ptr = cfg["OPENAI_CLIP_VIT_ENCODER"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section OPENAI_CLIP_VIT_ENCODER missing or not a table";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;
    auto init_status = _m_net.init(cfg_content, {"input"}, {"output"});
    if (init_status != StatusCode::OK) {
        _m_successfully_init_model = false;
        return init_status;
    }
    _m_input_shape = _m_net.input("input")->shape();
    _m_output_shape = _m_net.output("output")->shape();

    if (_m_input_shape.size() != 4 || _m_output_shape.size() != 2) {
        LOG(ERROR) << "invalid encoder input/output node shape";
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load openai clip vit encoder";
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
jinq::common::StatusCode OpenAiClipVitEncoder::Impl::encode(
    const cv::Mat &input_image,
    std::vector<float> &image_embeddings) {
    // preprocess image
    auto preprocessed_image = preprocess_image(input_image);
    auto input_tensor_values = CvUtils::convert_to_chw_vec(preprocessed_image);
    if (input_tensor_values.empty()) {
        LOG(ERROR) << "empty input data for sam vit encoder";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // run encoder
    auto input_tensor_user = MNN::Tensor(_m_net.input("input"), MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!CvUtils::copy_image_to_tensor(input_tensor_data, input_tensor_values, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("input")->copyFromHostTensor(&input_tensor_user);

    _m_net.run_session();

    MNN::Tensor output_tensor_user(_m_net.output("output"), MNN::Tensor::DimensionType::CAFFE);
    _m_net.output("output")->copyToHostTensor(&output_tensor_user);

    auto embeds_size = std::accumulate(
        std::begin(_m_output_shape), std::end(_m_output_shape), 1, std::multiplies());
    image_embeddings.resize(embeds_size);
    auto img_embeds_val = output_tensor_user.host<float>();
    for (auto idx = 0; idx < embeds_size; ++idx) {
        image_embeddings[idx] = img_embeds_val[idx];
    }

    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @return
 */
cv::Mat OpenAiClipVitEncoder::Impl::preprocess_image(const cv::Mat &input_image) {

    auto input_node_h = static_cast<int>(_m_input_shape[2]);
    auto input_node_w = static_cast<int>(_m_input_shape[3]);

    cv::Mat result;
    cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);
    cv::resize(result, result,cv::Size(input_node_w, input_node_h));
    result.convertTo(result, CV_32FC3);

    cv::divide(result, 255.0, result);
    cv::subtract(result, cv::Scalar(0.48145466, 0.4578275, 0.40821073), result);
    cv::divide(result, cv::Scalar(0.26862954, 0.26130258, 0.27577711), result);

    return result;
}

/***
 *
 */
OpenAiClipVitEncoder::OpenAiClipVitEncoder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
OpenAiClipVitEncoder::~OpenAiClipVitEncoder() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode OpenAiClipVitEncoder::init(const toml::table &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_image
 * @param image_embeddings
 * @return
 */
jinq::common::StatusCode OpenAiClipVitEncoder::encode(const cv::Mat &input_image, std::vector<float> &image_embeddings) {
    return _m_pimpl->encode(input_image, image_embeddings);
}

/***
 *
 * @return
 */
std::vector<int> OpenAiClipVitEncoder::get_encoder_input_shape() const {
    return _m_pimpl->get_encoder_input_shape();
}

/***
 *
 * @return
 */
bool OpenAiClipVitEncoder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}
