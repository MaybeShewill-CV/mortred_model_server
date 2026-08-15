/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mobilenetv2.cpp
* Date: 22-6-13
************************************************/

#include "mobilenetv2.h"
#include "models/cv_image_input.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::cv_utils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::base64;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;

namespace classification {

using jinq::models::io_define::classification::std_classification_output;

namespace mobilenetv2_impl {

using internal_input = mat_input;
using internal_output = std_classification_output;

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template<typename INPUT>
internal_input transform_input(const INPUT& in) {
    internal_input result{};
    result.input_image = jinq::models::cv_input::load_image(in);
    return result;
}

/***
* transform different type of internal output into external output
* @tparam EXTERNAL_OUTPUT
* @tparam dummy
* @param in
* @return
*/
template<typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_classification_output>::type>::value, std_classification_output>::type
transform_output(const mobilenetv2_impl::internal_output& internal_out) {
    return internal_out;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class MobileNetv2<INPUT, OUTPUT>::Impl {
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
    * @param transformer
    */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const toml::table& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

private:
    jinq::models::MnnNet _m_net;
    // class id to names
    std::unordered_map<uint16_t, std::string> _m_class_id2names;
    // MNN Input Tensor Size
    cv::Size _m_input_tensor_size = cv::Size(224, 224);
    // flag
    bool _m_successfully_initialized = false;

private:
    /***
     * preprocess
     * @param input_image : input image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;
};


/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode MobileNetv2<INPUT, OUTPUT>::Impl::init(const toml::table& config) {
    if (!config.contains("MOBILENETV2")) {
        LOG(ERROR) << "Config file does not contain MOBILENETV2 section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* cfg_content_ptr = config["MOBILENETV2"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section MOBILENETV2 is not a table";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    auto init_status = _m_net.init(cfg_content, {"input_tensor"}, {"output_tensor"});
    if (init_status != StatusCode::OK) {
        _m_successfully_initialized = false;
        return init_status;
    }
    if (cfg_content.contains("model_input_image_size")) {
        _m_input_tensor_size.height = static_cast<int>(cfg_content["model_input_image_size"][0].value_or<int64_t>(0));
        _m_input_tensor_size.width = static_cast<int>(cfg_content["model_input_image_size"][1].value_or<int64_t>(0));
    } else {
        auto* input_tensor = _m_net.input("input_tensor");
        _m_input_tensor_size = cv::Size(input_tensor->shape()[3], input_tensor->shape()[2]);
    }

    // init class id to names
    if (cfg_content.contains("class_name_file")) {
        std::string file_path = cfg_content["class_name_file"].value_or<std::string>("");
        if (!FilePathUtil::is_file_exist(file_path)) {
            LOG(WARNING) << "class name file: " << file_path << " not exist";
        } else {
            std::ifstream file(file_path, std::ios::in);
            std::string info;
            uint16_t line_num = 0;
            while (std::getline(file, info)) {
                _m_class_id2names[line_num] = info;
                ++line_num;
            }
            file.close();
        }
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "MobileNetv2 classification model initialization complete !!!";
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode MobileNetv2<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = mobilenetv2_impl::transform_input(in);

    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    auto preprocessed_image = preprocess_image(internal_in.input_image);

    // run session
    MNN::Tensor input_tensor_user(_m_net.input("input_tensor"), MNN::Tensor::DimensionType::TENSORFLOW);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!cv_utils::copy_image_to_tensor(input_tensor_data, preprocessed_image, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("input_tensor")->copyFromHostTensor(&input_tensor_user);
    _m_net.run_session();

    // decode output tensor
    MNN::Tensor output_tensor_user(_m_net.output("output_tensor"), _m_net.output("output_tensor")->getDimensionType());
    _m_net.output("output_tensor")->copyToHostTensor(&output_tensor_user);
    auto* host_data = output_tensor_user.host<float>();
    
    // transform output
    mobilenetv2_impl::internal_output internal_out;

    const int output_size = output_tensor_user.elementSize();
    internal_out.scores.reserve(output_size);
    for (int index = 0; index < output_size; ++index) {
        internal_out.scores.push_back(host_data[index]);
    }
    if (internal_out.scores.empty()) {
        LOG(ERROR) << "classification model output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    auto max_score = std::max_element(internal_out.scores.begin(), internal_out.scores.end());
    auto cls_id = static_cast<int>(std::distance(internal_out.scores.begin(), max_score));
    internal_out.class_id = cls_id;
    if (_m_class_id2names.find(cls_id) != _m_class_id2names.end()) {
        internal_out.category = _m_class_id2names.at(cls_id);
    }
    out = mobilenetv2_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param input_image
 * @return
 */
template<typename INPUT, typename OUTPUT>
cv::Mat MobileNetv2<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
    // resize input image
    cv::Mat tmp;
    cv::resize(input_image, tmp, cv::Size(256, 256));
    auto dw = static_cast<int>(std::floor((256 - _m_input_tensor_size.width) / 2));
    auto dh = static_cast<int>(std::floor((256 - _m_input_tensor_size.height) / 2));
    tmp = tmp(cv::Rect(dw, dh, _m_input_tensor_size.width, _m_input_tensor_size.height));

    // normalize image
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
    tmp.convertTo(tmp, CV_32FC3);
    cv::subtract(tmp, cv::Scalar(123.68f, 116.78f, 103.94f), tmp);
    cv::divide(tmp, cv::Scalar(58.393, 57.12, 57.375), tmp);

    return tmp;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
MobileNetv2<INPUT, OUTPUT>::MobileNetv2() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
MobileNetv2<INPUT, OUTPUT>::~MobileNetv2() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode MobileNetv2<INPUT, OUTPUT>::init(const toml::table& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool MobileNetv2<INPUT, OUTPUT>::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param input
 * @param output
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode MobileNetv2<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}
