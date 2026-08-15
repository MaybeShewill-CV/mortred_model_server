/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.inl
 * Date: 22-6-10
 ************************************************/

#include "libface_detector.h"
#include "models/cv_image_input.h"

#include <random>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>


#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::base64;
using jinq::common::cv_utils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::mat_input;

namespace object_detection {

using jinq::models::io_define::object_detection::face_bbox;
using jinq::models::io_define::object_detection::std_face_detection_output;

namespace libface_impl {

using internal_input = mat_input;
using internal_output = std_face_detection_output;

struct FaceAnchor {
    double cx;
    double cy;
    double s_kx;
    double s_ky;
};

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
template <typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_face_detection_output>::type>::value, std_face_detection_output>::type
transform_output(const libface_impl::internal_output &internal_out) {
    return internal_out;
}

} // namespace libface_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT> 
class LibFaceDetector<INPUT, OUTPUT>::Impl {
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
    Impl(const Impl &transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl &operator=(const Impl &transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const toml::table &config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode run(const INPUT &in, OUTPUT &out);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const { return _m_successfully_initialized; };

  private:
    jinq::models::MnnNet _m_net;
    // score thresh
    double _m_score_threshold = 0.6;
    // nms thresh
    double _m_nms_threshold = 0.3;
    // top_k keep
    size_t _m_keep_topk = 250;
    // input image size
    cv::Size _m_input_size_user = cv::Size();
    //　input node size
    cv::Size _m_input_size_host = cv::Size();
    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     * preprocess
     * @param input_image
     */
    cv::Mat preprocess_image(const cv::Mat &input_image) const;

    /***
     *
     * @return
     */
    std::vector<libface_impl::FaceAnchor> generate_prior_anchors();

    /***
     *
     * @return
     */
    libface_impl::internal_output decode_output_tensor();
};

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode LibFaceDetector<INPUT, OUTPUT>::Impl::init(const toml::table &config) {
    if (!config.contains("LIBFACE")) {
        LOG(ERROR) << "Config missing LIBFACE section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* cfg_content_ptr = config["LIBFACE"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section LIBFACE missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    // init score thresh
    if (!cfg_content.contains("model_score_threshold")) {
        LOG(WARNING) << "Config missing model_score_threshold field, use default 0.5";
        _m_score_threshold = 0.5;
    } else {
        _m_score_threshold = cfg_content["model_score_threshold"].value_or<double>(0.0);
    }
    _m_score_threshold = std::max(_m_score_threshold, 0.5);

    // nms thresh
    if (!cfg_content.contains("model_nms_threshold")) {
        LOG(WARNING) << "Config missing model_nms_threshold field, use default 0.3";
        _m_nms_threshold = 0.3;
    } else {
        _m_nms_threshold = cfg_content["model_nms_threshold"].value_or<double>(0.0);
    }

    // top k
    if (!cfg_content.contains("model_keep_top_k")) {
        LOG(WARNING) << "Config missing model_keep_top_k field, use default 250";
        _m_keep_topk = 250;
    } else {
        _m_keep_topk = cfg_content["model_keep_top_k"].value_or<int64_t>(0);
    }

    auto init_status = _m_net.init(cfg_content, {"input"}, {"loc", "conf"});
    if (init_status != StatusCode::OK) {
        _m_successfully_initialized = false;
        return init_status;
    }
    _m_input_size_host.width = _m_net.input("input")->width();
    _m_input_size_host.height = _m_net.input("input")->height();

    // init input image size
    if (!cfg_content.contains("model_input_image_size")) {
        LOG(WARNING) << "Config missing model_input_image_size field, use default [320, 240]";
        _m_input_size_user.width = 320;
        _m_input_size_user.height = 240;
    } else {
        _m_input_size_user.width = static_cast<int>(cfg_content["model_input_image_size"][0].value_or<int64_t>(0));
        _m_input_size_user.height = static_cast<int>(cfg_content["model_input_image_size"][1].value_or<int64_t>(0));
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "LibFace model initialization complete!!!";
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @return
 */
template <typename INPUT, typename OUTPUT>
cv::Mat LibFaceDetector<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat &input_image) const {
    cv::Mat tmp;
    if (input_image.size() != _m_input_size_host) {
        cv::resize(input_image, tmp, _m_input_size_host);
    } else {
        input_image.copyTo(tmp);
    }
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    return tmp;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode LibFaceDetector<INPUT, OUTPUT>::Impl::run(const INPUT &in, OUTPUT &out) {
    // transform external input into internal input
    auto internal_in = libface_impl::transform_input(in);

    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = cv_utils::convert_to_chw_vec(preprocessed_image);

    // run session
    MNN::Tensor input_tensor_user(_m_net.input("input"), MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!cv_utils::copy_image_to_tensor(input_tensor_data, input_chw_image_data, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("input")->copyFromHostTensor(&input_tensor_user);
    _m_net.run_session();

    // decode output tensor
    auto faces_result = decode_output_tensor();
    // do nms
    libface_impl::internal_output nms_result = cv_utils::nms_bboxes(faces_result, _m_nms_threshold);
    if (nms_result.size() > _m_keep_topk) {
        nms_result.resize(_m_keep_topk);
    }

    // refine bbox coords
    auto width_scale = _m_input_size_user.width / static_cast<float>(_m_input_size_host.width);
    auto height_scale = _m_input_size_user.height / static_cast<float>(_m_input_size_host.height);
    for (auto &face_box : nms_result) {
        face_box.bbox.x *= width_scale;
        face_box.bbox.y *= height_scale;
        face_box.bbox.width *= width_scale;
        face_box.bbox.height *= height_scale;
        for (auto &landmark : face_box.landmarks) {
            landmark.x *= width_scale;
            landmark.y *= height_scale;
        }
        face_box.category = "face";
    }

    // transform internal output into external output
    out = libface_impl::transform_output<OUTPUT>(nms_result);
    return StatusCode::OK;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT> 
std::vector<libface_impl::FaceAnchor> LibFaceDetector<INPUT, OUTPUT>::Impl::generate_prior_anchors() {

    std::vector<std::vector<double>> min_sizes = {{10., 16., 24.}, {32., 48.}, {64., 96.}, {128., 192., 256.}};
    std::vector<double> steps = {8., 16., 32., 64.};

    auto in_h = _m_input_size_host.height;
    auto in_w = _m_input_size_host.width;

    std::vector<int> feature_map_2th = {int(int((in_h + 1) / 2) / 2), int(int((in_w + 1) / 2) / 2)};
    std::vector<int> feature_map_3th = {int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)};
    std::vector<int> feature_map_4th = {int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)};
    std::vector<int> feature_map_5th = {int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)};
    std::vector<int> feature_map_6th = {int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2)};

    std::vector<std::vector<int>> feature_maps = {feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th};

    std::vector<libface_impl::FaceAnchor> anchors;
    for (size_t k = 0; k < feature_maps.size(); ++k) {
        auto tmp_feature_map = feature_maps[k];
        auto tmp_min_sizes = min_sizes[k];
        for (size_t i = 0; i < tmp_feature_map[0]; ++i) {
            for (size_t j = 0; j < tmp_feature_map[1]; ++j) {
                for (auto min_size : tmp_min_sizes) {
                    double s_kx = min_size / in_w;
                    double s_ky = min_size / in_h;

                    double cx = (static_cast<double>(j) + 0.5) * steps[k] / in_w;
                    double cy = (static_cast<double>(i) + 0.5) * steps[k] / in_h;

                    libface_impl::FaceAnchor tmp_anchor{};
                    tmp_anchor.s_kx = s_kx;
                    tmp_anchor.s_ky = s_ky;
                    tmp_anchor.cx = cx;
                    tmp_anchor.cy = cy;
                    anchors.push_back(tmp_anchor);
                }
            }
        }
    }
    return anchors;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template <typename INPUT, typename OUTPUT> 
libface_impl::internal_output LibFaceDetector<INPUT, OUTPUT>::Impl::decode_output_tensor() {
    // convert tensor format
    MNN::Tensor loc_tensor_user(_m_net.output("loc"), MNN::Tensor::DimensionType::TENSORFLOW);
    _m_net.output("loc")->copyToHostTensor(&loc_tensor_user);
    MNN::Tensor conf_tensor_user(_m_net.output("conf"), MNN::Tensor::DimensionType::TENSORFLOW);
    _m_net.output("conf")->copyToHostTensor(&conf_tensor_user);

    // fetch tensor data
    std::vector<float> loc_tensordata(loc_tensor_user.elementSize());
    ::memcpy(&loc_tensordata[0], loc_tensor_user.host<float>(), loc_tensor_user.elementSize() * sizeof(float));
    std::vector<float> conf_tensordata(conf_tensor_user.elementSize());
    ::memcpy(&conf_tensordata[0], conf_tensor_user.host<float>(), conf_tensor_user.elementSize() * sizeof(float));

    auto batch_nums = loc_tensor_user.shape()[0];
    auto raw_pred_bbox_nums = loc_tensor_user.shape()[1];
    auto priors = generate_prior_anchors();

    std::vector<face_bbox> decode_result;

    for (size_t batch_num = 0; batch_num < batch_nums; ++batch_num) {
        for (size_t bbox_index = 0; bbox_index < raw_pred_bbox_nums; ++bbox_index) {
            auto prior = priors[bbox_index];

            // decode conf
            auto raw_conf = conf_tensordata[bbox_index + raw_pred_bbox_nums];
            if (raw_conf <= _m_score_threshold) {
                continue;
            }
            // decode bbox
            auto raw_bbox_x = loc_tensordata[bbox_index];
            auto raw_bbox_y = loc_tensordata[bbox_index + 1 * raw_pred_bbox_nums];
            auto raw_bbox_w = loc_tensordata[bbox_index + 2 * raw_pred_bbox_nums];
            auto raw_bbox_h = loc_tensordata[bbox_index + 3 * raw_pred_bbox_nums];
            auto pred_bbox_x = prior.cx + raw_bbox_x * 0.1 * prior.s_kx;
            auto pred_bbox_y = prior.cy + raw_bbox_y * 0.1 * prior.s_ky;
            auto pred_bbox_w = prior.s_kx * std::exp(raw_bbox_w * 0.2);
            auto pred_bbox_h = prior.s_ky * std::exp(raw_bbox_h * 0.2);
            pred_bbox_x = (pred_bbox_x - pred_bbox_w / 2.0) * _m_input_size_host.width;
            pred_bbox_y = (pred_bbox_y - pred_bbox_h / 2.0) * _m_input_size_host.height;
            pred_bbox_w *= _m_input_size_host.width;
            pred_bbox_h *= _m_input_size_host.height;
            // decode landmarks
            std::vector<cv::Point2f> landmarks;
            for (size_t landmark_index = 4; landmark_index < 14; landmark_index += 2) {
                auto raw_landmark_x = loc_tensordata[bbox_index + raw_pred_bbox_nums * landmark_index];
                auto raw_landmark_y = loc_tensordata[bbox_index + raw_pred_bbox_nums * (landmark_index + 1)];
                auto pred_landmark_x = (prior.cx + raw_landmark_x * 0.1 * prior.s_kx) * _m_input_size_host.width;
                auto pred_landmark_y = (prior.cy + raw_landmark_y * 0.1 * prior.s_ky) * _m_input_size_host.height;
                landmarks.emplace_back(cv::Point2f(pred_landmark_x, pred_landmark_y));
            }

            face_bbox tmp_face_box;
            tmp_face_box.score = raw_conf;
            tmp_face_box.landmarks = landmarks;
            tmp_face_box.bbox = cv::Rect2f(pred_bbox_x, pred_bbox_y, pred_bbox_w, pred_bbox_h);
            tmp_face_box.class_id = 0;
            decode_result.push_back(tmp_face_box);
        }
    }
    return decode_result;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> 
LibFaceDetector<INPUT, OUTPUT>::LibFaceDetector() { 
    _m_pimpl = std::make_unique<Impl>(); 
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> 
LibFaceDetector<INPUT, OUTPUT>::~LibFaceDetector() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode LibFaceDetector<INPUT, OUTPUT>::init(const toml::table &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT> 
bool LibFaceDetector<INPUT, OUTPUT>::is_successfully_initialized() const {
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
template <typename INPUT, typename OUTPUT> 
StatusCode LibFaceDetector<INPUT, OUTPUT>::run(const INPUT &input, OUTPUT &output) {
    return _m_pimpl->run(input, output);
}

} // namespace object_detection
} // namespace models
} // namespace jinq