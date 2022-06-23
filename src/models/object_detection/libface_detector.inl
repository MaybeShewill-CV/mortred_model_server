/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.inl
 * Date: 22-6-10
 ************************************************/

#include "libface_detector.h"

#include <random>

#include "MNN/Interpreter.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>


#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace morted {
namespace models {

using morted::common::Base64;
using morted::common::CvUtils;
using morted::common::FilePathUtil;
using morted::common::StatusCode;
using morted::models::io_define::common_io::base64_input;
using morted::models::io_define::common_io::file_input;
using morted::models::io_define::common_io::mat_input;

namespace object_detection {

using morted::models::io_define::object_detection::face_bbox;
using morted::models::io_define::object_detection::std_face_detection_output;

namespace libface_impl {

struct internal_input {
    cv::Mat input_image;
};

using internal_output = std_face_detection_output;

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<file_input>::type>::value, internal_input>::type 
transform_input(const INPUT &in) {
    internal_input result{};

    if (!FilePathUtil::is_file_exist(in.input_image_path)) {
        DLOG(WARNING) << "input image: " << in.input_image_path << " not exist";
        return result;
    }

    result.input_image = cv::imread(in.input_image_path, cv::IMREAD_UNCHANGED);
    return result;
}

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<mat_input>::type>::value, internal_input>::type 
transform_input(const INPUT &in) {
    internal_input result{};
    result.input_image = in.input_image;
    return result;
}

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<base64_input>::type>::value, internal_input>::type 
transform_input(const INPUT &in) {
    internal_input result{};
    auto image_decode_string = morted::common::Base64::base64_decode(in.input_image_content);
    std::vector<uchar> image_vec_data(image_decode_string.begin(), image_decode_string.end());

    if (image_vec_data.empty()) {
        DLOG(WARNING) << "image data empty";
        return result;
    } else {
        cv::Mat ret;
        cv::imdecode(image_vec_data, cv::IMREAD_UNCHANGED).copyTo(result.input_image);
        return result;
    }
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
    std_face_detection_output result;
    for (auto &value : internal_out) {
        result.push_back(value);
    }
    return result;
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
    ~Impl() {
        if (_m_net != nullptr && _m_session != nullptr) {
            _m_net->releaseModel();
            _m_net->releaseSession(_m_session);
        }
    }

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
    StatusCode init(const decltype(toml::parse("")) &config);

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
    // 模型文件存储路径
    std::string _m_model_file_path;
    // MNN Interpreter
    std::unique_ptr<MNN::Interpreter> _m_net = nullptr;
    // MNN Session
    MNN::Session *_m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor *_m_input_tensor = nullptr;
    // MNN Loc Output tensor node
    MNN::Tensor *_m_loc_output_tensor = nullptr;
    // MNN conf Output tensor node
    MNN::Tensor *_m_conf_output_tensor = nullptr;
    // MNN后端使用线程数
    uint _m_threads_nums = 4;
    // 得分阈值
    double _m_score_threshold = 0.6;
    // nms阈值
    double _m_nms_threshold = 0.3;
    // top_k keep阈值
    size_t _m_keep_topk = 250;
    // 用户输入网络的图像尺寸
    cv::Size _m_input_size_user = cv::Size();
    //　计算图定义的输入node尺寸
    cv::Size _m_input_size_host = cv::Size();
    // 是否成功初始化标志位
    bool _m_successfully_initialized = false;

    struct FaceAnchor {
        double cx;
        double cy;
        double s_kx;
        double s_ky;
    };

  private:
    /***
     * 图像预处理, 转换图像为CV_32FC3, 通过dst = src / 127.5 - 1.0来归一化图像到[-1.0, 1.0]
     * @param input_image : 输入图像
     */
    cv::Mat preprocess_image(const cv::Mat &input_image) const;

    /***
     *
     * @return
     */
    std::vector<FaceAnchor> generate_prior_anchors();

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
StatusCode LibFaceDetector<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    if (!config.contains("LIBFACE")) {
        LOG(ERROR) << "Config missing LIBFACE section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("LIBFACE");

    // 初始化线程模型计算使用的线程数
    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config missing model_threads_num field, use default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = cfg_content["model_threads_num"].as_integer();
    }

    // 初始化得分阈值
    if (!cfg_content.contains("model_score_threshold")) {
        LOG(WARNING) << "Config missing model_score_threshold field, use default 0.5";
        _m_score_threshold = 0.5;
    } else {
        _m_score_threshold = cfg_content["model_score_threshold"].as_floating();
    }
    _m_score_threshold = std::max(_m_score_threshold, 0.5);

    // nms阈值
    if (!cfg_content.contains("model_nms_threshold")) {
        LOG(WARNING) << "Config missing model_nms_threshold field, use default 0.3";
        _m_nms_threshold = 0.3;
    } else {
        _m_nms_threshold = cfg_content["model_nms_threshold"].as_floating();
    }

    // top k阈值
    if (!cfg_content.contains("model_keep_top_k")) {
        LOG(WARNING) << "Config missing model_keep_top_k field, use default 250";
        _m_keep_topk = 250;
    } else {
        _m_keep_topk = cfg_content["model_keep_top_k"].as_integer();
    }

    // 初始化MNN Interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config missing model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content["model_file_path"].as_string();
    }
    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));
    if (nullptr == _m_net) {
        LOG(ERROR) << "Create LibFace Interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // 初始化MNN Session
    MNN::ScheduleConfig mnn_config;
    if (!cfg_content.contains("compute_backend")) {
        LOG(WARNING) << "Config missing compute_backend field, use default cpu";
        mnn_config.type = MNN_FORWARD_CPU;
    } else {
        std::string compute_backend = cfg_content.at("compute_backend").as_string();
        if (std::strcmp(compute_backend.c_str(), "cuda") == 0) {
            mnn_config.type = MNN_FORWARD_CUDA;
        } else if (std::strcmp(compute_backend.c_str(), "cpu") == 0) {
            mnn_config.type = MNN_FORWARD_CPU;
        } else {
            LOG(WARNING) << "not support compute backend, use default cpu";
            mnn_config.type = MNN_FORWARD_CPU;
        }
    }
    mnn_config.numThread = _m_threads_nums;

    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_High;
    backend_config.power = MNN::BackendConfig::Power_High;
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);
    if (nullptr == _m_session) {
        LOG(ERROR) << "create libface session failed";
        _m_successfully_initialized = false;
        return;
    }

    // 初始化graph input node和output node
    _m_input_tensor = _m_net->getSessionInput(_m_session, "input");
    _m_loc_output_tensor = _m_net->getSessionOutput(_m_session, "loc");
    _m_conf_output_tensor = _m_net->getSessionOutput(_m_session, "conf");
    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "fetch libface net input node failed";
        _m_successfully_initialized = false;
        return;
    }
    if (_m_loc_output_tensor == nullptr || _m_conf_output_tensor == nullptr) {
        LOG(ERROR) << "fetch libface net output node failed";
        _m_successfully_initialized = false;
        return;
    }
    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

    // 初始化用户输入的图像归一化尺寸
    if (!cfg_content.contains("model_input_image_size")) {
        LOG(WARNING) << "Config missing model_input_image_size field, use default [320, 240]";
        _m_input_size_user.width = 320;
        _m_input_size_user.height = 240;
    } else {
        _m_input_size_user.width = cfg_content["model_input_image_size"].as_array()[0].as_integer();
        _m_input_size_user.height = cfg_content["model_input_image_size"].as_array()[1].as_integer();
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "LibFace model: " << FilePathUtil::get_file_name(_m_model_file_path) << " initialization complete!!!";
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
    cv::Mat input_image_copy = preprocess_image(internal_in.input_image);
    // run session
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, input_image_copy.data, input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);

    // decode output tensor
    auto faces_result = decode_output_tensor();
    // do nms
    libface_impl::internal_output nms_result = CvUtils::nms(faces_result, _m_nms_threshold);
    if (nms_result.size() > _m_keep_topk) {
        nms_result.resize(_m_keep_topk);
    }

    // refine bbox coords
    auto width_scale = _m_input_size_user.width / static_cast<double>(_m_input_size_host.width);
    auto height_scale = _m_input_size_user.height / static_cast<double>(_m_input_size_host.height);
    for (auto &face_box : nms_result) {
        face_box.bbox.x *= width_scale;
        face_box.bbox.y *= height_scale;
        face_box.bbox.width *= width_scale;
        face_box.bbox.height *= height_scale;
        for (auto &landmark : face_box.landmarks) {
            landmark.x *= width_scale;
            landmark.y *= height_scale;
        }
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
template <typename INPUT, typename OUTPUT> 
std::vector<FaceAnchor> LibFaceDetector<INPUT, OUTPUT>::Impl::generate_prior_anchors() {

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

    std::vector<FaceAnchor> anchors;
    for (size_t k = 0; k < feature_maps.size(); ++k) {
        auto tmp_feature_map = feature_maps[k];
        auto tmp_min_sizes = min_sizes[k];
        for (size_t i = 0; i < tmp_feature_map[0]; ++i) {
            for (size_t j = 0; j < tmp_feature_map[1]; ++j) {
                for (auto min_size : tmp_min_sizes) {
                    double s_kx = min_size / in_w;
                    double s_ky = min_size / in_h;

                    double cx = (j + 0.5) * steps[k] / in_w;
                    double cy = (i + 0.5) * steps[k] / in_h;

                    FaceAnchor tmp_anchor{};
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
    MNN::Tensor loc_tensor_user(_m_loc_output_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    _m_loc_output_tensor->copyToHostTensor(&loc_tensor_user);
    MNN::Tensor conf_tensor_user(_m_conf_output_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    _m_conf_output_tensor->copyToHostTensor(&conf_tensor_user);

    // fetch tensor data
    std::vector<float> loc_tensordata(loc_tensor_user.elementSize());
    ::memcpy(&loc_tensordata[0], loc_tensor_user.host<float>(), loc_tensor_user.elementSize() * sizeof(float));
    std::vector<float> conf_tensordata(conf_tensor_user.elementSize());
    ::memcpy(&conf_tensordata[0], conf_tensor_user.host<float>(), conf_tensor_user.elementSize() * sizeof(float));

    auto batch_nums = loc_tensor_user.shape()[0];
    auto raw_pred_bbox_nums = loc_tensor_user.shape()[1];
    auto priors = generate_prior_anchors();

    std::vector<FaceBox> decode_result;

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
            tmp_face_box.conf = raw_conf;
            tmp_face_box.landmarks = landmarks;
            tmp_face_box.bbox = cv::Rect2f(pred_bbox_x, pred_bbox_y, pred_bbox_w, pred_bbox_h);
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
StatusCode LibFaceDetector<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
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
} // namespace morted