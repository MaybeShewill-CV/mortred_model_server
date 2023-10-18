/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: CenterFace.cpp
 * Date: 23-10-18
 ************************************************/

#include "centerface_detector.h"

#include <random>

#include "MNN/Interpreter.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>


#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {

using jinq::common::Base64;
using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::mat_input;

namespace object_detection {

using jinq::models::io_define::object_detection::face_bbox;
using jinq::models::io_define::object_detection::std_face_detection_output;

namespace centerface_impl {

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
    auto image_decode_string = jinq::common::Base64::base64_decode(in.input_image_content);
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
transform_output(const centerface_impl::internal_output &internal_out) {
    std_face_detection_output result;
    for (auto &value : internal_out) {
        result.push_back(value);
    }
    return result;
}

} // namespace centerface_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT> 
class CenterFaceDetector<INPUT, OUTPUT>::Impl {
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
    // model file path
    std::string _m_model_file_path;
    // MNN Interpreter
    std::unique_ptr<MNN::Interpreter> _m_net = nullptr;
    // MNN Session
    MNN::Session *_m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor *_m_input_tensor = nullptr;
    // MNN heatmap Output tensor node
    MNN::Tensor *_m_heatmap_output_tensor = nullptr;
    // MNN scale Output tensor node
    MNN::Tensor *_m_scale_output_tensor = nullptr;
    // MNN offset Output tensor node
    MNN::Tensor *_m_offset_output_tensor = nullptr;
    // MNN landmark Output tensor node
    MNN::Tensor *_m_landmark_output_tensor = nullptr;
    // MNN threads
    uint _m_threads_nums = 4;
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
    // resize session flag
    bool _m_need_resize_tensor = false;
    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     * preprocess
     * @param input_image
     */
    cv::Mat preprocess_image(const cv::Mat &input_image);

    /***
     *
     * @return
     */
    centerface_impl::internal_output decode_output_tensor();
};

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode CenterFaceDetector<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    if (!config.contains("CENTER_FACE")) {
        LOG(ERROR) << "Config missing CENTER_FACE section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("CENTER_FACE");

    // init mnn threads
    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config missing model_threads_num field, use default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = cfg_content["model_threads_num"].as_integer();
    }

    // init score thresh
    if (!cfg_content.contains("model_score_threshold")) {
        LOG(WARNING) << "Config missing model_score_threshold field, use default 0.5";
        _m_score_threshold = 0.5;
    } else {
        _m_score_threshold = cfg_content["model_score_threshold"].as_floating();
    }
    _m_score_threshold = std::max(_m_score_threshold, 0.5);

    // nms thresh
    if (!cfg_content.contains("model_nms_threshold")) {
        LOG(WARNING) << "Config missing model_nms_threshold field, use default 0.3";
        _m_nms_threshold = 0.3;
    } else {
        _m_nms_threshold = cfg_content["model_nms_threshold"].as_floating();
    }

    // top k
    if (!cfg_content.contains("model_keep_top_k")) {
        LOG(WARNING) << "Config missing model_keep_top_k field, use default 250";
        _m_keep_topk = 250;
    } else {
        _m_keep_topk = cfg_content["model_keep_top_k"].as_integer();
    }

    // init mnn interpreter
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
        LOG(ERROR) << "Create CenterFace Interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init mnn session
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
    if (!cfg_content.contains("backend_precision_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_precision_mode field default Precision_Normal";
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(cfg_content.at("backend_precision_mode").as_integer());
    }
    if (!cfg_content.contains("backend_power_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_power_mode field default Power_Normal";
        backend_config.power = MNN::BackendConfig::Power_Normal;
    } else {
        backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(cfg_content.at("backend_power_mode").as_integer());
    }
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);
    if (nullptr == _m_session) {
        LOG(ERROR) << "create center face session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init graph input node and output node
    _m_input_tensor = _m_net->getSessionInput(_m_session, "input.1");
    _m_heatmap_output_tensor = _m_net->getSessionOutput(_m_session, "537");
    _m_scale_output_tensor = _m_net->getSessionOutput(_m_session, "538");
    _m_offset_output_tensor = _m_net->getSessionOutput(_m_session, "539");
    _m_landmark_output_tensor = _m_net->getSessionOutput(_m_session, "540");
    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "fetch center face net input node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_heatmap_output_tensor == nullptr || _m_scale_output_tensor == nullptr ||
        _m_offset_output_tensor == nullptr || _m_landmark_output_tensor == nullptr) {
        LOG(ERROR) << "fetch center face net output node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

    _m_successfully_initialized = true;
    LOG(INFO) << "CenterFace model: " << FilePathUtil::get_file_name(_m_model_file_path) << " initialization complete!!!";
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @return
 */
template <typename INPUT, typename OUTPUT>
cv::Mat CenterFaceDetector<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat &input_image) {
    cv::Mat tmp;
    // bgr to rgb
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);

    // resize image
    int width = input_image.cols;
    int height = input_image.rows;
    int width_resized = static_cast<int>(std::ceil(static_cast<float>(width) / 32.0f) * 32);
    int height_resized = static_cast<int>(std::ceil(static_cast<float>(height) / 32.0f) * 32);
    cv::resize(tmp, tmp, cv::Size(width_resized, height_resized));
    if (width_resized != _m_input_size_host.width || height_resized != _m_input_size_host.height) {
        _m_need_resize_tensor = true;
        _m_input_size_host.width = width_resized;
        _m_input_size_host.height = height_resized;
    } else {
        _m_need_resize_tensor = false;
    }

    // convert to float32
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
StatusCode CenterFaceDetector<INPUT, OUTPUT>::Impl::run(const INPUT &in, OUTPUT &out) {
    // transform external input into internal input
    auto internal_in = centerface_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run session
    if (_m_need_resize_tensor) {
        _m_net->resizeTensor(_m_input_tensor, {1, 3, _m_input_size_host.height, _m_input_size_host.width});
        _m_net->resizeSession(_m_session);
        _m_need_resize_tensor = false;
    }
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, input_chw_image_data.data(), input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);

    // decode output tensor
    auto faces_result = decode_output_tensor();

    // refine bbox coords
    auto width_scale = _m_input_size_user.width / static_cast<float>(_m_input_size_host.width);
    auto height_scale = _m_input_size_user.height / static_cast<float>(_m_input_size_host.height);
    for (auto &face_box : faces_result) {
        face_box.bbox.x *= width_scale;
        face_box.bbox.y *= height_scale;
        face_box.bbox.width *= width_scale;
        face_box.bbox.height *= height_scale;
        for (auto &landmark : face_box.landmarks) {
            landmark.x *= width_scale;
            landmark.y *= height_scale;
        }
    }

    // do nms
    centerface_impl::internal_output nms_result = CvUtils::nms_bboxes(faces_result, _m_nms_threshold);
    if (nms_result.size() > _m_keep_topk) {
        nms_result.resize(_m_keep_topk);
    }

    // transform internal output into external output
    out = centerface_impl::transform_output<OUTPUT>(nms_result);
    return StatusCode::OK;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template <typename INPUT, typename OUTPUT> 
centerface_impl::internal_output CenterFaceDetector<INPUT, OUTPUT>::Impl::decode_output_tensor() {
    // convert tensor format
    MNN::Tensor heatmap_host(_m_heatmap_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    MNN::Tensor scale_host(_m_scale_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    MNN::Tensor offset_host(_m_offset_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    MNN::Tensor landmark_host(_m_landmark_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    _m_heatmap_output_tensor->copyToHostTensor(&heatmap_host);
    _m_scale_output_tensor->copyToHostTensor(&scale_host);
    _m_offset_output_tensor->copyToHostTensor(&offset_host);
    _m_landmark_output_tensor->copyToHostTensor(&landmark_host);

    // decode face info
    int output_width = heatmap_host.width();
    int output_height = heatmap_host.height();
    int channel_step = output_width * output_height;
    std::vector<face_bbox> decode_result;
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            int index = h * output_width + w;
            float score = heatmap_host.host<float>()[index];
            if (score < _m_score_threshold) {
                continue;
            }
            float s0 = 4 * exp(scale_host.host<float>()[index]);
            float s1 = 4 * exp(scale_host.host<float>()[index + channel_step]);
            float o0 = offset_host.host<float>()[index];
            float o1 = offset_host.host<float>()[index + channel_step];

            float ymin = MAX(0, 4 * (h + o0 + 0.5) - 0.5 * s0);
            float xmin = MAX(0, 4 * (w + o1 + 0.5) - 0.5 * s1);
            float ymax = MIN(ymin + s0, _m_input_size_host.height);
            float xmax = MIN(xmin + s1, _m_input_size_host.width);

            face_bbox face_info;
            face_info.score = score;
            face_info.bbox.x = xmin;
            face_info.bbox.y = ymin;
            face_info.bbox.width = (xmax - xmin);
            face_info.bbox.height = (ymax - ymin);

            for (int num = 0; num < 5; ++num) {
                cv::Point2f landmark;
                landmark.x = s1 * landmark_host.host<float>()[(2 * num + 1) * channel_step + index] + xmin;
                landmark.y = s0 * landmark_host.host<float>()[(2 * num + 0) * channel_step + index] + ymin;
                face_info.landmarks.push_back(landmark);
            }
            face_info.class_id = 0;
            decode_result.push_back(face_info);
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
CenterFaceDetector<INPUT, OUTPUT>::CenterFaceDetector() { 
    _m_pimpl = std::make_unique<Impl>(); 
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> 
CenterFaceDetector<INPUT, OUTPUT>::~CenterFaceDetector() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode CenterFaceDetector<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT> 
bool CenterFaceDetector<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode CenterFaceDetector<INPUT, OUTPUT>::run(const INPUT &input, OUTPUT &output) {
    return _m_pimpl->run(input, output);
}

} // namespace object_detection
} // namespace models
} // namespace jinq
