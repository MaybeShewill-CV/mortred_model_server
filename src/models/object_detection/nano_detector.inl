/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: nano_detector.inl
* Date: 22-6-10
************************************************/

#include "nano_detector.h"

#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace mortred {
namespace models {

using mortred::common::FilePathUtil;
using mortred::common::StatusCode;
using mortred::common::Base64;
using mortred::common::CvUtils;
using mortred::models::io_define::common_io::mat_input;
using mortred::models::io_define::common_io::file_input;
using mortred::models::io_define::common_io::base64_input;

namespace object_detection {

using mortred::models::io_define::object_detection::bbox;
using mortred::models::io_define::object_detection::std_object_detection_output;


namespace nano_impl {

struct internal_input {
    cv::Mat input_image;
};

using internal_output = std_object_detection_output;

/***
*
* @tparam INPUT
* @param in
* @return
*/
template<typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<file_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
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
template<typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<mat_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
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
template<typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<base64_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    internal_input result{};
    auto image_decode_string = mortred::common::Base64::base64_decode(in.input_image_content);
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
template<typename OUTPUT>
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_object_detection_output>::type>::value, std_object_detection_output>::type
transform_output(const nano_impl::internal_output& internal_out) {
    std_object_detection_output result;
    for (auto& value : internal_out) {
        result.push_back(value);
    }
    return result;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class NanoDetector<INPUT, OUTPUT>::Impl {
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
    StatusCode init(const decltype(toml::parse(""))& config);

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
    // 模型文件存储路径
    std::string _m_model_file_path;
    // MNN Interpreter
    std::unique_ptr<MNN::Interpreter> _m_net = nullptr;
    // MNN Session
    MNN::Session* _m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor* _m_input_tensor = nullptr;
    // MNN Loc Output tensor node
    MNN::Tensor* _m_output_tensor = nullptr;
    // MNN后端使用线程数
    int _m_threads_nums = 4;
    // 得分阈值
    double _m_score_threshold = 0.4;
    // nms阈值
    double _m_nms_threshold = 0.35;
    // top_k keep阈值
    long _m_keep_topk = 250;
    // 模型类别数量
    int _m_class_nums = 80;
    // 用户输入网络的图像尺寸
    cv::Size _m_input_size_user = cv::Size();
    //　计算图定义的输入node尺寸
    cv::Size _m_input_size_host = cv::Size();
    // 是否成功初始化标志位
    bool _m_successfully_initialized = false;
    // center priors
    std::vector<CenterPrior> _m_center_priors;
    // strides
    std::vector<int> _m_strides = {8, 16, 32, 64};
    // reg max origin
    int _m_reg_max = 7;

private:
    /***
     * 图像预处理, 转换图像为CV_32FC3, 通过dst = src / 127.5 - 1.0来归一化图像到[-1.0, 1.0]
     * @param input_image : 输入图像
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @return
     */
    nano_impl::internal_output decode_output_tensor() const;

    /***
     *
     * @param preds
     * @param ct_x
     * @param ct_y
     * @param stride
     * @return
     */
    std::vector<float> refine_bbox_coords(const float* preds, int ct_x, int ct_y, int stride) const;

    /***
     *
     * @param input_height
     * @param input_width
     * @param strides
     * @param center_priors
     */
    void generate_grid_center_priors();

    /***
     *
     * @param x
     * @return
     */
    static inline float fast_exp(float x) {
        union {
            uint32_t i;
            float f;
        } v{};
        v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
        return v.f;
    }

    /***
     *
     * @param src
     * @param dst
     * @param length
     * @return
     */
    static void activation_function_softmax(const float* src, float* dst, int length) {
        const float alpha = *std::max_element(src, src + length);
        float denominator{0};

        for (int i = 0; i < length; ++i) {
            dst[i] = fast_exp(src[i] - alpha);
            denominator += dst[i];
        }

        for (int i = 0; i < length; ++i) {
            dst[i] /= denominator;
        }
    }
};

/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode NanoDetector<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("NanoDet")) {
        LOG(ERROR) << "Config file does not contain NanoDet section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    toml::value cfg_content = config.at("NanoDet");

    // init threads
    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config doesn\'t have model_threads_num field default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = static_cast<int>(cfg_content.at("model_threads_num").as_integer());
    }

    // init Interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "NanoDet Detection model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(
                 MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));

    if (nullptr == _m_net) {
        LOG(ERROR) << "Create NanoDet detection model interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // init Session
    MNN::ScheduleConfig mnn_config;

    if (!cfg_content.contains("compute_backend")) {
        LOG(WARNING) << "Config doesn\'t have compute_backend field default cpu";
        mnn_config.type = MNN_FORWARD_CPU;
    } else {
        std::string compute_backend = cfg_content.at("compute_backend").as_string();

        if (std::strcmp(compute_backend.c_str(), "cuda") == 0) {
            mnn_config.type = MNN_FORWARD_CUDA;
        } else if (std::strcmp(compute_backend.c_str(), "cpu") == 0) {
            mnn_config.type = MNN_FORWARD_CPU;
        } else {
            LOG(WARNING) << "not supported compute backend use default cpu instead";
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
        LOG(ERROR) << "Create obstacle detection model session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    _m_input_tensor = _m_net->getSessionInput(_m_session, "data");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "output");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch yolov5 detection model input node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch yolov5 detection model output node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

    if (!cfg_content.contains("model_input_image_size")) {
        _m_input_size_user.width = 416;
        _m_input_size_user.height = 416;
    } else {
        _m_input_size_user.width = static_cast<int>(
                                       cfg_content.at("model_input_image_size").as_array()[1].as_integer());
        _m_input_size_user.height = static_cast<int>(
                                        cfg_content.at("model_input_image_size").as_array()[0].as_integer());
    }

    if (!cfg_content.contains("model_score_threshold")) {
        _m_score_threshold = 0.4;
    } else {
        _m_score_threshold = cfg_content.at("model_score_threshold").as_floating();
    }

    if (!cfg_content.contains("model_nms_threshold")) {
        _m_nms_threshold = 0.35;
    } else {
        _m_nms_threshold = cfg_content.at("model_nms_threshold").as_floating();
    }

    if (!cfg_content.contains("model_keep_top_k")) {
        _m_keep_topk = 250;
    } else {
        _m_keep_topk = cfg_content.at("model_keep_top_k").as_integer();
    }

    if (!cfg_content.contains("model_class_nums")) {
        _m_class_nums = 80;
    } else {
        _m_class_nums = static_cast<int>(cfg_content.at("model_class_nums").as_integer());
    }

    // generate center priors
    generate_grid_center_priors();

    _m_successfully_initialized = true;
    LOG(INFO) << "NanoDet detection model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
    return StatusCode::OK;
}

/***
*
* @param input_image
* @return
*/
template<typename INPUT, typename OUTPUT>
cv::Mat NanoDetector<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
    // resize image
    cv::Mat tmp;
    cv::resize(input_image, tmp, _m_input_size_host);

    // normalize
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    cv::divide(tmp, cv::Scalar(255.0f, 255.0f, 255.0f), tmp);
    cv::subtract(tmp, cv::Scalar(0.406, 0.456, 0.485), tmp);
    cv::divide(tmp, cv::Scalar(0.225, 0.224, 0.229), tmp);

    return tmp;
}

/***
*
* @param in
* @param out
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode NanoDetector<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = nano_impl::transform_input(in);

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
    auto bbox_result = decode_output_tensor();

    // do nms
    nano_impl::internal_output nms_result = CvUtils::nms_bboxes(bbox_result, _m_nms_threshold);
    if (nms_result.size() > _m_keep_topk) {
        nms_result.resize(_m_keep_topk);
    }

    // transform internal output into external output
    out = nano_impl::transform_output<OUTPUT>(nms_result);
    return StatusCode::OK;
}

/***
*
* @return
*/
template<typename INPUT, typename OUTPUT>
std::vector<bbox> NanoDetector<INPUT, OUTPUT>::Impl::decode_output_tensor() const {
    // convert tensor format
    MNN::Tensor tensor_preds_host(_m_output_tensor, _m_output_tensor->getDimensionType());
    _m_output_tensor->copyToHostTensor(&tensor_preds_host);

    // decode ouptut tensor
    std::vector<bbox> result;
    const int num_points = static_cast<int>(_m_center_priors.size());
    const int num_channels = _m_class_nums + (_m_reg_max + 1) * 4;

    for (int idx = 0; idx < num_points; idx++) {
        const int ct_x = _m_center_priors[idx].x;
        const int ct_y = _m_center_priors[idx].y;
        const int stride = _m_center_priors[idx].stride;

        const float* scores = tensor_preds_host.host<float>() + (idx * num_channels);
        auto max_score_iter = std::max_element(scores, scores + _m_class_nums);
        float score = *max_score_iter;
        int cur_label = static_cast<int>(std::distance(scores, max_score_iter));

        if (score > _m_score_threshold) {
            const float* bbox_pred = tensor_preds_host.host<float>() + idx * num_channels + _m_class_nums;
            auto obj_box_coords = refine_bbox_coords(bbox_pred, ct_x, ct_y, stride);
            bbox obj_box;
            obj_box.score = score;
            obj_box.class_id = cur_label;
            obj_box.bbox = cv::Rect2f(
                               obj_box_coords[0], obj_box_coords[1],
                               obj_box_coords[2], obj_box_coords[3]);
            result.push_back(obj_box);
        }
    }

    return result;
}

/***
 *
 * @param preds
 * @param ct_x
 * @param ct_y
 * @param stride
 * @return
 */
template<typename INPUT, typename OUTPUT>
std::vector<float> NanoDetector<INPUT, OUTPUT>::Impl::refine_bbox_coords(const float* preds, int x, int y,
        int stride) const {
    auto ct_x = static_cast<float>(x * stride);
    auto ct_y = static_cast<float>(y * stride);
    std::vector<float> dis_pred;
    dis_pred.resize(4);

    for (int i = 0; i < 4; i++) {
        float dis = 0;
        auto* dis_after_sm = new float[_m_reg_max + 1];
        activation_function_softmax(preds + i * (_m_reg_max + 1), dis_after_sm, _m_reg_max + 1);

        for (int j = 0; j < _m_reg_max + 1; j++) {
            dis += static_cast<float>(j) * dis_after_sm[j];
        }

        dis *= static_cast<float>(stride);
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }

    float xmin = std::max(ct_x - dis_pred[0], .0f);
    float ymin = std::max(ct_y - dis_pred[1], .0f);
    float xmax = std::min(ct_x + dis_pred[2], static_cast<float>(_m_input_size_host.width));
    float ymax = std::min(ct_y + dis_pred[3], static_cast<float>(_m_input_size_host.height));

    xmin *= static_cast<float>(_m_input_size_user.width) / static_cast<float>(_m_input_size_host.width);
    ymin *= static_cast<float>(_m_input_size_user.height) / static_cast<float>(_m_input_size_host.height);
    xmax *= static_cast<float>(_m_input_size_user.width) / static_cast<float>(_m_input_size_host.width);
    ymax *= static_cast<float>(_m_input_size_user.height) / static_cast<float>(_m_input_size_host.height);

    return {xmin, ymin, xmax - xmin, ymax - ymin};
}

/***
 *
 */
template<typename INPUT, typename OUTPUT>
void NanoDetector<INPUT, OUTPUT>::Impl::generate_grid_center_priors() {
    for (const auto& stride : _m_strides) {
        int feat_w = std::ceil(static_cast<float>(_m_input_size_host.width) / static_cast<float>(stride));
        int feat_h = std::ceil(static_cast<float>(_m_input_size_host.height) / static_cast<float>(stride));

        for (int y = 0; y < feat_h; y++) {
            for (int x = 0; x < feat_w; x++) {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                _m_center_priors.push_back(ct);
            }
        }
    }
}


/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
NanoDetector<INPUT, OUTPUT>::NanoDetector() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
NanoDetector<INPUT, OUTPUT>::~NanoDetector() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode NanoDetector<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool NanoDetector<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode NanoDetector<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}