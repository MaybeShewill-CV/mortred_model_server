/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov7_detector.inl
* Date: 22-7-14
************************************************/

#include "yolov7_detector.h"
#include "models/cv_image_input.h"
#include "models/cv_image_input.h"

#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::base64;
using jinq::common::cv_utils;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;

namespace object_detection {

using jinq::models::io_define::object_detection::bbox;
using jinq::models::io_define::object_detection::std_object_detection_output;

namespace yolov7_impl {

using internal_input = mat_input;
using internal_output = std_object_detection_output;

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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_object_detection_output>::type>::value, std_object_detection_output>::type
transform_output(const yolov7_impl::internal_output& internal_out) {
    return internal_out;
}
}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class YoloV7Detector<INPUT, OUTPUT>::Impl {
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

public:
    // model save path
    std::string _m_model_file_path;
    // MNN Interpreter
    MNN::Interpreter* _m_net = nullptr;
    // MNN Session
    MNN::Session* _m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor* _m_input_tensor = nullptr;
    // MNN output tensor nodes (3 yolo heads)
    MNN::Tensor* _m_output_tensor_80 = nullptr; // "output": 80x80, stride 8
    MNN::Tensor* _m_output_tensor_40 = nullptr; // "518": 40x40, stride 16
    MNN::Tensor* _m_output_tensor_20 = nullptr; // "532": 20x20, stride 32
    // mnn threads
    int _m_threads_nums = 4;
    // score thresh
    double _m_score_threshold = 0.4;
    // nms thresh
    double _m_nms_threshold = 0.35;
    // top_k keep thresh
    long _m_keep_topk = 250;
    // class nums
    int _m_class_nums = 80;
    // class id to names
    std::map<int, std::string> _m_class_id2names;
    // input image size
    cv::Size _m_input_size_user = cv::Size();
    //　input node size
    cv::Size _m_input_size_host = cv::Size();
    // init flag
    bool _m_successfully_initialized = false;

public:
    /***
     * preprocess
     * @param input_image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @return
     */
    yolov7_impl::internal_output decode_outputs() const;
};

/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode YoloV7Detector<INPUT, OUTPUT>::Impl::init(const toml::table& config) {
    if (!config.contains("YOLOV7")) {
        LOG(ERROR) << "Config file does not contain YOLOV7 section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* cfg_content_ptr = config["YOLOV7"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section YOLOV7 missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    // init threads
    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config doesn\'t have model_threads_num field default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = static_cast<int>(cfg_content["model_threads_num"].value_or<int64_t>(0));
    }

    // init Interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content["model_file_path"].value_or<std::string>("");
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "YOLOV7 Detection model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = MNN::Interpreter::createFromFile(_m_model_file_path.c_str());
    if (nullptr == _m_net) {
        LOG(ERROR) << "Create yolov7 detection model interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init session
    MNN::ScheduleConfig mnn_config;
    if (!cfg_content.contains("compute_backend")) {
        LOG(WARNING) << "Config doesn\'t have compute_backend field default cpu";
        mnn_config.type = MNN_FORWARD_CPU;
    } else {
        std::string compute_backend = cfg_content["compute_backend"].value_or<std::string>("");

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
    if (!cfg_content.contains("backend_precision_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_precision_mode field default Precision_Normal";
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(cfg_content["backend_precision_mode"].value_or<int64_t>(0));
    }
    if (!cfg_content.contains("backend_power_mode")) {
        LOG(WARNING) << "Config doesn\'t have backend_power_mode field default Power_Normal";
        backend_config.power = MNN::BackendConfig::Power_Normal;
    } else {
        backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(cfg_content["backend_power_mode"].value_or<int64_t>(0));
    }
    mnn_config.backendConfig = &backend_config;

    _m_session = _m_net->createSession(mnn_config);

    if (nullptr == _m_session) {
        LOG(ERROR) << "Create obstacle detection model session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_tensor = _m_net->getSessionInput(_m_session, "images");
    _m_output_tensor_80 = _m_net->getSessionOutput(_m_session, "output");
    _m_output_tensor_40 = _m_net->getSessionOutput(_m_session, "518");
    _m_output_tensor_20 = _m_net->getSessionOutput(_m_session, "532");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch yolov7 detection model input node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor_80 == nullptr || _m_output_tensor_40 == nullptr || _m_output_tensor_20 == nullptr) {
        LOG(ERROR) << "Fetch yolov7 detection model output nodes failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

    if (!cfg_content.contains("model_input_image_size")) {
        _m_input_size_user.width = 640;
        _m_input_size_user.height = 640;
    } else {
        _m_input_size_user.width = static_cast<int>(
                                       cfg_content["model_input_image_size"][1].value_or<int64_t>(0));
        _m_input_size_user.height = static_cast<int>(
                                        cfg_content["model_input_image_size"][0].value_or<int64_t>(0));
    }

    if (!cfg_content.contains("model_score_threshold")) {
        _m_score_threshold = 0.4;
    } else {
        _m_score_threshold = cfg_content["model_score_threshold"].value_or<double>(0.0);
    }

    if (!cfg_content.contains("model_nms_threshold")) {
        _m_nms_threshold = 0.35;
    } else {
        _m_nms_threshold = cfg_content["model_nms_threshold"].value_or<double>(0.0);
    }

    if (!cfg_content.contains("model_keep_top_k")) {
        _m_keep_topk = 250;
    } else {
        _m_keep_topk = cfg_content["model_keep_top_k"].value_or<int64_t>(0);
    }

    if (!cfg_content.contains("model_class_nums")) {
        _m_class_nums = 80;
    } else {
        _m_class_nums = static_cast<int>(cfg_content["model_class_nums"].value_or<int64_t>(0));
    }

    if (!cfg_content.contains("class_names")) {
        for (auto idx = 0; idx < _m_class_nums; ++idx) {
            _m_class_id2names.insert(std::make_pair(idx, ""));
        }
    } else {
        const toml::array* cls_names = cfg_content["class_names"].as_array();
    if (cls_names == nullptr) {
        LOG(ERROR) << "Config field class_names is not an array";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
        for (auto idx = 0; idx < cls_names->size(); ++idx) {
            _m_class_id2names.insert(std::make_pair(idx, (*cls_names)[idx].value_or<std::string>("")));
        }
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "YoloV7 detection model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @return
 */
template<typename INPUT, typename OUTPUT>
cv::Mat YoloV7Detector<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
    // resize image
    cv::Mat tmp;
    cv::resize(input_image, tmp, _m_input_size_host);

    // convert bgr 2 rgb
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);

    // normalize
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    tmp /= 255.0;

    return tmp;
}

/***
*
* @param in
* @param out
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode YoloV7Detector<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = yolov7_impl::transform_input(in);

    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    auto preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = cv_utils::convert_to_chw_vec(preprocessed_image);

    // run session
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!cv_utils::copy_image_to_tensor(input_tensor_data, input_chw_image_data, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);

    // decode all output heads
    auto bbox_result = decode_outputs();

    // do nms
    yolov7_impl::internal_output nms_result = cv_utils::nms_bboxes(bbox_result, _m_nms_threshold);
    if (nms_result.size() > _m_keep_topk) {
        nms_result.resize(_m_keep_topk);
    }
    for (auto& bbox : nms_result) {
        auto cls_id = bbox.class_id;
        if (_m_class_id2names.find(cls_id) != _m_class_id2names.end()) {
            bbox.category = _m_class_id2names.at(cls_id);
        }
    }

    // transform internal output into external output
    out = yolov7_impl::transform_output<OUTPUT>(nms_result);
    return StatusCode::OK;
}

/***
*
* @return
*/
template<typename INPUT, typename OUTPUT>
yolov7_impl::internal_output YoloV7Detector<INPUT, OUTPUT>::Impl::decode_outputs() const {
    // yolov7.mnn exports three raw output heads [1, 3, H, W, 85]:
    //   "output" -> 80x80 (stride 8), "518" -> 40x40 (stride 16), "532" -> 20x20 (stride 32)
    // values are raw (pre-activation): xy/wh are grid-relative offsets, obj/cls are logits.
    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

    const MNN::Tensor* heads[3] = {_m_output_tensor_80, _m_output_tensor_40, _m_output_tensor_20};
    const int strides[3] = {8, 16, 32};
    const float anchors[3][3][2] = {
        {{12, 16}, {19, 36}, {40, 28}},
        {{36, 75}, {76, 55}, {72, 146}},
        {{142, 110}, {192, 243}, {459, 401}},
    };

    yolov7_impl::internal_output decode_result;
    for (int hi = 0; hi < 3; ++hi) {
        MNN::Tensor output_tensor_user(heads[hi], MNN::Tensor::DimensionType::CAFFE);
        heads[hi]->copyToHostTensor(&output_tensor_user);
        auto shape = output_tensor_user.shape();
        if (shape.size() != 5) {
            continue;
        }
        int anchor_nums = shape[1];
        int grid_h = shape[2];
        int grid_w = shape[3];
        int attrs = shape[4];
        const float* data = output_tensor_user.host<float>();
        int stride = strides[hi];

        for (int a = 0; a < anchor_nums && a < 3; ++a) {
            float anchor_w = anchors[hi][a][0];
            float anchor_h = anchors[hi][a][1];
            for (int row = 0; row < grid_h; ++row) {
                for (int col = 0; col < grid_w; ++col) {
                    const float* p = data + (((a * grid_h + row) * grid_w + col) * attrs);
                    float obj_score = sigmoid(p[4]);
                    if (obj_score < 0.05f) {
                        continue;
                    }
                    int class_id = -1;
                    float max_cls_score = 0.0f;
                    for (int c = 5; c < attrs; ++c) {
                        float cls_score = sigmoid(p[c]);
                        if (cls_score > max_cls_score) {
                            max_cls_score = cls_score;
                            class_id = c - 5;
                        }
                    }
                    float bbox_score = obj_score * max_cls_score;
                    if (bbox_score < _m_score_threshold) {
                        continue;
                    }
                    float center_x = (2.0f * sigmoid(p[0]) - 0.5f + col) * stride;
                    float center_y = (2.0f * sigmoid(p[1]) - 0.5f + row) * stride;
                    float box_w = std::pow(2.0f * sigmoid(p[2]), 2.0f) * anchor_w;
                    float box_h = std::pow(2.0f * sigmoid(p[3]), 2.0f) * anchor_h;
                    if (box_w <= 0.0f || box_h <= 0.0f) {
                        continue;
                    }
                    bbox tmp_bbox;
                    tmp_bbox.class_id = class_id;
                    tmp_bbox.score = bbox_score;
                    tmp_bbox.bbox.x = center_x - box_w / 2.0f;
                    tmp_bbox.bbox.y = center_y - box_h / 2.0f;
                    tmp_bbox.bbox.width = box_w;
                    tmp_bbox.bbox.height = box_h;
                    if (tmp_bbox.bbox.area() < 5) {
                        continue;
                    }
                    decode_result.push_back(tmp_bbox);
                }
            }
        }
    }

    // rescale boxes from 640-space to the original image size
    auto w_scale = static_cast<float>(_m_input_size_user.width) /
                   static_cast<float>(_m_input_size_host.width);
    auto h_scale = static_cast<float>(_m_input_size_user.height) /
                   static_cast<float>(_m_input_size_host.height);
    for (auto& bbox : decode_result) {
        bbox.bbox.x *= w_scale;
        bbox.bbox.y *= h_scale;
        bbox.bbox.width *= w_scale;
        bbox.bbox.height *= h_scale;
    }
    return decode_result;
}


/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
YoloV7Detector<INPUT, OUTPUT>::YoloV7Detector() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
YoloV7Detector<INPUT, OUTPUT>::~YoloV7Detector() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode YoloV7Detector<INPUT, OUTPUT>::init(const toml::table& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool YoloV7Detector<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode YoloV7Detector<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}
