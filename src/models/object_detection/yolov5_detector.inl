/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov5_detector.inl
* Date: 22-6-7
************************************************/

#include "yolov5_detector.h"

#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/file_path_util.h"
#include "common/base64.h"

namespace morted {
namespace models {

using morted::common::FilePathUtil;
using morted::common::StatusCode;
using morted::common::Base64;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::common_io::file_input;
using morted::models::io_define::common_io::base64_input;

namespace object_detection {

using morted::models::io_define::object_detection::common_out;

namespace yolov5_impl {

struct internal_input {
    cv::Mat input_image;
};

struct internal_output {
    cv::Rect2f bbox;
    float score = 0.0;
    int32_t class_id;
};

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
        LOG(WARNING) << "input image: " << in.input_image_path << " not exist";
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
    auto image_decode_string = morted::common::Base64::base64_decode(in.input_image_content);
    std::vector<uchar> image_vec_data(image_decode_string.begin(), image_decode_string.end());

    if (image_vec_data.empty()) {
        LOG(WARNING) << "image data empty";
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<common_out>::type>::value, common_out>::type
transform_output(const yolov5_impl::internal_output& internal_out) {
    common_out result;
    result.bbox = internal_out.bbox;
    result.score = internal_out.score;
    return result;
}

/***
*
* @param class_counts
* @return
*/
std::map<int, cv::Scalar> generate_color_map(int class_counts) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 255);

    std::set<int> color_set_r;
    std::set<int> color_set_g;
    std::set<int> color_set_b;
    std::map<int, cv::Scalar> color_map;
    int class_id = 0;

    while (color_map.size() != class_counts) {
        int r = distrib(gen);
        int g = distrib(gen);
        int b = distrib(gen);
        cv::Scalar color(b, g, r);

        if (color_set_r.find(r) != color_set_r.end() && color_set_g.find(g) != color_set_g.end()
                && color_set_b.find(b) != color_set_b.end()) {
            continue;
        } else {
            color_map.insert(std::make_pair(class_id, color));
            color_set_r.insert(r);
            color_set_g.insert(g);
            color_set_b.insert(b);
            class_id++;
        }
    }

    return color_map;
}

/***
 *
 * @param box1
 * @param box2
 * @return
 */
template<typename BBOX1, typename BBOX2>
typename std::enable_if <
std::is_same<BBOX1, std::decay<internal_output>::type>::value&&
std::is_same<BBOX2, std::decay<internal_output>::type>::value, float >::type
calc_iou(const BBOX1& box1, const BBOX2& box2) {
    float x1 = std::max(box1.bbox.x, box2.bbox.x);
    float y1 = std::max(box1.bbox.y, box2.bbox.y);
    float x2 = std::min(box1.bbox.x + box1.bbox.width, box2.bbox.x + box2.bbox.width);
    float y2 = std::min(box1.bbox.y + box1.bbox.height, box2.bbox.y + box2.bbox.height);
    float w = std::max(0.0f, x2 - x1 + 1);
    float h = std::max(0.0f, y2 - y1 + 1);
    float over_area = w * h;
    return over_area /
           (box1.bbox.width * box1.bbox.height + box2.bbox.width * box2.bbox.height - over_area);
}

/***
 *
 * @param bboxes
 * @param nms_threshold
 * @return
 */
std::vector<internal_output> nms_bboxes(std::vector<internal_output>& bboxes, double nms_threshold) {
    std::vector<internal_output> result;

    if (bboxes.empty()) {
        return result;
    }

    std::map<int, std::vector<internal_output> > bboxes_split;

    for (const auto& bbox : bboxes) {
        auto cls_id = bbox.class_id;

        if (bboxes_split.find(cls_id) == bboxes_split.end()) {
            bboxes_split.insert(std::make_pair(cls_id, std::vector<internal_output>({bbox})));
        } else {
            bboxes_split[cls_id].push_back(bbox);
        }
    }

    for (auto& iter : bboxes_split) {
        auto tmp_bboxes = iter.second;
        // sort the bounding boxes by the detection score
        std::sort(tmp_bboxes.begin(), tmp_bboxes.end(),
        [](const internal_output & box1, const internal_output & box2) {
            return box1.score < box2.score;
        });

        while (!tmp_bboxes.empty()) {
            auto last_elem = --std::end(tmp_bboxes);
            const auto& rect1 = last_elem->bbox;

            tmp_bboxes.erase(last_elem);

            for (auto pos = std::begin(tmp_bboxes); pos != std::end(tmp_bboxes);) {
                auto overlap = calc_iou(*last_elem, *pos);

                if (overlap > nms_threshold) {
                    pos = tmp_bboxes.erase(pos);
                } else {
                    ++pos;
                }
            }

            result.push_back(*last_elem);
        }
    }

    return result;
}

}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class YoloV5Detector<INPUT, OUTPUT>::Impl {
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
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

    /***
    *
    * @param in
    * @param out
    * @return
    */
    StatusCode run(const INPUT& in, std::vector<OUTPUT>& out) {
        // transform external input into internal input
        auto internal_in = yolov5_impl::transform_input(in);

        if (!internal_in.input_image.data || internal_in.input_image.empty()) {
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }

        // preprocess image
        auto preprocessed_image = preprocess_image(internal_in.input_image);

        // run session
        MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
        auto input_tensor_data = input_tensor_user.host<float>();
        auto input_tensor_size = input_tensor_user.size();
        ::memcpy(input_tensor_data, preprocessed_image.data, input_tensor_size);
        _m_input_tensor->copyFromHostTensor(&input_tensor_user);
        _m_net->runSession(_m_session);

        // decode output tensor
        auto bbox_result = decode_output_tensor();

        // do nms
        std::vector<yolov5_impl::internal_output> nms_result = yolov5_impl::nms_bboxes(
                    bbox_result, _m_nms_threshold);

        if (nms_result.size() > _m_keep_topk) {
            nms_result.resize(_m_keep_topk);
        }

        // transform internal output into external output
        out.clear();

        for (auto& bbox : nms_result) {
            out.push_back(yolov5_impl::transform_output<OUTPUT>(bbox));
        }

        return StatusCode::OK;
    }

public:
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
    // init color map
    std::map<int, cv::Scalar> _m_color_map;
    // 是否成功初始化标志位
    bool _m_successfully_initialized = false;

public:
    /***
     * 图像预处理, 转换图像为CV_32FC3, 通过dst = src / 127.5 - 1.0来归一化图像到[-1.0, 1.0]
     * @param input_image : 输入图像
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @return
     */
    std::vector<yolov5_impl::internal_output> decode_output_tensor() const;
};

/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode YoloV5Detector<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("YOLOV5")) {
        LOG(ERROR) << "Config文件没有YoloV5相关配置, 请重新检查配置文件";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("YOLOV5");

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
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "YoloV5 Detection model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(
                 MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));

    if (nullptr == _m_net) {
        LOG(ERROR) << "Create yolov5 detection model interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
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
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_tensor = _m_net->getSessionInput(_m_session, "images");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "output");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch yolov5 detection model input node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch yolov5 detection model output node failed";
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

    _m_color_map = yolov5_impl::generate_color_map(_m_class_nums);

    _m_successfully_initialized = true;
    LOG(INFO) << "YoloV5 detection model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @return
 */
template<typename INPUT, typename OUTPUT>
cv::Mat YoloV5Detector<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
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
* @return
*/
template<typename INPUT, typename OUTPUT>
std::vector<yolov5_impl::internal_output> YoloV5Detector<INPUT, OUTPUT>::Impl::decode_output_tensor() const {

    // convert tensor format
    MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::TENSORFLOW);
    _m_output_tensor->copyToHostTensor(&output_tensor_user);

    // fetch tensor data
    std::vector<float> output_tensordata(output_tensor_user.elementSize());
    ::memcpy(&output_tensordata[0], output_tensor_user.host<float>(),
             output_tensor_user.elementSize() * sizeof(float));

    auto batch_nums = output_tensor_user.shape()[0];
    auto raw_pred_bbox_nums = output_tensor_user.shape()[1];

    std::vector<std::vector<float> > raw_output;
    raw_output.resize(raw_pred_bbox_nums);

    for (auto&& tmp : raw_output) {
        tmp.resize(_m_class_nums + 5, 0.0);
    }

    for (auto index = 0; index < raw_pred_bbox_nums; ++index) {
        for (auto idx = 0; idx < _m_class_nums + 5; idx++) {
            raw_output[index][idx] = output_tensordata[index + raw_pred_bbox_nums * idx];
        }
    }

    std::vector<yolov5_impl::internal_output> decode_result;

    for (size_t batch_num = 0; batch_num < batch_nums; ++batch_num) {
        for (size_t bbox_index = 0; bbox_index < raw_pred_bbox_nums; ++bbox_index) {
            std::vector<float> raw_bbox_info = raw_output[bbox_index];
            // thresh bboxes with lower score
            int class_id = -1;
            float max_cls_score = 0.0;

            for (auto cls_idx = 0; cls_idx < _m_class_nums; ++cls_idx) {
                if (raw_bbox_info[cls_idx + 5] > max_cls_score) {
                    max_cls_score = raw_bbox_info[cls_idx + 5];
                    class_id = cls_idx;
                }
            }

            auto bbox_score = raw_bbox_info[4] * max_cls_score;

            if (bbox_score < _m_score_threshold) {
                continue;
            }

            // thresh invalid bboxes
            if (raw_bbox_info[2] <= 0 || raw_bbox_info[3] <= 0) {
                continue;
            }

            auto bbox_area = std::sqrt(raw_bbox_info[2] * raw_bbox_info[3]);

            if (bbox_area < 0 || bbox_area > std::numeric_limits<float>::max()) {
                continue;
            }

            // rescale boxes from img_size to im0 size
            std::vector<float> coords = {
                raw_bbox_info[0] - raw_bbox_info[2] / 2.0f,
                raw_bbox_info[1] - raw_bbox_info[3] / 2.0f,
                raw_bbox_info[0] + raw_bbox_info[2] / 2.0f,
                raw_bbox_info[1] + raw_bbox_info[3] / 2.0f
            };
            auto w_scale = static_cast<float>(_m_input_size_user.width) /
                           static_cast<float>(_m_input_size_host.width);
            auto h_scale = static_cast<float>(_m_input_size_user.height) /
                           static_cast<float>(_m_input_size_host.height);
            coords[0] *= w_scale;
            coords[1] *= h_scale;
            coords[2] *= w_scale;
            coords[3] *= h_scale;

            yolov5_impl::internal_output tmp_bbox;
            tmp_bbox.class_id = class_id;
            tmp_bbox.score = bbox_score;
            tmp_bbox.bbox.x = coords[0];
            tmp_bbox.bbox.y = coords[1];
            tmp_bbox.bbox.width = coords[2] - coords[0];
            tmp_bbox.bbox.height = coords[3] - coords[1];

            if (tmp_bbox.bbox.area() < 5) {
                continue;
            }

            decode_result.push_back(tmp_bbox);
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
template<typename INPUT, typename OUTPUT>
YoloV5Detector<INPUT, OUTPUT>::YoloV5Detector() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
YoloV5Detector<INPUT, OUTPUT>::~YoloV5Detector() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode YoloV5Detector<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool YoloV5Detector<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode YoloV5Detector<INPUT, OUTPUT>::run(const INPUT& input, std::vector<OUTPUT>& output) {
    return _m_pimpl->run(input, output);
}

}
}
}
