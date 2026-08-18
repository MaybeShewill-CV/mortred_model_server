/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov5_detector.inl
* Date: 22-6-7
************************************************/

#include "yolov5_detector.h"
#include "models/cv_image_input.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::CvUtils;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;

namespace object_detection {

using jinq::models::io_define::object_detection::bbox;
using jinq::models::io_define::object_detection::std_object_detection_output;

namespace yolov5_impl {

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
transform_output(const yolov5_impl::internal_output& internal_out) {
    return internal_out;
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

public:
    jinq::models::MnnNet _m_net;
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
     * @param input_image : input image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @return
     */
    yolov5_impl::internal_output decode_output_tensor() const;
};

/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode YoloV5Detector<INPUT, OUTPUT>::Impl::init(const toml::table& config) {
    if (!config.contains("YOLOV5")) {
        LOG(ERROR) << "Config file does not contain YOLOV5 section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const toml::table* cfg_content_ptr = config["YOLOV5"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section YOLOV5 missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;

    auto init_status = _m_net.init(cfg_content, {"images"}, {"output"});
    if (init_status != StatusCode::OK) {
        _m_successfully_initialized = false;
        return init_status;
    }
    _m_input_size_host.width = _m_net.input("images")->width();
    _m_input_size_host.height = _m_net.input("images")->height();

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
        for (size_t idx = 0; idx < cls_names->size(); ++idx) {
            _m_class_id2names.insert(std::make_pair(idx, (*cls_names)[idx].value_or<std::string>("")));
        }
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "YoloV5 detection model initialization complete!!!";
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
* @param in
* @param out
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode YoloV5Detector<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = yolov5_impl::transform_input(in);

    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    // record the original input image size for box rescaling
    _m_input_size_user = internal_in.input_image.size();
    auto preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = CvUtils::convert_to_chw_vec(preprocessed_image);

    // run session
    MNN::Tensor input_tensor_user(_m_net.input("images"), MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    if (!CvUtils::copy_image_to_tensor(input_tensor_data, input_chw_image_data, input_tensor_size)) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input("images")->copyFromHostTensor(&input_tensor_user);
    _m_net.run_session();

    // decode output tensor
    auto bbox_result = decode_output_tensor();

    // do nms
    yolov5_impl::internal_output nms_result = CvUtils::nms_bboxes(bbox_result, _m_nms_threshold);
    if (nms_result.size() > static_cast<size_t>(_m_keep_topk)) {
        nms_result.resize(_m_keep_topk);
    }
    for (auto& bbox : nms_result) {
        auto cls_id = bbox.class_id;
        if (_m_class_id2names.find(cls_id) != _m_class_id2names.end()) {
            bbox.category = _m_class_id2names.at(cls_id);
        }
    }

    // transform internal output into external output
    out = yolov5_impl::transform_output<OUTPUT>(nms_result);
    return StatusCode::OK;
}

/***
*
* @return
*/
template<typename INPUT, typename OUTPUT>
yolov5_impl::internal_output YoloV5Detector<INPUT, OUTPUT>::Impl::decode_output_tensor() const {

    // convert tensor format
    MNN::Tensor output_tensor_user(_m_net.output("output"), MNN::Tensor::DimensionType::CAFFE);
    _m_net.output("output")->copyToHostTensor(&output_tensor_user);

    // fetch tensor data
    std::vector<float> output_tensordata(output_tensor_user.elementSize());
    ::memcpy(&output_tensordata[0], output_tensor_user.host<float>(),
             output_tensor_user.elementSize() * sizeof(float));

    const auto& tensor_shape = output_tensor_user.shape();
    if (tensor_shape.size() < 2) {
        LOG(ERROR) << "unexpected output tensor shape";
        return yolov5_impl::internal_output();
    }
    auto batch_nums = tensor_shape[0];
    auto raw_pred_bbox_nums = tensor_shape[1];
    const size_t row_size = static_cast<size_t>(_m_class_nums + 5);

    yolov5_impl::internal_output decode_result;

    for (int batch_num = 0; batch_num < batch_nums; ++batch_num) {
        const size_t batch_offset = batch_num * raw_pred_bbox_nums * row_size;
        for (int bbox_index = 0; bbox_index < raw_pred_bbox_nums; ++bbox_index) {
            const size_t offset = batch_offset + bbox_index * row_size;
            // thresh bboxes with lower score
            int class_id = -1;
            float max_cls_score = 0.0;

            for (auto cls_idx = 0; cls_idx < _m_class_nums; ++cls_idx) {
                const float cls_score = output_tensordata[offset + cls_idx + 5];
                if (cls_score > max_cls_score) {
                    max_cls_score = cls_score;
                    class_id = cls_idx;
                }
            }

            const float obj_score = output_tensordata[offset + 4];
            auto bbox_score = obj_score * max_cls_score;

            if (bbox_score < _m_score_threshold) {
                continue;
            }

            const float box_w = output_tensordata[offset + 2];
            const float box_h = output_tensordata[offset + 3];
            // thresh invalid bboxes
            if (box_w <= 0 || box_h <= 0) {
                continue;
            }

            // rescale boxes from img_size to im0 size
            std::vector<float> coords = {
                output_tensordata[offset + 0] - box_w / 2.0f,
                output_tensordata[offset + 1] - box_h / 2.0f,
                output_tensordata[offset + 0] + box_w / 2.0f,
                output_tensordata[offset + 1] + box_h / 2.0f
            };
            auto w_scale = static_cast<float>(_m_input_size_user.width) /
                           static_cast<float>(_m_input_size_host.width);
            auto h_scale = static_cast<float>(_m_input_size_user.height) /
                           static_cast<float>(_m_input_size_host.height);
            coords[0] *= w_scale;
            coords[1] *= h_scale;
            coords[2] *= w_scale;
            coords[3] *= h_scale;

            bbox tmp_bbox;
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
StatusCode YoloV5Detector<INPUT, OUTPUT>::init(const toml::table& cfg) {
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
StatusCode YoloV5Detector<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}
