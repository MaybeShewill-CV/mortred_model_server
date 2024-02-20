/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: db_text_detector.cpp
* Date: 22-6-6
************************************************/

#include "db_text_detector.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/base64.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::Base64;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::base64_input;

namespace ocr {
using jinq::models::io_define::ocr::text_region;
using jinq::models::io_define::ocr::std_text_regions_output;

namespace dbtext_impl {

using internal_input = mat_input;
using internal_output = std_text_regions_output;

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
    return in;
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
    auto image_decode_string = jinq::common::Base64::base64_decode(in.input_image_content);
    std::vector<uchar> image_vec_data(image_decode_string.begin(), image_decode_string.end());

    if (image_vec_data.empty()) {
        DLOG(WARNING) << "image data empty";
        return result;
    } else {
        cv::Mat ret;
        result.input_image = cv::imdecode(image_vec_data, cv::IMREAD_UNCHANGED);
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_text_regions_output>::type>::value, std_text_regions_output>::type
transform_output(const dbtext_impl::internal_output& internal_out) {
    return internal_out;
}


}

/***************** Impl Function Sets ******************/

template<typename INPUT, typename OUTPUT>
class DBTextDetector<INPUT, OUTPUT>::Impl {
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
    // model file path
    std::string _m_model_file_path;
    // MNN Interpreter
    MNN::Interpreter* _m_net = nullptr;
    // MNN Session
    MNN::Session* _m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor* _m_input_tensor = nullptr;
    // MNN Output tensor node
    MNN::Tensor* _m_output_tensor = nullptr;
    // threads nums
    int _m_threads_nums = 4;
    // score thresh
    double _m_score_threshold = 0.4;
    // rotate bbox short side thresh
    float _m_sside_threshold = 3;
    // top_k keep thresh
    long _m_keep_topk = 250;
    // user input size
    cv::Size _m_input_size_user = cv::Size();
    //　input tensor size
    cv::Size _m_input_size_host = cv::Size();
    // segmentation prob mat
    cv::Mat _m_seg_prob_mat;
    // segmentation score map
    cv::Mat _m_seg_score_mat;
    // init flag
    bool _m_successfully_initialized = false;

private:
    /***
     * preprocess func
     * @param input_image : input image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @return
     */
    dbtext_impl::internal_output postprocess() const;

    /***
     *
     * @return
     */
    void decode_segmentation_result_mat() const;

    /***
     *
     * @param seg_probs_mat
     * @return
     */
    dbtext_impl::internal_output get_boxes_from_bitmap() const;
};


/***
*
* @param cfg_file_path
* @return
*/
template<typename INPUT, typename OUTPUT>
StatusCode DBTextDetector<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("DB_TEXT")) {
        LOG(ERROR) << "Config file missing DB_TEXT section please check";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("DB_TEXT");

    // init threads
    if (!cfg_content.contains("model_threads_num")) {
        LOG(WARNING) << "Config doesn\'t have model_threads_num field default 4";
        _m_threads_nums = 4;
    } else {
        _m_threads_nums = static_cast<int>(cfg_content.at("model_threads_num").as_integer());
    }

    // init interpreter
    if (!cfg_content.contains("model_file_path")) {
        LOG(ERROR) << "Config doesn\'t have model_file_path field";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    } else {
        _m_model_file_path = cfg_content.at("model_file_path").as_string();
    }

    if (!FilePathUtil::is_file_exist(_m_model_file_path)) {
        LOG(ERROR) << "DB_TEXT Detection model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = MNN::Interpreter::createFromFile(_m_model_file_path.c_str());
    if (nullptr == _m_net) {
        LOG(ERROR) << "Create db_text detection model interpreter failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    // init session
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
        LOG(ERROR) << "Create db_text detection model session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_tensor = _m_net->getSessionInput(_m_session, "x");
    _m_output_tensor = _m_net->getSessionOutput(_m_session, "sigmoid_0.tmp_0");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch db_text detection model input node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_output_tensor == nullptr) {
        LOG(ERROR) << "Fetch db_text detection model loc output node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();
    _m_seg_prob_mat.create(_m_input_size_host, CV_8UC1);
    _m_seg_score_mat.create(_m_input_size_host, CV_32FC1);

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

    if (!cfg_content.contains("model_keep_top_k")) {
        _m_keep_topk = 250;
    } else {
        _m_keep_topk = cfg_content.at("model_keep_top_k").as_integer();
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "DB_Text detection model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
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
cv::Mat DBTextDetector<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat& input_image) const {
    // resize image
    cv::Mat tmp;
    cv::resize(input_image, tmp, _m_input_size_host);

    // normalize
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    tmp /= 255.0;
    cv::subtract(tmp, cv::Scalar(0.485, 0.456, 0.406), tmp);
    cv::divide(tmp, cv::Scalar(0.229, 0.224, 0.225), tmp);

    return tmp;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode DBTextDetector<INPUT, OUTPUT>::Impl::run(const INPUT& in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = dbtext_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    auto preprocessed_image = preprocess_image(internal_in.input_image);
    auto input_chw_image_data = CvUtils::convert_to_chw_vec(preprocessed_image);
    // run session
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, input_chw_image_data.data(), input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);
    // postprocess
    dbtext_impl::internal_output text_regions = postprocess();
    // transform internal output into external output
    out = dbtext_impl::transform_output<OUTPUT>(text_regions);
    
    return StatusCode::OK;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
void DBTextDetector<INPUT, OUTPUT>::Impl::decode_segmentation_result_mat() const {
    // convert tensor format
    MNN::Tensor output_tensor_user(_m_output_tensor, MNN::Tensor::DimensionType::CAFFE);
    _m_output_tensor->copyToHostTensor(&output_tensor_user);

    // construct segmentation prob map
    auto ele_size = output_tensor_user.elementSize();
    std::vector<uchar> seg_mat_vec;
    seg_mat_vec.resize(ele_size);

    for (int index = 0; index < ele_size; ++index) {
        if (output_tensor_user.host<float>()[index] >= _m_score_threshold) {
            seg_mat_vec[index] = static_cast<uchar>(output_tensor_user.host<float>()[index] * 255.0);
        } else {
            seg_mat_vec[index] = static_cast<uchar>(0);
            output_tensor_user.host<float>()[index] = 0.0;
        }
    }

    ::memcpy(_m_seg_prob_mat.data, seg_mat_vec.data(), seg_mat_vec.size() * sizeof(uchar));

    // construct segmentation score map
    ::memcpy(_m_seg_score_mat.data, output_tensor_user.host<float>(), ele_size * sizeof(float));
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
dbtext_impl::internal_output DBTextDetector<INPUT, OUTPUT>::Impl::postprocess() const {
    // decode seg prob mat
    decode_segmentation_result_mat();
    // get bboxes from bitmap
    auto bbox_result = get_boxes_from_bitmap();

    if (bbox_result.size() <= _m_keep_topk) {
        return bbox_result;
    } else {
        dbtext_impl::internal_output keep_top_k(bbox_result.cbegin(), bbox_result.cbegin() + _m_keep_topk);
        return keep_top_k;
    }
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
dbtext_impl::internal_output DBTextDetector<INPUT, OUTPUT>::Impl::get_boxes_from_bitmap() const {
    dbtext_impl::internal_output result;
    auto host_width = static_cast<float>(_m_input_size_host.width);
    auto host_height = static_cast<float>(_m_input_size_host.height);
    auto user_width = static_cast<float>(_m_input_size_user.width);
    auto user_height = static_cast<float>(_m_input_size_user.height);
    // contours analysis
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(_m_seg_prob_mat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        cv::RotatedRect r_bbox = cv::minAreaRect(contour);
        cv::Rect2f r_bounding_box = r_bbox.boundingRect2f();
        cv::Point2f r_vertices[4];
        r_bbox.points(r_vertices);
        auto sside = std::min(r_bbox.size.height, r_bbox.size.width);

        // thresh those with short sside
        if (sside < _m_sside_threshold) {
            continue;
        }

        // calculate rotated bbox score
        auto valid_roi = r_bounding_box & cv::Rect2f(0, 0, _m_seg_score_mat.cols, _m_seg_score_mat.rows);
        float score = static_cast<float>(cv::mean(_m_seg_score_mat(valid_roi))[0]);

        if (score < _m_score_threshold) {
            continue;
        }

        // rescale bbox coords to origin user image size
        for (auto& pt : r_vertices) {
            pt.x = pt.x * user_width / host_width;
            pt.y = pt.y * user_height / host_height;
        }

        r_bounding_box.x = r_bounding_box.x * user_width / host_width;
        r_bounding_box.y = r_bounding_box.y * user_height / host_height;
        r_bounding_box.width = r_bounding_box.width * user_width / host_width;
        r_bounding_box.height = r_bounding_box.height * user_height / host_height;

        text_region region;
        region.bbox = r_bounding_box;
        region.polygon = std::vector<cv::Point2f>(r_vertices, r_vertices + 4);
        region.score = score;

        result.push_back(region);
    }

    return result;
}


/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
DBTextDetector<INPUT, OUTPUT>::DBTextDetector() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
DBTextDetector<INPUT, OUTPUT>::~DBTextDetector() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode DBTextDetector<INPUT, OUTPUT>::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template<typename INPUT, typename OUTPUT>
bool DBTextDetector<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode DBTextDetector<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

}
}
}