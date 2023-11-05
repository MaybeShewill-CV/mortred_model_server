/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: lightglue.inl
* Date: 23-11-03
************************************************/

#include "lightglue.h"

#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
// #include <cuda_provider_factory.h>

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"

namespace jinq {
namespace models {

using jinq::common::CvUtils;
using jinq::common::Base64;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::mat_input;

namespace feature_point {
using jinq::models::io_define::feature_point::fp;
using jinq::models::io_define::feature_point::std_feature_point_output;

namespace lightglue_impl {

struct internal_input {
    cv::Mat input_image;
};
using internal_output = std_feature_point_output;

/***
 *
 * @tparam INPUT
 * @param in
 * @return
 */
template <typename INPUT>
typename std::enable_if<std::is_same<INPUT, std::decay<file_input>::type>::value, internal_input>::type transform_input(const INPUT &in) {
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
typename std::enable_if<std::is_same<INPUT, std::decay<mat_input>::type>::value, internal_input>::type transform_input(const INPUT &in) {
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
typename std::enable_if<std::is_same<INPUT, std::decay<base64_input>::type>::value, internal_input>::type transform_input(const INPUT &in) {
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
typename std::enable_if<std::is_same<OUTPUT, std::decay<std_feature_point_output>::type>::value, std_feature_point_output>::type
transform_output(const LightGlue_impl::internal_output &internal_out) {
    std_feature_point_output result;
    for (auto& value : internal_out) {
        result.push_back(value);
    }
    return result;
}

} // namespace lightglue_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT> 
class LightGlue<INPUT, OUTPUT>::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        // todo(luoyao@baidu.com) implement deconstruct function
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
    StatusCode run(const INPUT &in, OUTPUT& out);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const { return _m_successfully_initialized; };

private:
    struct ONNXParams {
        std::string model_file_path;
        // ort env params
        int thread_nums = 1;
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::Session session;
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<Ort::Value> output_tensors;
        // input/output node info
        std::vector<char*> input_node_names;
        std::vector<std::vector<int64_t>> input_node_shapes;
        std::vector<char*> output_node_names;
        std::vector<std::vector<int64_t>> output_node_shapes;
        // feature point scales
        std::vector<float> scales = {1.0f , 1.0f};
    };

    enum BackendType {
        ONNX = 0,
        TRT = 1,
    };

  private:
    // model backend type
    BackendType _m_backend_type = ONNX;

    // user input size
    cv::Size _m_input_size_user = cv::Size();
    // input node size
    cv::Size _m_input_size_host = cv::Size();
    // match thresh value
    float _m_match_thresh = 0.0f;

    // flag
    bool _m_successfully_initialized = false;

  private:
    /***
     * preprocess
     * @param input_image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image);

    /***
     *
     * @param config
     * @return
     */
    StatusCode init_onnx(const toml::value& config);

    /***
     *
     * @param in
     * @param out
     * @return
     */
    StatusCode onnx_run(const INPUT& in, OUTPUT& out);

    /***
     *
     * @return
     */
    lightglue_impl::internal_output onnx_decode_output();
};

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode LightGlue<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    // choose backend type
    auto backend_dict = config.at("BACKEND_DICT");
    auto backend_name = config.at("LIGHTGLUE").at("backend_type").as_string();
    _m_backend_type = static_cast<BackendType>(backend_dict[backend_name].as_integer());

    // init metric3d configs
    toml::value metric3d_cfg;
    if (_m_backend_type == ONNX) {
        metric3d_cfg = config.at("LIGHTGLUE_ONNX");
    } else {
        metric3d_cfg = config.at("LIGHTGLUE_TRT");
    }
    auto model_file_name = FilePathUtil::get_file_name(metric3d_cfg.at("model_file_path").as_string());

    StatusCode init_status;
    if (_m_backend_type == ONNX) {
        init_status = init_onnx(metric3d_cfg);
    } else {
        // todo(luoyao@baidu.com) init trt func
        // init_status = init_trt(metric3d_cfg);
    }

    if (init_status == StatusCode::OK) {
        _m_successfully_initialized = true;
        LOG(INFO) << "Successfully load lightglue model from: " << model_file_name;
    } else {
        _m_successfully_initialized = false;
        LOG(INFO) << "Failed load lightglue model from: " << model_file_name;
    }

    return init_status;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode LightGlue<INPUT, OUTPUT>::Impl::run(const INPUT &in, OUTPUT& out) {
    StatusCode infer_status;
    if (_m_backend_type == MNN) {
        infer_status = onnx_run(in, out);
    } else {
        // todo(luoyao@baidu.com) implement trt run func
        // infer_status = trt_run(in, out);
    }
    return infer_status;
}

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT> 
cv::Mat LightGlue<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat &input_image) const {
    // resize image
    cv::Mat tmp;
    if (input_image.size() != _m_input_size_host) {
        cv::resize(input_image, tmp, _m_input_size_host);
    } else {
        tmp = input_image;
    }
    // convert bgr to gray
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);

    // normalize
    if (tmp.type() != CV_32FC1) {
        tmp.convertTo(tmp, CV_32FC1);
    }
    tmp /= 255.0;
    return tmp;
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param config
 * @return
 */
template <typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::Impl::init_onnx(const toml::value& config) {
    

    return StatusCode::OK;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::Impl::onnx_run(const INPUT &in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = LightGlue_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);

    // todo(luoyao@baidu.com) implement onnx run func

    return StatusCode::OK;
}



/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> LightGlue<INPUT, OUTPUT>::LightGlue() { 
    _m_pimpl = std::make_unique<Impl>(); 
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> LightGlue<INPUT, OUTPUT>::~LightGlue() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT> StatusCode LightGlue<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT> bool LightGlue<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode LightGlue<INPUT, OUTPUT>::run(const INPUT &input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace feature_point
} // namespace models
} // namespace jinq