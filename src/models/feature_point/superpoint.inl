/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: superpoint.inl
* Date: 22-6-15
************************************************/

#include "superpoint.h"

#include "MNN/Interpreter.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>

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

namespace superpoint_impl {

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
transform_output(const superpoint_impl::internal_output &internal_out) {
    std_feature_point_output result;
    for (auto& value : internal_out) {
        result.push_back(value);
    }
    return result;
}

} // namespace superpoint_impl

/***************** Impl Function Sets ******************/

template <typename INPUT, typename OUTPUT> 
class SuperPoint<INPUT, OUTPUT>::Impl {
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
    StatusCode run(const INPUT &in, OUTPUT& out);

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
    // MNN Output tensor node semi
    MNN::Tensor *_m_output_tensor_semi = nullptr;
    // MNN Output tensor node coarse
    MNN::Tensor *_m_output_tensor_coarse = nullptr;
    // MNN后端使用线程数
    int _m_threads_nums = 4;
    // 得分阈值
    double _m_score_threshold = 0.015;
    // nms阈值
    double _m_nms_threshold = 4.0;
    // cell size
    int _m_cell_size = 8;

    // 用户输入网络的图像尺寸
    cv::Size _m_input_size_user = cv::Size();
    //　计算图定义的输入node尺寸
    cv::Size _m_input_size_host = cv::Size();
    // init color map
    std::map<int, cv::Scalar> _m_color_map;
    // 是否成功初始化标志位
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @param input_image
     * @return
     */
    cv::Mat preprocess_image(const cv::Mat &input_image) const;

    /***
     *
     * @param key_points
     * @return
     */
    void decode_fp_location_and_score(superpoint_impl::internal_output& key_points) const;

    /***
     *
     * @param key_points
     * @return
     */
    void decode_fp_descriptor(superpoint_impl::internal_output& key_points) const;
};

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT> 
StatusCode SuperPoint<INPUT, OUTPUT>::Impl::init(const decltype(toml::parse("")) &config) {
    if (!config.contains("SUPERPOINT")) {
        LOG(ERROR) << "Config file missing SUPERPOINT section please check config file";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    toml::value cfg_content = config.at("SUPERPOINT");

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
        LOG(ERROR) << "superpoint model file: " << _m_model_file_path << " not exist";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_m_model_file_path.c_str()));
    if (nullptr == _m_net) {
        LOG(ERROR) << "Create superpoint model interpreter failed";
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
        LOG(ERROR) << "Create superpoint model session failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_tensor = _m_net->getSessionInput(_m_session, "input");
    _m_output_tensor_semi = _m_net->getSessionOutput(_m_session, "output_1");
    _m_output_tensor_coarse = _m_net->getSessionOutput(_m_session, "output_2");

    if (_m_input_tensor == nullptr) {
        LOG(ERROR) << "Fetch superpoint model input node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_output_tensor_semi == nullptr || _m_output_tensor_coarse == nullptr) {
        LOG(ERROR) << "Fetch superpoint model output node failed";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_input_size_host.width = _m_input_tensor->width();
    _m_input_size_host.height = _m_input_tensor->height();

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

    _m_successfully_initialized = true;
    LOG(INFO) << "Superpoint feature extractor model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";

    return StatusCode::OK;
}

/***
 *
 * @param in
 * @param out
 * @return
 */
template<typename INPUT, typename OUTPUT>
StatusCode SuperPoint<INPUT, OUTPUT>::Impl::run(const INPUT &in, OUTPUT& out) {
    // transform external input into internal input
    auto internal_in = superpoint_impl::transform_input(in);
    if (!internal_in.input_image.data || internal_in.input_image.empty()) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    // preprocess image
    _m_input_size_user = internal_in.input_image.size();
    cv::Mat preprocessed_image = preprocess_image(internal_in.input_image);

    // run session
    MNN::Tensor input_tensor_user(_m_input_tensor, MNN::Tensor::DimensionType::CAFFE);
    auto input_tensor_data = input_tensor_user.host<float>();
    auto input_tensor_size = input_tensor_user.size();
    ::memcpy(input_tensor_data, preprocessed_image.data, input_tensor_size);
    _m_input_tensor->copyFromHostTensor(&input_tensor_user);
    _m_net->runSession(_m_session);

    // decode feture point locations and scores
    superpoint_impl::internal_output internal_out;
    decode_fp_location_and_score(internal_out);

    // decode feture point descriptor
    decode_fp_descriptor(internal_out);

    // rescale feature point locations
    float h_scale = static_cast<float>(_m_input_size_user.height) / _m_input_size_host.height;
    float w_scale = static_cast<float>(_m_input_size_user.width) / _m_input_size_host.width;
    for (auto& pt : internal_out) {
        pt.location.x *= w_scale;
        pt.location.y *= h_scale;
    }

    // transform result
    out = superpoint_impl::transform_output<OUTPUT>(internal_out);

    return StatusCode::OK;
}

/***
 *
 * @param cfg_file_path
 * @return
 */
template <typename INPUT, typename OUTPUT> 
cv::Mat SuperPoint<INPUT, OUTPUT>::Impl::preprocess_image(const cv::Mat &input_image) const {
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
 * @return
 */
template <typename INPUT, typename OUTPUT> 
void SuperPoint<INPUT, OUTPUT>::Impl::decode_fp_location_and_score(superpoint_impl::internal_output& key_points) const {
    MNN::Tensor output_tensor_semi_user(_m_output_tensor_semi, MNN::Tensor::DimensionType::CAFFE);
    _m_output_tensor_semi->copyToHostTensor(&output_tensor_semi_user);
    auto host_data = output_tensor_semi_user.host<float>();
    // reshape semi vec
    const int dense_map_row = _m_input_size_host.height / _m_cell_size;
    const int dense_map_col = _m_input_size_host.width / _m_cell_size;
    const int dense_map_channels = 65;

    std::vector<float> semi_tdata_reshape(output_tensor_semi_user.elementSize());
    for (auto row = 0; row < dense_map_row; ++row) {
        for (auto col = 0; col < dense_map_col; ++col) {
            for (auto channel = 0; channel < dense_map_channels; ++channel) {
                auto to_index = row * dense_map_col * dense_map_channels + col * dense_map_channels + channel;
                auto from_index = channel * dense_map_row * dense_map_col + row * dense_map_col + col;
                semi_tdata_reshape[to_index] = host_data[from_index];
            }
        }
    }
    cv::Mat dense(dense_map_row, dense_map_col, CV_32FC(dense_map_channels), semi_tdata_reshape.data());
    // softmax
    std::vector<cv::Mat> dense_split;
    cv::split(dense, dense_split);
    cv::Mat dense_channel_sum = cv::Mat::zeros(dense_map_row, dense_map_col, CV_32FC1);

    for (auto& split : dense_split) {
        cv::exp(split, split);
        dense_channel_sum += split;
    }
    for (auto& split : dense_split) {
        cv::divide(split, dense_channel_sum, split);
    }
    cv::Mat dense_softmax;
    cv::merge(std::vector<cv::Mat>(dense_split.begin(), dense_split.end() - 1), dense_softmax);
    // select interest point
    for (auto row = 0; row < dense_map_row; row++) {
        for (auto col = 0; col < dense_map_col; col++) {
            for (int row_ext_index = 0; row_ext_index < _m_cell_size; ++row_ext_index) {
                for (int col_ext_index = 0; col_ext_index < _m_cell_size; ++col_ext_index) {
                    int score_idx = row_ext_index * _m_cell_size + col_ext_index;
                    float score = dense_softmax.at<cv::Vec<float, dense_map_channels - 1> >(row, col)[score_idx];
                    int interest_pt_x = col * _m_cell_size + col_ext_index;
                    int interest_pt_y = row * _m_cell_size + row_ext_index;
                    cv::Point2f interest_pt(static_cast<float>(interest_pt_x), static_cast<float>(interest_pt_y));
                    if (score >= _m_score_threshold) {
                        fp key_pt;
                        key_pt.location = interest_pt;
                        key_pt.score = score;
                        key_points.push_back(key_pt);
                    }
                }
            }
        }
    }
    // nms interest point
    std::sort(key_points.begin(), key_points.end(), [](const fp& pt1, const fp& pt2) {return pt1.score >= pt2.score;});
    auto iter = key_points.begin();
    while (iter != key_points.end()) {
        auto comp = iter + 1;
        while (comp != key_points.end()) {
            auto diff_x = iter->location.x - comp->location.x;
            auto diff_y = iter->location.y - comp->location.y;
            auto distance = std::sqrt(std::pow(diff_x, 2) + std::pow(diff_y, 2));

            if (distance <= _m_nms_threshold) {
                comp = key_points.erase(comp);
            }
            else {
                comp++;
            }
        }
        iter++;
    }
}

/***
 * @return
 */
template <typename INPUT, typename OUTPUT> 
void SuperPoint<INPUT, OUTPUT>::Impl::decode_fp_descriptor(superpoint_impl::internal_output& key_points) const {
    MNN::Tensor output_tensor_desc_user(_m_output_tensor_coarse, MNN::Tensor::DimensionType::CAFFE);
    _m_output_tensor_coarse->copyToHostTensor(&output_tensor_desc_user);
    auto host_data = output_tensor_desc_user.host<float>();
    // reshape desc vec
    const int desc_map_row = _m_input_size_host.height / _m_cell_size;
    const int desc_map_col = _m_input_size_host.width / _m_cell_size;
    const int desc_map_channels = 256;
    std::vector<float> desc_tdata_reshape(output_tensor_desc_user.elementSize());
    for (auto row = 0; row < desc_map_row; ++row) {
        for (auto col = 0; col < desc_map_col; ++col) {
            for (auto channel = 0; channel < desc_map_channels; ++channel) {
                auto from_index = channel * desc_map_row * desc_map_col + row * desc_map_col + col;
                auto to_index = row * desc_map_col * desc_map_channels + col * desc_map_channels + channel;
                desc_tdata_reshape[to_index] = host_data[from_index];
            }
        }
    }
    cv::Mat desc(desc_map_row, desc_map_col, CV_32FC(desc_map_channels), desc_tdata_reshape.data());

    // grid sample descriptor
    for (auto& key_pt : key_points) {
        float x = static_cast<float>(key_pt.location.x) / static_cast<float>(_m_cell_size);
        float y = static_cast<float>(key_pt.location.y) / static_cast<float>(_m_cell_size);
        float x1 = std::floor(x);
        float x2 = std::ceil(x);
        float y1 = std::floor(y);
        float y2 = std::ceil(y);

        auto f_q11 = desc.at<cv::Vec<float, 256>>(static_cast<int>(y1), static_cast<int>(x1));
        auto f_q21 = desc.at<cv::Vec<float, 256>>(static_cast<int>(y1), static_cast<int>(x2));
        auto f_q12 = desc.at<cv::Vec<float, 256>>(static_cast<int>(y2), static_cast<int>(x1));
        auto f_q22 = desc.at<cv::Vec<float, 256>>(static_cast<int>(y2), static_cast<int>(x2));

        cv::Vec<float, 256> f_r1;
        cv::Vec<float, 256> f_r2;
        cv::Vec<float, 256> f_p;

        if (std::abs(x2 - x1) < 0.0000000001) {
            f_r1 = f_q11;
            f_r2 = f_q11;
        } else {
            f_r1 = (x2 - x) / (x2 - x1) * f_q11 + (x - x1) / (x2 - x1) * f_q21;
            f_r2 = (x2 - x) / (x2 - x1) * f_q12 + (x - x1) / (x2 - x1) * f_q22;
        }

        if (std::abs(y2 - y1) < 0.00000000001) {
            f_p = f_r1;
        } else {
            f_p = (y2 - y) / (y2 - y1) * f_r1 + (y - y1) / (y2 - y1) * f_r2;
        }

        std::vector<float> sample_descriptor(f_p.val, f_p.val + 256);
        float vec_norm = 0.0;

        for (const auto v : sample_descriptor) {
            vec_norm += static_cast<float>(std::pow(v, 2));
        }
        vec_norm = std::sqrt(vec_norm);
        for (auto& v : sample_descriptor) {
            v /= vec_norm;
        }
        key_pt.descriptor = sample_descriptor;
    }
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> SuperPoint<INPUT, OUTPUT>::SuperPoint() { 
    _m_pimpl = std::make_unique<Impl>(); 
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT> SuperPoint<INPUT, OUTPUT>::~SuperPoint() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
template <typename INPUT, typename OUTPUT> StatusCode SuperPoint<INPUT, OUTPUT>::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @return
 */
template <typename INPUT, typename OUTPUT> bool SuperPoint<INPUT, OUTPUT>::is_successfully_initialized() const {
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
StatusCode SuperPoint<INPUT, OUTPUT>::run(const INPUT &input, OUTPUT& output) {
    return _m_pimpl->run(input, output);
}

} // namespace feature_point
} // namespace models
} // namespace jinq