/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: fast_sam_segmentor.inl
 * Date: 23-9-14
 ************************************************/

#include <cstring>
#include <type_traits>
#include <algorithm>

#include "glog/logging.h"

#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "common/time_stamp.h"
#include "models/mnn_helper.h"

namespace jinq {
namespace models {

using jinq::common::cv_utils;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace segment_anything {

namespace fast_sam_segmentor_impl {

using internal_input = cv::Mat;
using internal_output = jinq::models::io_define::segment_anything::std_fast_sam_output;

/***
 * 将用户自定义输入转换为模型内部输入
 */
template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<internal_input>::type>::value, internal_input>::type
transform_input(const INPUT& in) {
    return in;
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::mat_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    return in.input_image;
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::file_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    if (!FilePathUtil::is_file_exist(in.input_image_path)) {
        DLOG(WARNING) << "input image: " << in.input_image_path << " not exist";
        return internal_input{};
    }
    return cv::imread(in.input_image_path, cv::IMREAD_UNCHANGED);
}

template <typename INPUT>
typename std::enable_if<
    std::is_same<INPUT, std::decay<jinq::models::io_define::common_io::base64_input>::type>::value,
    internal_input>::type
transform_input(const INPUT& in) {
    return cv_utils::decode_base64_str_into_cvmat(in.input_image_content);
}

/***
 * 将模型内部输出转换为用户自定义输出
 */
template <typename OUTPUT>
typename std::enable_if<
    std::is_same<OUTPUT, std::decay<internal_output>::type>::value, internal_output>::type
transform_output(const internal_output& internal_out) {
    return internal_out;
}

struct _m_preds_bbox {
    cv::Rect2f bbox;
    float score = 0.0;
    std::vector<float> masks;
    int class_id = 0;
};

/***
 * FastSAM 分割模型内部实现
 */
class Impl {
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
     * @param cfg
     * @return
     */
    StatusCode init(const toml::table& cfg);

    /***
     *
     * @param input_image
     * @param everything_mask
     * @return
     */
    StatusCode everything(const cv::Mat& input_image, cv::Mat& everything_mask);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

  private:
    // model input/output names
    std::string _m_input_name;
    std::string _m_output_0_name;
    std::string _m_output_1_name;

    // MNN runtime (owns interpreter/session/tensors)
    jinq::models::MnnNet _m_net;

    // model output shape info
    std::vector<int> _m_output_0_shape;
    std::vector<int> _m_output_1_shape;

    // input image size
    cv::Size _m_input_image_size;
    // input tensor size
    cv::Size _m_input_tensor_size;
    // preds mask shape
    cv::Size _m_preds_mask_size;

    // conf threshold
    double _m_conf_thresh = 0.25;
    // nms iou threshold
    double _m_iou_thresh = 0.9;

    // init flag
    bool _m_successfully_init_model = false;

  private:
    /***
     *
     * @param input_image
     * @return
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @param mask
     * @return
     */
    cv::Mat upscale_mask_image(const cv::Mat& mask);

    /***
     *
     */
    StatusCode decode_all_masks(std::vector<cv::Mat>& preds_masks);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode Impl::init(const toml::table &cfg) {
    // init sam encoder configs
    const toml::table* cfg_content_ptr = cfg["FAST_SAM"].as_table();
    if (cfg_content_ptr == nullptr) {
        LOG(ERROR) << "Config section FAST_SAM missing or not a table";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }
    const toml::table& cfg_content = *cfg_content_ptr;
    // init session with named tensors
    _m_input_name = "images";
    _m_output_0_name = "output0";
    _m_output_1_name = "output1";
    auto init_status = _m_net.init(
        cfg_content, {_m_input_name}, {_m_output_0_name, _m_output_1_name});
    if (init_status != StatusCode::OK) {
        _m_successfully_init_model = false;
        return init_status;
    }

    // fetch input tensor
    auto* input_tensor = _m_net.input(_m_input_name);
    if (input_tensor->shape().size() != 4) {
        LOG(INFO) << "Invalid input tensor shape. Input tensor should be with [n, c, h, w] four dims but " << input_tensor->shape().size() << " dims instead";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size = cv::Size(input_tensor->shape()[3], input_tensor->shape()[2]);

    // fetch output tensor 0
    _m_output_0_shape = _m_net.output(_m_output_0_name)->shape();

    // fetch output tensor 1
    auto* output_tensor_1 = _m_net.output(_m_output_1_name);
    if (output_tensor_1->shape().size() != 4) {
        LOG(INFO) << "Invalid output tensor 1 shape. Output tensor 1 should be with [n, c, h, w] four dims but " << output_tensor_1->shape().size() << " dims instead";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_output_1_shape = output_tensor_1->shape();
    _m_preds_mask_size = cv::Size(_m_output_1_shape[3], _m_output_1_shape[2]);

    // init conf thresh and iou thresh
    _m_conf_thresh = cfg_content["conf_thresh"].value_or<double>(0.0);
    _m_iou_thresh = cfg_content["iou_thresh"].value_or<double>(0.0);

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully load fastsam model";
    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @param everything_mask
 * @return
 */
StatusCode Impl::everything(const cv::Mat& input_image, cv::Mat& everything_mask) {
    // check input image
    if (!input_image.data || input_image.empty()) {
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // preprocess image
    _m_input_image_size = input_image.size();
    auto preprocessed_image = preprocess_image(input_image);
    auto input_image_nchw_data = cv_utils::convert_to_chw_vec(preprocessed_image);

    // run session
    auto input_tensor_host = MNN::Tensor(_m_net.input(_m_input_name), MNN::Tensor::DimensionType::CAFFE);
    if (!cv_utils::copy_image_to_tensor(input_tensor_host.host<float>(), input_image_nchw_data, input_tensor_host.size())) {
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    _m_net.input(_m_input_name)->copyFromHostTensor(&input_tensor_host);

    _m_net.run_session();

    // decode all mask
    std::vector<cv::Mat> predicted_all_masks;
    auto status = decode_all_masks(predicted_all_masks);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "decode all masks failed, status code: " << status;
        return status;
    }
    if (predicted_all_masks.empty()) {
        LOG(WARNING) << "predicted mask counts: 0";
        return StatusCode::OK;
    }

    // reorder mask by area and generate everything mask
    auto comp_area = [](const cv::Mat& a, const cv::Mat& b) -> bool {
        auto a_area = cv::countNonZero(a);
        auto b_area = cv::countNonZero(b);
        return a_area >= b_area;
    };
    std::sort(predicted_all_masks.begin(), predicted_all_masks.end(), comp_area);
    everything_mask = cv::Mat::zeros(_m_input_image_size, CV_32SC1);
    for (auto idx = 0; idx < predicted_all_masks.size(); ++idx) {
        auto obj_id = idx + 1;
        auto mask = predicted_all_masks[idx];
        everything_mask.setTo(obj_id, mask);
    }

    return StatusCode::OK;
}

/***
 *
 * @param input_image
 * @return
 */
cv::Mat Impl::preprocess_image(const cv::Mat &input_image) const {

    auto input_node_h = _m_input_tensor_size.height;
    auto input_node_w = _m_input_tensor_size.width;
    auto ori_img_width = static_cast<float>(_m_input_image_size.width);
    auto ori_img_height = static_cast<float>(_m_input_image_size.height);
    auto long_side = std::max(_m_input_image_size.width, _m_input_image_size.height);
    float scale = static_cast<float>(input_node_h) / static_cast<float>(long_side);
    cv::Size target_size = cv::Size(
        static_cast<int>(scale * ori_img_width), static_cast<int>(scale * ori_img_height));

    cv::Mat result;
    cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);
    cv::resize(result, result,target_size);
    result.convertTo(result, CV_32FC3);
    cv::divide(result, 255.0, result);

    // pad image
    auto pad_h = input_node_h - target_size.height;
    auto pad_w = input_node_w - target_size.width;
    cv::copyMakeBorder(result, result, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, 0.0);

    return result;
}

/***
 *
 * @param mask
 * @return
 */
cv::Mat Impl::upscale_mask_image(const cv::Mat &mask) {
    auto input_node_h = _m_preds_mask_size.height;
    auto input_node_w = _m_preds_mask_size.width;
    auto ori_img_width = static_cast<float>(_m_input_image_size.width);
    auto ori_img_height = static_cast<float>(_m_input_image_size.height);
    auto long_side = std::max(_m_input_image_size.width, _m_input_image_size.height);
    float scale = static_cast<float>(input_node_h) / static_cast<float>(long_side);
    cv::Size target_size = cv::Size(
        static_cast<int>(scale * ori_img_width), static_cast<int>(scale * ori_img_height));
    auto pad_h = input_node_h - target_size.height;
    auto pad_w = input_node_w - target_size.width;

    cv::Mat result_mask;
    cv::Rect src_mask_roi = cv::Rect(0, 0, mask.cols - pad_w, mask.rows - pad_h) &
                            cv::Rect(0, 0, mask.cols, mask.rows);
    mask(src_mask_roi).copyTo(result_mask);

    cv::resize(result_mask, result_mask, _m_input_image_size, 0.0, 0.0, cv::INTER_LINEAR);
    return result_mask;
}

/***
 *
 * @param preds_masks
 * @param merged_mask
 * @return
 */
StatusCode Impl::decode_all_masks(std::vector<cv::Mat>& preds_masks) {
    // decode output preds info
    auto output_tensor_0_host = MNN::Tensor(_m_net.output(_m_output_0_name), _m_net.output(_m_output_0_name)->getDimensionType());
    _m_net.output(_m_output_0_name)->copyToHostTensor(&output_tensor_0_host);
    auto* output_tensor_0_data = output_tensor_0_host.host<float>();
    if (output_tensor_0_data == nullptr) {
        LOG(ERROR) << "fetch output tensor 0 inference result failed, output tensor 0's data is nullptr";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    auto bbox_info_len = _m_output_0_shape[1];
    auto bbox_nums = _m_output_0_shape[2];
    std::vector<std::vector<float> > total_preds;
    total_preds.resize(bbox_nums);
    for (auto& bbox : total_preds) {
        bbox.resize(bbox_info_len);
    }
    for (auto idx_0 = 0; idx_0 < bbox_info_len; ++idx_0) {
        for (auto idx_1 = 0; idx_1 < bbox_nums; ++idx_1) {
            auto data_idx = idx_0 * bbox_nums + idx_1;
            total_preds[idx_1][idx_0] = output_tensor_0_data[data_idx];
        }
    }

    std::vector<_m_preds_bbox> threshed_preds;
    for (auto& bbox : total_preds) {
        auto conf = bbox[4];
        if (conf > _m_conf_thresh) {
            _m_preds_bbox b;
            b.score = bbox[4];
            b.masks = {bbox.begin() + 5, bbox.end()};
            auto cx = bbox[0];
            auto cy = bbox[1];
            auto width = bbox[2];
            auto height = bbox[3];
            auto x = cx - width / 2.0f;
            if (x < 0.0) {
                x = 0.0f;
            }
            auto y = cy - height / 2.0f;
            if (y < 0.0) {
                y = 0.0f;
            }
            b.bbox = cv::Rect2f(x, y, width, height);
            threshed_preds.push_back(b);
        }
    }

    auto nms_result = cv_utils::nms_bboxes(threshed_preds, _m_iou_thresh);
    auto c = _m_output_1_shape[1];
    auto mh = _m_preds_mask_size.height;
    auto mw = _m_preds_mask_size.width;

    auto output_tensor_1_host = MNN::Tensor(_m_net.output(_m_output_1_name), _m_net.output(_m_output_1_name)->getDimensionType());
    _m_net.output(_m_output_1_name)->copyToHostTensor(&output_tensor_1_host);
    auto* output_tensor_1_data = output_tensor_1_host.host<float>();
    if (output_tensor_1_data == nullptr) {
        LOG(ERROR) << "fetch output tensor 1 inference result failed, output tensor 1's data is nullptr";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    std::vector<float> output_tensor_1_data_vec(output_tensor_1_data, output_tensor_1_data + output_tensor_1_host.elementSize());
    auto mask_proto_hwc = cv_utils::convert_to_hwc_vec(output_tensor_1_data_vec, 1, c, mh * mw);
    cv::Mat mask_proto(cv::Size(mh * mw, c), CV_32FC1, mask_proto_hwc.data());

    float downscale_h = static_cast<float>(mh) / static_cast<float>(_m_input_tensor_size.height);
    float downscale_w = static_cast<float>(mw) / static_cast<float>(_m_input_tensor_size.width);
    for (auto& bbox : nms_result) {
        // decode mask
        cv::Mat mask_in(cv::Size(c, 1), CV_32FC1, bbox.masks.data());
        cv::Mat mask_output = mask_in * mask_proto;
        mask_output = mask_output.reshape(1, {mw, mh});
        cv::Mat tmp_exp(mask_output.size(), CV_32FC1);
        cv::exp(-mask_output, tmp_exp);
        cv::Mat sigmoid_output = cv::Mat::zeros(mask_output.size(), CV_32FC1);
        sigmoid_output = 1.0f / (1.0f + tmp_exp);

        // crop mask
        for (auto row = 0; row < sigmoid_output.rows; ++row) {
            auto row_data = sigmoid_output.ptr<float>(row);
            for (auto col = 0; col < sigmoid_output.cols; ++col) {
                // downscale preds bounding box
                auto scaled_bbox_tlx = static_cast<int>(bbox.bbox.x * downscale_w);
                auto scaled_bbox_tly = static_cast<int>(bbox.bbox.y * downscale_h);
                auto scaled_bbox_rbx = scaled_bbox_tlx + static_cast<int>(bbox.bbox.width * downscale_w);
                auto scaled_bbox_rby = scaled_bbox_tly + static_cast<int>(bbox.bbox.height * downscale_h);
                // crop mask via bounding box
                if (row > scaled_bbox_tly && row < scaled_bbox_rby && col > scaled_bbox_tlx && col < scaled_bbox_rbx) {
                    continue;
                } else {
                    row_data[col] = 0.0f;
                }
            }
        }

        // thresh mask
        auto upscaled_sigmoid_output = upscale_mask_image(sigmoid_output);
        cv::Mat mask = cv::Mat::zeros(upscaled_sigmoid_output.size(), CV_8UC1);
        for (auto row = 0; row < upscaled_sigmoid_output.rows; ++row) {
            auto row_data = mask.ptr(row);
            for (auto col = 0; col < upscaled_sigmoid_output.cols; ++col) {
                if (upscaled_sigmoid_output.at<float>(row, col) >= 0.5) {
                    row_data[col] = 255;
                }
            }
        }
        preds_masks.push_back(mask);
    }

    return StatusCode::OK;
}

} // namespace fast_sam_segmentor_impl

/************ Template Implementation ************/

template <typename INPUT, typename OUTPUT>
FastSamSegmentor<INPUT, OUTPUT>::FastSamSegmentor() {
    _m_pimpl = std::make_unique<fast_sam_segmentor_impl::Impl>();
}

template <typename INPUT, typename OUTPUT>
FastSamSegmentor<INPUT, OUTPUT>::~FastSamSegmentor() = default;

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode FastSamSegmentor<INPUT, OUTPUT>::init(const toml::table &cfg) {
    return _m_pimpl->init(cfg);
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode FastSamSegmentor<INPUT, OUTPUT>::run(const INPUT& input, OUTPUT& output) {
    auto internal_input = fast_sam_segmentor_impl::transform_input(input);
    fast_sam_segmentor_impl::internal_output internal_output;
    auto status = _m_pimpl->everything(internal_input, internal_output);
    output = fast_sam_segmentor_impl::transform_output<OUTPUT>(internal_output);
    return status;
}

template <typename INPUT, typename OUTPUT>
jinq::common::StatusCode FastSamSegmentor<INPUT, OUTPUT>::everything(
    const cv::Mat &input_image, cv::Mat &everything_mask) {
    return _m_pimpl->everything(input_image, everything_mask);
}

template <typename INPUT, typename OUTPUT>
bool FastSamSegmentor<INPUT, OUTPUT>::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

} // namespace segment_anything
} // namespace models
} // namespace jinq
