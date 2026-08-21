/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: sam_prompt_decoder.cpp
 * Date: 23-6-7
 ************************************************/

#include "sam_prompt_decoder.h"

#include <algorithm>
#include <cstring>
#include <map>

#include "glog/logging.h"

#include "models/backend/tensor.h"

namespace jinq {
namespace models {
namespace segment_anything {

using jinq::common::StatusCode;
using jinq::models::backend::InferenceSession;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorInfo;

class SamPromptDecoder::Impl {
  public:
    explicit Impl(std::unique_ptr<InferenceSession> session)
        : _m_session(std::move(session)) {}

    ~Impl() = default;

    StatusCode init() {
        if (_m_session == nullptr) {
            LOG(ERROR) << "sam decoder session is null";
            return StatusCode::MODEL_INIT_FAILED;
        }

        const std::vector<std::string> required_inputs = {
            "image_embeddings", "point_coords", "point_labels", "mask_input",
            "has_mask_input"};
        for (const auto& name : required_inputs) {
            const auto* info = find_info(name);
            if (info == nullptr || info->dtype != jinq::models::backend::DType::F32) {
                LOG(ERROR) << "sam decoder input missing or invalid: " << name;
                return StatusCode::MODEL_INIT_FAILED;
            }
            if (name != "point_coords" && name != "point_labels" && info->dynamic) {
                LOG(ERROR) << "sam decoder input '" << name << "' must be static";
                return StatusCode::MODEL_INIT_FAILED;
            }
        }

        _m_has_orig_size_input = find_info("orig_im_size") != nullptr;
        if (_m_has_orig_size_input) {
            const auto* info = find_info("orig_im_size");
            if (info->dtype != jinq::models::backend::DType::F32 ||
                info->shape.size() != 1 ||
                jinq::models::backend::shape_volume(info->shape) != 2) {
                LOG(ERROR) << "invalid sam decoder orig_im_size input: " << info->to_string();
                return StatusCode::MODEL_INIT_FAILED;
            }
        }

        const auto* iou_info = find_output("iou_predictions");
        const bool has_full_masks = find_output("masks") != nullptr;
        const bool has_low_res_masks = find_output("low_res_masks") != nullptr;
        if (iou_info == nullptr || (!has_full_masks && !has_low_res_masks)) {
            LOG(ERROR) << "sam decoder must expose iou_predictions and masks/low_res_masks";
            return StatusCode::MODEL_INIT_FAILED;
        }
        for (const auto& item : _m_session->outputs()) {
            if (item.dtype != jinq::models::backend::DType::F32) {
                LOG(ERROR) << "invalid sam decoder output: " << item.to_string();
                return StatusCode::MODEL_INIT_FAILED;
            }
        }

        _m_successfully_initialized = true;
        return StatusCode::OK;
    }

    void set_ori_image_size(const cv::Size& ori_image_size) {
        _m_ori_image_size = ori_image_size;
    }

    void set_encoder_input_size(const cv::Size& input_node_size) {
        _m_encoder_input_size = input_node_size;
    }

    StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<cv::Rect2f>& bboxes,
        std::vector<cv::Mat>& predicted_masks) {
        predicted_masks.clear();
        for (const auto& bbox : bboxes) {
            std::vector<float> points = {
                bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height, 0.0f, 0.0f};
            std::vector<float> labels = {2.0f, 3.0f, -1.0f};
            cv::Mat mask;
            const auto status = get_mask(image_embeddings, points, labels, mask);
            if (status != StatusCode::OK) {
                return status;
            }
            predicted_masks.push_back(std::move(mask));
        }
        return StatusCode::OK;
    }

    StatusCode decode(
        const std::vector<float>& image_embeddings,
        const std::vector<std::vector<cv::Point2f>>& points,
        std::vector<cv::Mat>& predicted_masks) {
        predicted_masks.clear();
        for (const auto& prompt_points : points) {
            std::vector<float> flat_points;
            std::vector<float> labels;
            flat_points.reserve(prompt_points.size() * 2 + 2);
            labels.reserve(prompt_points.size() + 1);
            for (const auto& point : prompt_points) {
                flat_points.push_back(point.x);
                flat_points.push_back(point.y);
                labels.push_back(1.0f);
            }
            flat_points.push_back(0.0f);
            flat_points.push_back(0.0f);
            labels.push_back(-1.0f);

            cv::Mat mask;
            const auto status = get_mask(image_embeddings, flat_points, labels, mask);
            if (status != StatusCode::OK) {
                return status;
            }
            predicted_masks.push_back(std::move(mask));
        }
        return StatusCode::OK;
    }

    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    }

  private:
    const TensorInfo* find_info(const std::string& name) const {
        const auto iter = std::find_if(
            _m_session->inputs().begin(), _m_session->inputs().end(),
            [&name](const TensorInfo& info) { return info.name == name; });
        return iter == _m_session->inputs().end() ? nullptr : &*iter;
    }

    const NamedTensor* find_output(
        const std::vector<NamedTensor>& outputs, const std::string& name) const {
        const auto iter = std::find_if(
            outputs.begin(), outputs.end(),
            [&name](const NamedTensor& item) { return item.name == name; });
        return iter == outputs.end() ? nullptr : &*iter;
    }

    const TensorInfo* find_output(const std::string& name) const {
        const auto iter = std::find_if(
            _m_session->outputs().begin(), _m_session->outputs().end(),
            [&name](const TensorInfo& info) { return info.name == name; });
        return iter == _m_session->outputs().end() ? nullptr : &*iter;
    }

    static bool fill_input(
        std::vector<NamedTensor>& inputs, const std::string& name,
        const TensorInfo& info, const std::vector<float>& values,
        const std::vector<int64_t>& shape) {
        NamedTensor named;
        named.name = name;
        named.tensor = Tensor::make<float>(shape);
        if (values.size() != static_cast<size_t>(named.tensor.element_count())) {
            LOG(ERROR) << "sam decoder input '" << name << "' element count "
                       << values.size() << " mismatches tensor "
                       << named.tensor.element_count();
            return false;
        }
        std::memcpy(named.tensor.buffer.data(), values.data(), named.tensor.byte_size());
        inputs.push_back(std::move(named));
        return true;
    }

    StatusCode get_mask(
        const std::vector<float>& image_embeddings,
        const std::vector<float>& points,
        const std::vector<float>& labels,
        cv::Mat& out_mask) {
        if (!_m_successfully_initialized) {
            return StatusCode::MODEL_INIT_FAILED;
        }
        if (points.empty() || points.size() != labels.size() * 2) {
            LOG(ERROR) << "invalid sam decoder prompt points/labels";
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }

        std::vector<NamedTensor> inputs;
        const auto* embedding_info = find_info("image_embeddings");
        if (!fill_input(
                inputs, "image_embeddings", *embedding_info, image_embeddings,
                embedding_info->shape)) {
            return StatusCode::MODEL_EMPTY_OUTPUT;
        }

        const auto* point_coords_info = find_info("point_coords");
        const auto* point_labels_info = find_info("point_labels");
        auto padded_points = points;
        auto padded_labels = labels;
        if (!point_coords_info->dynamic && !point_labels_info->dynamic) {
            const auto expected_count = point_coords_info->shape[1];
            while (static_cast<int64_t>(padded_labels.size()) < expected_count) {
                padded_points.push_back(0.0f);
                padded_points.push_back(0.0f);
                padded_labels.push_back(-1.0f);
            }
        }
        if (!fill_input(
                inputs, "point_coords", *point_coords_info, padded_points,
                {1, static_cast<int64_t>(padded_labels.size()), 2}) ||
            !fill_input(
                inputs, "point_labels", *point_labels_info, padded_labels,
                {1, static_cast<int64_t>(padded_labels.size())})) {
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }

        const auto* mask_info = find_info("mask_input");
        if (!fill_input(
                inputs, "mask_input", *mask_info,
                std::vector<float>(
                    static_cast<size_t>(jinq::models::backend::shape_volume(mask_info->shape))),
                mask_info->shape)) {
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        const auto* has_mask_info = find_info("has_mask_input");
        if (!fill_input(
                inputs, "has_mask_input", *has_mask_info,
                std::vector<float>(
                    static_cast<size_t>(
                        jinq::models::backend::shape_volume(has_mask_info->shape))),
                has_mask_info->shape)) {
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        if (_m_has_orig_size_input) {
            const auto* size_info = find_info("orig_im_size");
            if (!fill_input(
                    inputs, "orig_im_size", *size_info,
                    {static_cast<float>(_m_ori_image_size.height),
                     static_cast<float>(_m_ori_image_size.width)},
                    size_info->shape)) {
                return StatusCode::MODEL_RUN_SESSION_FAILED;
            }
        }

        std::vector<NamedTensor> outputs;
        const auto run_status = _m_session->run(inputs, outputs);
        if (run_status != StatusCode::OK) {
            return run_status;
        }

        const auto* iou = find_output(outputs, "iou_predictions");
        if (iou == nullptr) {
            LOG(ERROR) << "sam decoder iou_predictions output missing";
            return StatusCode::MODEL_EMPTY_OUTPUT;
        }
        const auto* iou_data = iou->tensor.template data<float>();
        const auto iou_count = static_cast<int>(iou->tensor.element_count());
        const auto best_mask_idx = static_cast<int>(
            std::distance(iou_data, std::max_element(iou_data, iou_data + iou_count)));

        const auto* full_masks = find_output(outputs, "masks");
        if (full_masks != nullptr) {
            return decode_full_mask(full_masks->tensor, best_mask_idx, out_mask);
        }
        const auto* low_res_masks = find_output(outputs, "low_res_masks");
        if (low_res_masks != nullptr) {
            return decode_low_res_mask(low_res_masks->tensor, best_mask_idx, out_mask);
        }
        LOG(ERROR) << "sam decoder masks/low_res_masks output missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    StatusCode decode_full_mask(
        const Tensor& tensor, int mask_idx, cv::Mat& out_mask) const {
        if (tensor.shape.size() != 4 && tensor.shape.size() != 3) {
            LOG(ERROR) << "invalid sam decoder full mask shape: "
                       << jinq::models::backend::shape_to_string(tensor.shape);
            return StatusCode::MODEL_EMPTY_OUTPUT;
        }
        const auto height = static_cast<int>(tensor.shape[tensor.shape.size() - 2]);
        const auto width = static_cast<int>(tensor.shape.back());
        const auto mask_area = static_cast<int64_t>(height) * width;
        const auto* data = tensor.template data<float>() + static_cast<int64_t>(mask_idx) * mask_area;

        cv::Mat mask(height, width, CV_8UC1);
        for (int row = 0; row < height; ++row) {
            auto* row_data = mask.ptr<uchar>(row);
            for (int col = 0; col < width; ++col) {
                row_data[col] = data[row * width + col] > 0.0f ? 255 : 0;
            }
        }
        mask.copyTo(out_mask);
        return StatusCode::OK;
    }

    StatusCode decode_low_res_mask(
        const Tensor& tensor, int mask_idx, cv::Mat& out_mask) const {
        if (tensor.shape.size() != 4) {
            LOG(ERROR) << "invalid sam decoder low-res mask shape: "
                       << jinq::models::backend::shape_to_string(tensor.shape);
            return StatusCode::MODEL_EMPTY_OUTPUT;
        }
        const auto height = static_cast<int>(tensor.shape[2]);
        const auto width = static_cast<int>(tensor.shape[3]);
        const auto* data = tensor.template data<float>() +
                            static_cast<int64_t>(mask_idx) * height * width;

        cv::Mat mask(height, width, CV_32FC1);
        for (int row = 0; row < height; ++row) {
            auto* row_data = mask.ptr<float>(row);
            for (int col = 0; col < width; ++col) {
                row_data[col] = data[row * width + col];
            }
        }

        cv::resize(mask, mask, _m_encoder_input_size);
        const auto long_side = std::max(_m_ori_image_size.height, _m_ori_image_size.width);
        const auto scale = static_cast<float>(_m_encoder_input_size.height) /
                           static_cast<float>(long_side);
        const cv::Size target_size(
            static_cast<int>(scale * _m_ori_image_size.width),
            static_cast<int>(scale * _m_ori_image_size.height));
        mask = mask(cv::Rect(cv::Point(), target_size));
        cv::resize(mask, mask, _m_ori_image_size);
        cv::Mat output(_m_ori_image_size, CV_8UC1);
        for (int row = 0; row < mask.rows; ++row) {
            const auto* mask_data = mask.ptr<float>(row);
            auto* output_data = output.ptr<uchar>(row);
            for (int col = 0; col < mask.cols; ++col) {
                output_data[col] = mask_data[col] > 0.0f ? 255 : 0;
            }
        }
        output.copyTo(out_mask);
        return StatusCode::OK;
    }

    std::unique_ptr<InferenceSession> _m_session;
    cv::Size _m_ori_image_size;
    cv::Size _m_encoder_input_size = cv::Size(1024, 1024);
    bool _m_has_orig_size_input = false;
    bool _m_successfully_initialized = false;
};

SamPromptDecoder::SamPromptDecoder(
    std::unique_ptr<jinq::models::backend::InferenceSession> session)
    : _m_pimpl(std::make_unique<Impl>(std::move(session))) {}

SamPromptDecoder::~SamPromptDecoder() = default;

StatusCode SamPromptDecoder::init() {
    return _m_pimpl->init();
}

void SamPromptDecoder::set_ori_image_size(const cv::Size& ori_image_size) {
    _m_pimpl->set_ori_image_size(ori_image_size);
}

void SamPromptDecoder::set_encoder_input_size(const cv::Size& input_node_size) {
    _m_pimpl->set_encoder_input_size(input_node_size);
}

StatusCode SamPromptDecoder::decode(
    const std::vector<float>& image_embeddings,
    const std::vector<cv::Rect2f>& bboxes,
    std::vector<cv::Mat>& predicted_masks) {
    return _m_pimpl->decode(image_embeddings, bboxes, predicted_masks);
}

StatusCode SamPromptDecoder::decode(
    const std::vector<float>& image_embeddings,
    const std::vector<std::vector<cv::Point2f>>& points,
    std::vector<cv::Mat>& predicted_masks) {
    return _m_pimpl->decode(image_embeddings, points, predicted_masks);
}

bool SamPromptDecoder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

} // namespace segment_anything
} // namespace models
} // namespace jinq
