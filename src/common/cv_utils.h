/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: cv_utils.h
* Date: 22-6-10
************************************************/

#ifndef MORTRED_MODEL_SERVER_CV_UTILS_H
#define MORTRED_MODEL_SERVER_CV_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/base64.h"

namespace jinq {
namespace common {

class CvUtils {
  public:

// deterministic palette: golden-angle hue sweep, O(n), works for any class count
static inline std::map<int, cv::Scalar> generate_color_map(int class_counts) {
    std::map<int, cv::Scalar> color_map;
    for (int i = 0; i < class_counts; ++i) {
        const double hue = std::fmod(i * 137.50776405003785, 360.0);
        const double x = 1.0 - std::fabs(std::fmod(hue / 60.0, 2.0) - 1.0);
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        switch (static_cast<int>(hue / 60.0) % 6) {
        case 0:
            r = 1.0; g = x;
            break;
        case 1:
            r = x; g = 1.0;
            break;
        case 2:
            g = 1.0; b = x;
            break;
        case 3:
            g = x; b = 1.0;
            break;
        case 4:
            r = x; b = 1.0;
            break;
        default:
            r = 1.0; b = x;
            break;
        }
        color_map[i] = cv::Scalar(b * 255.0, g * 255.0, r * 255.0);
    }
    return color_map;
}

template<typename T>
static inline void vis_object_detection(cv::Mat& input_image, const std::vector<T>& objs, int cls_nums) {
    auto color_map = generate_color_map(cls_nums);

    for (const auto& obj : objs) {
        auto bbox = obj.bbox;
        auto conf = obj.score;
        int cls_id = obj.class_id;
        cv::Scalar bbox_color(0, 0, 0);

        if (color_map.find(cls_id) != color_map.end()) {
            bbox_color = color_map[cls_id];
        }

        cv::rectangle(input_image, bbox, bbox_color, 3);
        char buf[128];
        snprintf(buf, sizeof(buf), "Score:%.2f, Class: %d", conf, cls_id);
        cv::putText(input_image, buf, cv::Point(bbox.x - 5, bbox.y - 5),
                    cv::FONT_ITALIC, 0.8, bbox_color, 2);
    }
}

template<typename T>
static inline void vis_text_detection(cv::Mat& input_image, const std::vector<T>& objs) {
    cv::Rect image_roi = cv::Rect(0, 0, input_image.cols, input_image.rows);

    for (const auto& obj : objs) {
        auto bbox_float = obj.bbox;
        auto conf = obj.score;
        auto bbox_int = cv::Rect(
                static_cast<int>(bbox_float.x), static_cast<int>(bbox_float.y),
                static_cast<int>(bbox_float.width), static_cast<int>(bbox_float.height));
        auto bbox_roi = bbox_int & image_roi;

        auto bbox_color = cv::Scalar(0, 0, 255);
        auto r_polygon_color = cv::Scalar(0, 255, 0);
        // draw bounding bbox
        cv::rectangle(input_image, bbox_roi, bbox_color, 2);
        // draw polygon
        std::vector<cv::Point> polygon;

        for (const auto& pt : obj.polygon) {
            polygon.push_back(cv::Point(pt));
        }

        std::vector<std::vector<cv::Point> > polygons = {polygon};
        cv::polylines(
            input_image, polygons, true, r_polygon_color, 2, cv::LINE_AA);
        // draw text information
        char buf[64];
        snprintf(buf, sizeof(buf), "Score:%.2f", conf);
        cv::putText(input_image, buf, cv::Point(bbox_int.x - 5, bbox_int.y - 5),
                    cv::FONT_ITALIC, 0.5, bbox_color, 1);
    }
}

template<typename T>
static inline void vis_feature_points(cv::Mat& input_image, const std::vector<T>& feature_points, int radius = 4) {
    for (const auto& key_pt : feature_points) {
        cv::circle(input_image, key_pt.location, static_cast<int>(radius), cv::Scalar(0, 0, 255), -1);
    }
}

static inline void colorize_segmentation_mask(const cv::Mat& input_image, cv::Mat& output_image, int cls_nums) {
    auto color_map = generate_color_map(cls_nums);

    if (output_image.empty()) {
        output_image.create(input_image.size(), CV_8UC3);
    }

    if (input_image.size() != output_image.size()) {
        LOG(ERROR) << "input image size mismatches output image size";
        return;
    }

    for (auto row = 0; row < input_image.rows; ++ row) {
        for (auto col = 0; col < input_image.cols; ++col) {
            auto cls_id = input_image.at<int32_t>(row, col);
            auto obj_color = cv::Scalar(0, 0, 0);

            if (color_map.find(cls_id) != color_map.end()) {
                obj_color = color_map[cls_id];
            }

            output_image.at<cv::Vec3b>(row, col)[0] = static_cast<uchar>(obj_color[0]);
            output_image.at<cv::Vec3b>(row, col)[1] = static_cast<uchar>(obj_color[1]);
            output_image.at<cv::Vec3b>(row, col)[2] = static_cast<uchar>(obj_color[2]);
        }
    }
}

// mask label ids are 0..max_value, so the palette needs max_value + 1 entries;
// out-of-range ids are clamped to 0
static inline void colorize_sam_everything_mask(const cv::Mat& everything_mask, cv::Mat& color_mask) {
    if (everything_mask.empty()) {
        LOG(ERROR) << "empty everything mask";
        return;
    }
    double max_value = 0.0;
    cv::minMaxIdx(everything_mask, nullptr, &max_value);
    if (max_value < 0.0) {
        LOG(ERROR) << "invalid everything mask";
        return;
    }
    const int obj_counts = static_cast<int>(max_value) + 1;
    const auto color_pool = generate_color_map(obj_counts);

    color_mask = cv::Mat::zeros(everything_mask.size(), CV_8UC3);
    for (auto row = 0; row < everything_mask.rows; ++row) {
        auto row_data = everything_mask.ptr<int32_t>(row);
        auto color_row_data = color_mask.ptr<cv::Vec3b>(row);
        for (auto col = 0; col < everything_mask.cols; ++col) {
            auto obj_id = row_data[col];
            if (obj_id < 0 || obj_id >= obj_counts) {
                obj_id = 0;
            }
            auto color = color_pool.at(obj_id);
            color_row_data[col][0] = static_cast<uchar>(color[0]);
            color_row_data[col][1] = static_cast<uchar>(color[1]);
            color_row_data[col][2] = static_cast<uchar>(color[2]);
        }
    }
}

static inline void add_segmentation_mask(
    const cv::Mat& input_image, const cv::Mat& segment_mask,
    cv::Mat& output_image, int cls_nums) {
    // prepare color map
    auto color_map = generate_color_map(cls_nums);

    if (output_image.empty()) {
        output_image.create(input_image.size(), CV_8UC3);
    }

    if (input_image.size() != output_image.size()) {
        LOG(ERROR) << "input image size mismatches output image size";
        return;
    }

    // make colorized segmentation mask
    cv::Mat colorized_mask;
    colorize_segmentation_mask(segment_mask, colorized_mask, cls_nums);

    // make add image
    cv::addWeighted(input_image, 0.6, colorized_mask, 0.4, 0.0, output_image);
}

static inline void visualize_sam_output_masks(const cv::Mat& input_image, const std::vector<cv::Mat>& masks, cv::Mat& output_image) {
    // prepare color map
    auto color_map = generate_color_map(static_cast<int>(masks.size()) + 1);
    output_image = input_image.clone();
    cv::Mat color_mask = cv::Mat::zeros(output_image.size(), CV_8UC3);

    // colorize color map
    for (size_t idx = 0; idx < masks.size(); ++idx) {
        auto color = color_map[idx];

        auto mask_b = masks[idx].clone();
        auto mask_g = masks[idx].clone();
        auto mask_r = masks[idx].clone();

        mask_b /= 255;
        mask_g /= 255;
        mask_r /= 255;

        mask_b *= color[0];
        mask_g *= color[1];
        mask_r *= color[2];

        std::vector<cv::Mat> mask_merge = {mask_b, mask_g, mask_r};
        cv::Mat tmp_color_mask;
        cv::merge(mask_merge, tmp_color_mask);
        color_mask += tmp_color_mask;
    }

    cv::addWeighted(output_image, 0.6, color_mask, 0.4, 0.0, output_image);
}

static inline void colorize_depth_map(const cv::Mat& depth_map, cv::Mat& color_mask) {
    if (depth_map.empty()) {
        LOG(ERROR) << "empty depth map";
        return;
    }
    double min_depth = 0.0;
    double max_depth = 0.0;
    cv::minMaxLoc(depth_map, &min_depth, &max_depth);
    if (max_depth <= 0.0) {
        LOG(ERROR) << "depth map is all zero, skip colorization";
        color_mask = cv::Mat::zeros(depth_map.size(), CV_8UC3);
        return;
    }
    // convert depth map
    cv::Mat normed_depth_map;
    cv::divide(depth_map, max_depth, normed_depth_map);
    normed_depth_map *= 255.0f;
    normed_depth_map.convertTo(normed_depth_map, CV_8UC1);

    // apply color map
    cv::applyColorMap(normed_depth_map, color_mask, cv::ColormapTypes::COLORMAP_JET);
}

template <class T>
static inline void visualize_fp_match_result(
    const cv::Mat& input_image0, const cv::Mat& input_image1, const std::vector<T>& match_result, cv::Mat& out_image) {
    std::vector<cv::KeyPoint> kpts0;
    std::vector<cv::KeyPoint> kpts1;
    std::vector<cv::DMatch> matches;

    for (size_t idx = 0; idx < match_result.size(); ++idx) {
        cv::KeyPoint kpt0(match_result[idx].m_fp.first.location, 0.0);
        cv::KeyPoint kpt1(match_result[idx].m_fp.second.location, 0.0);
        cv::DMatch dmatch(static_cast<int>(idx), static_cast<int>(idx), 0.0);
        kpts0.push_back(kpt0);
        kpts1.push_back(kpt1);
        matches.push_back(dmatch);
    }

    cv::drawMatches(input_image0, kpts0, input_image1, kpts1, matches, out_image);
}

// IoU with +1 pixel offset (VOC style); zero-area boxes yield 0
static inline float calc_iou(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    if (box1.width <= 0.0f || box1.height <= 0.0f ||
        box2.width <= 0.0f || box2.height <= 0.0f) {
        return 0.0f;
    }
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    float w = std::max(0.0f, x2 - x1 + 1);
    float h = std::max(0.0f, y2 - y1 + 1);
    float over_area = w * h;
    float union_area = box1.width * box1.height + box2.width * box2.height - over_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return over_area / union_area;
}

template<typename T>
static inline float calc_iou(const T& box1, const T& box2) {
    return calc_iou(box1.bbox, box2.bbox);
}

// per-class NMS: boxes are expected to expose bbox/score/class_id
template<class T>
static inline std::vector<T> nms_bboxes(const std::vector<T>& bboxes, double nms_threshold) {
    std::vector<T> result;

    if (bboxes.empty()) {
        return result;
    }

    std::map<int, std::vector<T> > bboxes_split;

    for (const auto& bbox : bboxes) {
        bboxes_split[bbox.class_id].push_back(bbox);
    }

    for (auto& iter : bboxes_split) {
        auto& candidates = iter.second;
        std::sort(candidates.begin(), candidates.end(), [](const T& a, const T& b) {
            return a.score > b.score;
        });
        std::vector<bool> suppressed(candidates.size(), false);
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (suppressed[i]) {
                continue;
            }
            result.push_back(candidates[i]);
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                if (!suppressed[j] && calc_iou(candidates[i], candidates[j]) > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

// base64 -> cv::Mat; flags default to IMREAD_COLOR (legacy behavior), the model
// input path passes IMREAD_UNCHANGED explicitly
static inline cv::Mat decode_base64_str_into_cvmat(const std::string& input, int flags = cv::IMREAD_COLOR) {
    cv::Mat ret;
    auto decoded = base64::decode(input);
    if (decoded.empty()) {
        DLOG(WARNING) << "empty or invalid base64 image data";
        return ret;
    }
    std::vector<uchar> image_vec_data(decoded.begin(), decoded.end());
    cv::imdecode(image_vec_data, flags).copyTo(ret);
    return ret;
}

static inline std::string encode_cvmat_into_base64_str(const cv::Mat& input) {
    if (input.empty()) {
        return "";
    }
    std::vector<uchar> imencode_buffer;
    cv::imencode(".jpg", input, imencode_buffer);
    return base64::encode(imencode_buffer.data(), imencode_buffer.size());
}

// HWC -> CHW float conversion; supports CV_32FC1 and CV_32FC3
static inline std::vector<float> convert_to_chw_vec(const cv::Mat& input) {
    std::vector<float> data;
    if (input.empty()) {
        LOG(ERROR) << "empty input mat";
        return data;
    }
    const size_t plane = static_cast<size_t>(input.rows) * input.cols;

    if (input.type() == CV_32FC3) {
        data.resize(3 * plane);
        for (int y = 0; y < input.rows; ++y) {
            auto raw_data = input.ptr<cv::Vec3f>(y);
            for (int x = 0; x < input.cols; ++x) {
                for (int c = 0; c < 3; ++c) {
                    data[c * plane + y * input.cols + x] = raw_data[x][c];
                }
            }
        }
    } else if (input.type() == CV_32FC1) {
        data.resize(plane);
        for (int y = 0; y < input.rows; ++y) {
            auto raw_data = input.ptr<float>(y);
            for (int x = 0; x < input.cols; ++x) {
                data[y * input.cols + x] = raw_data[x];
            }
        }
    } else {
        LOG(ERROR) << "unsupported mat type: " << input.type() << ", only CV_32FC1/CV_32FC3 supported";
    }
    return data;
}

// CHW -> HWC conversion
template<class T>
static inline std::vector<T> convert_to_hwc_vec(const std::vector<T>& input, int c, int h, int w) {
    // only support 3 channel image
    if (input.size() != static_cast<size_t>(h) * w * c) {
        LOG(ERROR) << "input size " << input.size() << " mismatches h*w*c "
                   << static_cast<size_t>(h) * w * c;
        return std::vector<T>();
    }
    std::vector<T> result;
    result.resize(input.size());

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            for (int channel = 0; channel < c; ++channel) {
                result[row * (w * c) + col * c + channel] = input[channel * (h * w) + row * w + col];
            }
        }
    }

    return result;
}

static inline cv::Mat stack_multiple_ddpm_images(const std::vector<cv::Mat>& multi_images, const int gap=2, const int images_per_row=8) {
    if (multi_images.empty()) {
        LOG(ERROR) << "input image vector is empty";
        return cv::Mat();
    }

    int h_spacing = gap;
    int v_spacing = gap;

    int image_width = multi_images[0].cols;
    int image_height = multi_images[0].rows;

    int total_rows = std::ceil(static_cast<float>(multi_images.size()) / images_per_row);
    int total_cols = std::min(static_cast<int>(multi_images.size()), images_per_row);

    int big_image_width = total_cols * image_width + (total_cols - 1) * h_spacing;
    int big_image_height = total_rows * image_height + (total_rows - 1) * v_spacing;

    cv::Mat big_image = cv::Mat::zeros(big_image_height, big_image_width, multi_images[0].type());

    for (size_t i = 0; i < multi_images.size(); ++i) {
        int row = static_cast<int>(i) / images_per_row;
        int col = static_cast<int>(i) % images_per_row;

        int x = col * (image_width + h_spacing);
        int y = row * (image_height + v_spacing);

        multi_images[i].copyTo(big_image(cv::Rect(x, y, image_width, image_height)));
    }

    return big_image;
}

// copy only when source and tensor byte sizes match; returns false on mismatch
static inline bool copy_image_to_tensor(void* dst, const cv::Mat& image, int dst_bytes) {
    size_t src_bytes = image.total() * image.elemSize();
    if (src_bytes != static_cast<size_t>(dst_bytes)) {
        LOG(ERROR) << "image byte size " << src_bytes << " mismatches tensor byte size " << dst_bytes;
        return false;
    }
    ::memcpy(dst, image.data, dst_bytes);
    return true;
}

template<typename T>
static inline bool copy_image_to_tensor(void* dst, const std::vector<T>& data, int dst_bytes) {
    size_t src_bytes = data.size() * sizeof(T);
    if (src_bytes != static_cast<size_t>(dst_bytes)) {
        LOG(ERROR) << "data byte size " << src_bytes << " mismatches tensor byte size " << dst_bytes;
        return false;
    }
    ::memcpy(dst, data.data(), dst_bytes);
    return true;
}

};
}  // namespace common
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_CV_UTILS_H
