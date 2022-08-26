/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: cv_utils.h
* Date: 22-6-10
************************************************/

#ifndef MM_AI_SERVER_CV_UTILS_H
#define MM_AI_SERVER_CV_UTILS_H

#include <map>
#include <random>

#include <opencv2/opencv.hpp>
#include "base64/libbase64.h"

namespace jinq {
namespace common {
class CvUtils {
public:
    CvUtils() = delete;
    ~CvUtils() = delete;
    CvUtils(const CvUtils& transformer) = delete;
    CvUtils& operator=(const CvUtils& transformer) = delete;

    /***
     *
     * @param class_counts
     * @return
     */
    static std::map<int, cv::Scalar> generate_color_map(int class_counts) {
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
     * @tparam T
     * @param input_image
     * @param objs
     * @param cls_nums
     */
    template<typename T>
    static void vis_object_detection(cv::Mat& input_image, std::vector<T>& objs, int cls_nums) {
        auto color_map = generate_color_map(cls_nums);

        for (auto& obj : objs) {
            auto bbox = obj.bbox;
            auto conf = obj.score;
            int cls_id = obj.class_id;
            cv::Scalar bbox_color(0, 0, 0);

            if (color_map.find(cls_id) != color_map.end()) {
                bbox_color = color_map[cls_id];
            }

            cv::rectangle(input_image, bbox, bbox_color, 3);
            char buf[128];
            sprintf(buf, "Score:%1.2f, Class: %d", conf, cls_id);
            cv::putText(input_image, buf, cv::Point(bbox.x - 5, bbox.y - 5),
                        cv::FONT_ITALIC, 0.8, bbox_color, 2);
        }
    }

    /***
     *
     * @tparam T
     * @param input_image
     * @param objs
     * @param cls_nums
     */
    template<typename T>
    static void vis_text_detection(cv::Mat& input_image, std::vector<T>& objs) {
        cv::Rect image_roi = cv::Rect(0, 0, input_image.cols, input_image.rows);

        for (auto& obj : objs) {
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

            for (auto& pt : obj.polygon) {
                cv::Point new_pt(pt);
                polygon.push_back(new_pt);
            }

            std::vector<std::vector<cv::Point> > polygons = {polygon};
            cv::polylines(
                input_image, polygons, true, r_polygon_color, 2, cv::LINE_AA);
            // draw text information
            char buf[64];
            sprintf(buf, "Score:%1.2f", conf);
            cv::putText(input_image, buf, cv::Point(bbox_int.x - 5, bbox_int.y - 5),
                        cv::FONT_ITALIC, 0.5, bbox_color, 1);
        }
    }

    /***
     *
     * @tparam T
     * @param input_image
     * @param objs
     */
    template<typename T>
    static void vis_feature_points(cv::Mat& input_image, const std::vector<T>& feature_points, int radius = 4) {
        for (const auto& key_pt : feature_points) {
            cv::circle(input_image, key_pt.location, static_cast<int>(radius), cv::Scalar(0, 0, 255), -1);
        }
    }

    /***
     *
     * @tparam T
     * @param input_image
     * @param objs
     * @param cls_nums
     */
    static void colorize_segmentation_mask(const cv::Mat& input_image, cv::Mat& output_image, int cls_nums) {
        auto color_map = generate_color_map(cls_nums);

        if (output_image.empty()) {
            output_image.create(input_image.size(), CV_8UC3);
        }

        assert(input_image.size() == output_image.size());

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

    /***
     *
     * @param input_image
     * @param output_image
     * @param cls_nums
     */
    static void add_segmentation_mask(
        const cv::Mat& input_image, const cv::Mat& segment_mask,
        cv::Mat& output_image, int cls_nums) {
        // prepare color map
        auto color_map = generate_color_map(cls_nums);

        if (output_image.empty()) {
            output_image.create(input_image.size(), CV_8UC3);
        }

        assert(input_image.size() == output_image.size());

        // make colorized segmentation mask
        cv::Mat colorized_mask;
        colorize_segmentation_mask(segment_mask, colorized_mask, cls_nums);

        // make add image
        cv::addWeighted(input_image, 0.6, colorized_mask, 0.4, 0.0, output_image);
    }

    /***
    *
    * @param box1
    * @param box2
    * @return
    */
    template<typename T>
    static float calc_iou(const T& box1, const T& box2) {
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
     * @tparam T
     * @param bboxes
     * @param nms_threshold
     * @return
     */
    template<class T>
    static std::vector<T> nms_bboxes(std::vector<T>& bboxes, double nms_threshold) {
        std::vector<T> result;

        if (bboxes.empty()) {
            return result;
        }

        std::map<int, std::vector<T> > bboxes_split;

        for (const auto& bbox : bboxes) {
            auto cls_id = bbox.class_id;

            if (bboxes_split.find(cls_id) == bboxes_split.end()) {
                bboxes_split.insert(std::make_pair(cls_id, std::vector<T>({bbox})));
            } else {
                bboxes_split[cls_id].push_back(bbox);
            }
        }

        for (auto& iter : bboxes_split) {
            auto tmp_bboxes = iter.second;
            // sort the bounding boxes by the detection score
            std::sort(tmp_bboxes.begin(), tmp_bboxes.end(), [](const T & box1, const T & box2) {
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

    /***
     *
     * @param input
     * @return
     */
    static cv::Mat decode_base64_str_into_cvmat(const std::string& input) {
        char out[input.size() * 2];
        size_t out_len = 0;
        base64_decode(input.c_str(), input.size(), out, &out_len, 0);
        std::vector<uchar> image_vec_data(out, out + out_len);
        cv::Mat ret;
        cv::imdecode(image_vec_data, cv::IMREAD_UNCHANGED).copyTo(ret);

        return ret;
    }

    /***
     *
     * @param input
     * @return
     */
    static std::string encode_cvmat_into_base64_str(const cv::Mat& input) {
        if (!input.data || input.empty()) {
            return "";
        }

        std::vector<uchar> imencode_buffer;
        cv::imencode(".jpg", input, imencode_buffer);

        char in[imencode_buffer.size() + 1];
        in[imencode_buffer.size()] = '\0';

        for (int idx = 0; idx < imencode_buffer.size(); ++idx) {
            in[idx] = static_cast<char>(imencode_buffer[idx]);
        }

        char out[imencode_buffer.size() * 2];
        size_t out_len = 0;
        base64_encode(in, imencode_buffer.size(), out, &out_len, 0);

        return out;
    }

    /***
     *
     * @param input
     * @return
     */
    static std::vector<float> convert_to_chw_vec(const cv::Mat& input) {
        std::vector<float> data;
        if (input.type() == CV_32FC3) {
            data.resize(input.channels() * input.rows * input.cols);
            for(int y = 0; y < input.rows; ++y) {
                for(int x = 0; x < input.cols; ++x) {
                    for(int c = 0; c < input.channels(); ++c) {
                        data[c * (input.rows * input.cols) + y * input.cols + x] =
                                input.at<cv::Vec3f>(y, x)[c];
                    }
                }
            }
            return data;
        } else {
            LOG(ERROR) << "Only support 32fc3. Not support for opencv mat type of: " << input.type();
            return data;
        }
    }

    /***
     *
     * @param input
     * @return
     */
    template<class T>
    static std::vector<T> convert_to_hwc_vec(const std::vector<T>& input, int c, int h, int w) {
        // only support 3 channel image
        assert(input.size() == h * w * c);
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
};
}
}

#endif //MM_AI_SERVER_CV_UTILS_H
