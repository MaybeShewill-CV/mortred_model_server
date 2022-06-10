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

namespace morted {
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
            int cls_id = obj.class_id;
            auto obj_color = cv::Scalar(0, 0, 0);
            float obj_score = obj.score;

            if (color_map.find(cls_id) != color_map.end()) {
                obj_color = color_map[cls_id];
            }

            cv::rectangle(input_image, obj.bbox, obj_color, 2);
            cv::Point text_org(obj.bbox.x - 5, obj.bbox.y - 5);
            std::string obj_str = "cls_id: " + std::to_string(cls_id) + ", score: " + std::to_string(obj_score);
            cv::putText(
                input_image, obj_str, text_org,
                cv::FONT_HERSHEY_PLAIN, 1.0, obj_color, 1);
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
};
}
}

#endif //MM_AI_SERVER_CV_UTILS_H
