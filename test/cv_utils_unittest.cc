/************************************************
 * Author: Codex
 * File: cv_utils_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <vector>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "models/model_io_define.h"

using jinq::common::CvUtils;
using jinq::models::io_define::object_detection::bbox;

TEST(cv_utils, calc_iou) {
    bbox a;
    a.bbox = cv::Rect2f(0, 0, 10, 10);
    a.score = 1.0f;
    a.class_id = 0;

    // partial overlap: intersection 6x6=36, union 100+100-36=164
    bbox b;
    b.bbox = cv::Rect2f(5, 5, 10, 10);
    b.score = 1.0f;
    b.class_id = 0;
    EXPECT_NEAR(CvUtils::calc_iou(a, b), 36.0 / 164.0, 1e-5);

    // disjoint boxes
    bbox c;
    c.bbox = cv::Rect2f(100, 100, 10, 10);
    c.score = 1.0f;
    c.class_id = 0;
    EXPECT_NEAR(CvUtils::calc_iou(a, c), 0.0, 1e-6);

    // zero-area boxes must not produce NaN (regression)
    bbox zero;
    zero.bbox = cv::Rect2f(0, 0, 0, 0);
    zero.score = 1.0f;
    zero.class_id = 0;
    EXPECT_EQ(CvUtils::calc_iou(zero, zero), 0.0f);
    EXPECT_EQ(CvUtils::calc_iou(zero, a), 0.0f);

    // rect overload
    EXPECT_NEAR(CvUtils::calc_iou(cv::Rect2f(0, 0, 10, 10), cv::Rect2f(5, 5, 10, 10)),
                36.0 / 164.0, 1e-5);
}

TEST(cv_utils, nms_boxes_per_class) {
    bbox low;
    low.bbox = cv::Rect2f(0, 0, 10, 10);
    low.score = 0.5f;
    low.class_id = 0;

    bbox high;
    high.bbox = cv::Rect2f(1, 1, 10, 10);
    high.score = 0.9f;
    high.class_id = 0;

    // heavily overlaps `high` but belongs to another class: per-class NMS
    // must keep it, a class agnostic NMS would drop it
    bbox other_class;
    other_class.bbox = cv::Rect2f(1, 1, 10, 10);
    other_class.score = 0.8f;
    other_class.class_id = 1;

    // below the score threshold: dropped before suppression
    bbox weak;
    weak.bbox = cv::Rect2f(100, 100, 5, 5);
    weak.score = 0.2f;
    weak.class_id = 2;

    std::vector<bbox> boxes = {low, high, other_class, weak};
    auto result = CvUtils::nms_boxes_per_class(boxes, 0.4, 0.5);

    ASSERT_EQ(result.size(), 2u);
    // class ascending, score descending inside a class
    EXPECT_EQ(result[0].class_id, 0);
    EXPECT_FLOAT_EQ(result[0].score, 0.9f);
    EXPECT_EQ(result[1].class_id, 1);
    EXPECT_FLOAT_EQ(result[1].score, 0.8f);
}

TEST(cv_utils, convert_chw_hwc_round_trip) {
    cv::Mat input(2, 2, CV_32FC3);
    input.at<cv::Vec3f>(0, 0) = cv::Vec3f(1, 2, 3);
    input.at<cv::Vec3f>(0, 1) = cv::Vec3f(4, 5, 6);
    input.at<cv::Vec3f>(1, 0) = cv::Vec3f(7, 8, 9);
    input.at<cv::Vec3f>(1, 1) = cv::Vec3f(10, 11, 12);

    auto chw = CvUtils::convert_to_chw_vec(input);
    ASSERT_EQ(chw.size(), 12u);
    EXPECT_FLOAT_EQ(chw[0], 1);
    EXPECT_FLOAT_EQ(chw[1], 4);
    EXPECT_FLOAT_EQ(chw[3], 10);
    EXPECT_FLOAT_EQ(chw[4], 2);
    EXPECT_FLOAT_EQ(chw[8], 3);
    EXPECT_FLOAT_EQ(chw[11], 12);

    auto hwc = CvUtils::convert_to_hwc_vec(chw, 3, 2, 2);
    ASSERT_EQ(hwc.size(), 12u);
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(hwc[i], static_cast<float>(i + 1));
    }
}

TEST(cv_utils, base64_cvmat_round_trip) {
    cv::Mat input(16, 16, CV_8UC3, cv::Scalar(1, 2, 3));
    auto encoded = CvUtils::encode_cvmat_into_base64_str(input);
    EXPECT_FALSE(encoded.empty());

    auto decoded = CvUtils::decode_base64_str_into_cvmat(encoded);
    EXPECT_EQ(decoded.size(), input.size());
    EXPECT_EQ(decoded.type(), input.type());
}

TEST(cv_utils, colorize_and_stack) {
    cv::Mat seg_mask(8, 8, CV_32SC1, cv::Scalar(0));
    seg_mask.row(0).setTo(1);
    seg_mask.row(1).setTo(2);
    cv::Mat color_mask;
    CvUtils::colorize_segmentation_mask(seg_mask, color_mask, 3);
    EXPECT_EQ(color_mask.size(), seg_mask.size());
    EXPECT_EQ(color_mask.type(), CV_8UC3);

    cv::Mat depth(8, 8, CV_32FC1, cv::Scalar(0));
    depth.row(0).setTo(255.0f);
    cv::Mat depth_color;
    CvUtils::colorize_depth_map(depth, depth_color);
    EXPECT_EQ(depth_color.size(), depth.size());
    EXPECT_EQ(depth_color.type(), CV_8UC3);

    // all-zero depth map must not divide by zero (regression)
    cv::Mat zero_depth(8, 8, CV_32FC1, cv::Scalar(0));
    cv::Mat zero_color;
    CvUtils::colorize_depth_map(zero_depth, zero_color);
    EXPECT_EQ(zero_color.size(), zero_depth.size());

    auto color_map = CvUtils::generate_color_map(5);
    EXPECT_EQ(color_map.size(), 5u);

    std::vector<cv::Mat> images;
    images.push_back(cv::Mat(4, 4, CV_8UC3, cv::Scalar(1)));
    images.push_back(cv::Mat(4, 4, CV_8UC3, cv::Scalar(2)));
    auto stacked = CvUtils::stack_multiple_ddpm_images(images, 0, 2);
    EXPECT_EQ(stacked.rows, 4);
    EXPECT_EQ(stacked.cols, 8);
}

// regression: the old random-and-reject palette spun forever beyond 256 classes
TEST(cv_utils, generate_color_map_large_class_count) {
    auto color_map = CvUtils::generate_color_map(1024);
    EXPECT_EQ(color_map.size(), 1024u);
}

// regression: mask label ids range over 0..max_value, the palette needs max+1 entries
TEST(cv_utils, colorize_sam_mask_max_label) {
    cv::Mat mask(4, 4, CV_32SC1, cv::Scalar(3));
    cv::Mat color_mask;
    CvUtils::colorize_sam_everything_mask(mask, color_mask);
    EXPECT_EQ(color_mask.size(), mask.size());
    EXPECT_EQ(color_mask.type(), CV_8UC3);
    // label 3 is in range, so its pixels must not be painted as the id-0 fallback
    EXPECT_NE(color_mask.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 255));
}
