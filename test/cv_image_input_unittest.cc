/************************************************
 * Author: Codex
 * File: cv_image_input_unittest.cc
 * Date: 2026-08-13
 ************************************************/

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common/base64.h"
#include "models/cv_image_input.h"

using jinq::common::StatusCode;
using jinq::models::cv_input::ImageInputLimits;
using jinq::models::cv_input::load_image;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::mat_input;

TEST(cv_image_input, mat_input_passthrough) {
    cv::Mat img(8, 8, CV_8UC3, cv::Scalar(1, 2, 3));
    mat_input in;
    in.input_image = img;
    auto out = load_image(in);
    EXPECT_FALSE(out.empty());
    EXPECT_EQ(out.cols, 8);
    EXPECT_EQ(out.rows, 8);
    EXPECT_EQ(out.channels(), 3);
}

TEST(cv_image_input, missing_file_yields_empty) {
    file_input in;
    in.input_image_path = "/tmp/definitely_not_exist_xyz.jpg";
    EXPECT_TRUE(load_image(in).empty());
}

TEST(cv_image_input, valid_file_yields_image) {
    file_input in;
    in.input_image_path = "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG";
    auto out = load_image(in);
    EXPECT_FALSE(out.empty());
    EXPECT_EQ(out.channels(), 3);
}

TEST(cv_image_input, base64_roundtrip) {
    cv::Mat img(16, 16, CV_8UC3, cv::Scalar(10, 20, 30));
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);
    auto b64 = jinq::common::base64::encode(buf.data(), buf.size());

    base64_input in;
    in.input_image_content = b64;
    auto out = load_image(in);
    EXPECT_FALSE(out.empty());
    EXPECT_EQ(out.rows, 16);
    EXPECT_EQ(out.cols, 16);
}

TEST(cv_image_input, normalizes_gray_and_four_channel_mats) {
    cv::Mat gray(4, 5, CV_8UC1, cv::Scalar(123));
    mat_input gray_input;
    gray_input.input_image = gray;
    auto gray_bgr = load_image(gray_input);
    EXPECT_EQ(gray_bgr.type(), CV_8UC3);
    EXPECT_EQ(gray_bgr.channels(), 3);
    EXPECT_EQ(gray_bgr.at<cv::Vec3b>(0, 0), cv::Vec3b(123, 123, 123));

    cv::Mat bgra(4, 5, CV_8UC4, cv::Scalar(1, 2, 3, 255));
    mat_input bgra_input;
    bgra_input.input_image = bgra;
    auto bgr = load_image(bgra_input);
    EXPECT_EQ(bgr.type(), CV_8UC3);
    EXPECT_EQ(bgr.at<cv::Vec3b>(0, 0), cv::Vec3b(3, 2, 1));
}

TEST(cv_image_input, rejects_unsupported_mat_type) {
    mat_input input;
    input.input_image = cv::Mat(4, 4, CV_32FC3, cv::Scalar(1, 2, 3));
    StatusCode status = StatusCode::OK;
    std::string error;
    const auto result = load_image(input, ImageInputLimits{}, &status, &error);
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
    EXPECT_NE(error.find("unsupported input Mat type"), std::string::npos);
}

TEST(cv_image_input, enforces_pixel_and_side_limits) {
    mat_input input;
    input.input_image = cv::Mat(32, 32, CV_8UC3, cv::Scalar(1, 2, 3));

    ImageInputLimits limits;
    limits.max_pixels = 100;
    limits.max_side = 8192;
    StatusCode status = StatusCode::OK;
    std::string error;
    EXPECT_TRUE(load_image(input, limits, &status, &error).empty());
    EXPECT_EQ(status, StatusCode::REQUEST_ENTITY_TOO_LARGE);
    EXPECT_NE(error.find("max_pixels"), std::string::npos);

    limits.max_pixels = 16777216;
    limits.max_side = 16;
    status = StatusCode::OK;
    error.clear();
    EXPECT_TRUE(load_image(input, limits, &status, &error).empty());
    EXPECT_EQ(status, StatusCode::REQUEST_ENTITY_TOO_LARGE);
    EXPECT_NE(error.find("max_side"), std::string::npos);
}

TEST(cv_image_input, garbage_base64_yields_empty) {
    base64_input in;
    in.input_image_content = "not valid base64 image data";
    EXPECT_TRUE(load_image(in).empty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
