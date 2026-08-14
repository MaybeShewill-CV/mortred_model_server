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
    in.input_image_path =
        "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG";
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

TEST(cv_image_input, garbage_base64_yields_empty) {
    base64_input in;
    in.input_image_content = "not valid base64 image data";
    EXPECT_TRUE(load_image(in).empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
