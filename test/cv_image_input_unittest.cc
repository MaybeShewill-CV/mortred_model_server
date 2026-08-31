/************************************************
 * Author: Codex
 * File: cv_image_input_unittest.cc
 * Date: 2026-08-13
 ************************************************/

#include <string>
#include <vector>
#include <memory>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common/base64.h"
#include "models/backend/backend_cv_model.h"
#include "models/cv_image_input.h"

using jinq::common::StatusCode;
using jinq::models::cv_input::ImageInputLimits;
using jinq::models::cv_input::load_image;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::byte_source;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::common_io::image_input;
using jinq::models::io_define::common_io::mat_input;

// CI (tests-only) builds do not compile models/session.cpp; the wiring tests
// below instantiate BackendCvModel's vtable but never call init(), so a null
// factory satisfies the link-time reference without pulling engine headers.
namespace jinq {
namespace models {
namespace backend {
std::unique_ptr<InferenceSession> InferenceSession::create(const BackendConfig &config, std::string *err) {
    (void)config;
    (void)err;
    return nullptr;
}
} // namespace backend
} // namespace models
} // namespace jinq

namespace {

jinq::models::io_define::common_io::byte_source png_source_of(const cv::Mat &image,
                                                              byte_source::origin_kind origin) {
    std::vector<uchar> buffer;
    cv::imencode(".png", image, buffer);
    byte_source source;
    source.origin = origin;
    if (origin == byte_source::origin_kind::base64_text) {
        source.data = jinq::common::base64::encode(buffer.data(), buffer.size());
    } else {
        source.data.assign(buffer.begin(), buffer.end());
    }
    return source;
}

class StubImageModel : public jinq::models::backend::BackendCvModel<image_input, int> {
  public:
    StubImageModel() : BackendCvModel("STUB_IMAGE_MODEL") {}

    // expose the default implementation for wiring assertions
    using BackendCvModel::prepare_inputs;

  protected:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override {
        return {jinq::models::backend::NamedTensor{
            "input", jinq::models::backend::Tensor::make<uint8_t>({1, image.rows, image.cols, image.channels()})}};
    }

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &,
                           const jinq::models::backend::InferenceContext &, int &) override {
        return StatusCode::OK;
    }
};

} // namespace

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
    EXPECT_EQ(bgr.at<cv::Vec3b>(0, 0), cv::Vec3b(1, 2, 3));
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

TEST(cv_image_input, raw_bytes_roundtrip) {
    const cv::Mat image(12, 9, CV_8UC3, cv::Scalar(10, 20, 30));
    image_input in;
    in.image = png_source_of(image, byte_source::origin_kind::raw_bytes);
    const auto out = load_image(in);
    EXPECT_FALSE(out.empty());
    EXPECT_EQ(out.rows, 12);
    EXPECT_EQ(out.cols, 9);
    EXPECT_EQ(out.type(), CV_8UC3);
}

TEST(cv_image_input, image_input_base64_origin_matches_base64_path) {
    const cv::Mat image(10, 10, CV_8UC3, cv::Scalar(40, 50, 60));
    image_input in;
    in.image = png_source_of(image, byte_source::origin_kind::base64_text);
    base64_input legacy;
    legacy.input_image_content = in.image.data;

    const auto via_image_input = load_image(in);
    const auto via_legacy = load_image(legacy);
    ASSERT_FALSE(via_image_input.empty());
    ASSERT_FALSE(via_legacy.empty());
    EXPECT_EQ(via_image_input.size(), via_legacy.size());
    EXPECT_EQ(cv::sum(cv::abs(via_image_input - via_legacy)), cv::Scalar(0, 0, 0, 0));
}

TEST(cv_image_input, raw_bytes_empty_and_undecodable_yield_clear_status) {
    StatusCode status = StatusCode::OK;
    std::string error;
    image_input empty_input;
    empty_input.image.origin = byte_source::origin_kind::raw_bytes;
    EXPECT_TRUE(load_image(empty_input, ImageInputLimits{}, &status, &error).empty());
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
    EXPECT_NE(error.find("raw data is empty"), std::string::npos);

    status = StatusCode::OK;
    error.clear();
    image_input garbage_input;
    garbage_input.image.origin = byte_source::origin_kind::raw_bytes;
    garbage_input.image.data = "these are definitely not image bytes";
    EXPECT_TRUE(load_image(garbage_input, ImageInputLimits{}, &status, &error).empty());
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
    EXPECT_NE(error.find("not a decodable image"), std::string::npos);
}

TEST(cv_image_input, raw_bytes_enforce_pixel_limits) {
    const cv::Mat image(32, 32, CV_8UC3, cv::Scalar(1, 2, 3));
    image_input in;
    in.image = png_source_of(image, byte_source::origin_kind::raw_bytes);

    ImageInputLimits limits;
    limits.max_pixels = 100;
    limits.max_side = 8192;
    StatusCode status = StatusCode::OK;
    std::string error;
    EXPECT_TRUE(load_image(in, limits, &status, &error).empty());
    EXPECT_EQ(status, StatusCode::REQUEST_ENTITY_TOO_LARGE);
    EXPECT_NE(error.find("max_pixels"), std::string::npos);
}

TEST(image_input_pipeline, image_input_satisfies_image_input_trait) {
    EXPECT_TRUE(jinq::models::backend::detail::is_image_input<image_input>::value);
}

TEST(image_input_pipeline, prepare_inputs_carries_params_and_geometry) {
    StubImageModel model;
    const cv::Mat image(6, 8, CV_8UC3, cv::Scalar(1, 2, 3));
    image_input in;
    in.image = png_source_of(image, byte_source::origin_kind::raw_bytes);

    jinq::models::backend::ParamSet params;
    params.set_f32("score_threshold", 0.75f);
    in.params = &params;

    const auto prepared = model.prepare_inputs(in);
    EXPECT_EQ(prepared.status, StatusCode::OK);
    ASSERT_EQ(prepared.inputs.size(), 1u);
    EXPECT_EQ(prepared.context.params, &params);
    ASSERT_NE(prepared.context.params, nullptr);
    EXPECT_FLOAT_EQ(prepared.context.params->get_f32("score_threshold", 0.0f), 0.75f);
    EXPECT_EQ(prepared.context.source_size, cv::Size(8, 6));
    EXPECT_EQ(prepared.context.network_size, cv::Size(8, 6));
}

TEST(image_input_pipeline, prepare_inputs_without_params_keeps_nullptr) {
    StubImageModel model;
    const cv::Mat image(4, 5, CV_8UC3, cv::Scalar(1, 2, 3));
    image_input in;
    in.image = png_source_of(image, byte_source::origin_kind::base64_text);

    const auto prepared = model.prepare_inputs(in);
    EXPECT_EQ(prepared.status, StatusCode::OK);
    EXPECT_EQ(prepared.context.params, nullptr);
    EXPECT_EQ(prepared.context.source_size, cv::Size(5, 4));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
