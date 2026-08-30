#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "models/backend/model_runtime.h"

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::ImagePipeline;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::OutputReader;
using jinq::models::backend::ParamReader;
using jinq::models::backend::SessionIoValidator;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorInfo;

namespace {

toml::table parse_toml(const std::string &content) {
    auto parsed = toml::parse(content);
    if (!parsed) {
        ADD_FAILURE() << "fixture TOML parse failed";
        return {};
    }
    return std::move(parsed).table();
}

NamedTensor scores_tensor(const std::string &name) {
    NamedTensor output;
    output.name = name;
    output.tensor = Tensor::make<float>({1, 3});
    auto *data = reinterpret_cast<float *>(output.tensor.buffer.data());
    data[0] = 0.1f;
    data[1] = 0.2f;
    data[2] = 0.7f;
    return output;
}

class FakeSession final : public jinq::models::backend::InferenceSession {
  public:
    FakeSession(std::vector<TensorInfo> inputs, std::vector<TensorInfo> outputs)
        : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}

    const std::vector<TensorInfo> &inputs() const override { return inputs_; }
    const std::vector<TensorInfo> &outputs() const override { return outputs_; }
    StatusCode run(const std::vector<NamedTensor> &, std::vector<NamedTensor> &) override { return StatusCode::MODEL_RUN_SESSION_FAILED; }

  private:
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
};

TensorInfo image_info(DType dtype = DType::F32) { return {"images", dtype, {1, 3, 4, 5}, false}; }

} // namespace

TEST(ImagePipeline, ProducesNchwAndPreservesSourceImage) {
    const cv::Mat source(2, 3, CV_8UC3, cv::Scalar(1, 2, 3));
    const cv::Mat source_copy = source.clone();

    auto result = ImagePipeline(source).bgr_to_rgb().to_float().nchw("images");
    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(result.value.name, "images");
    EXPECT_EQ(result.value.tensor.dtype, DType::F32);
    EXPECT_EQ(result.value.tensor.shape, std::vector<int64_t>({1, 3, 2, 3}));
    ASSERT_EQ(result.value.tensor.buffer.size(), 18u * sizeof(float));

    const auto *data = reinterpret_cast<const float *>(result.value.tensor.buffer.data());
    for (size_t idx = 0; idx < 6; ++idx) {
        EXPECT_FLOAT_EQ(data[idx], 3.0f);
        EXPECT_FLOAT_EQ(data[6 + idx], 2.0f);
        EXPECT_FLOAT_EQ(data[12 + idx], 1.0f);
    }
    EXPECT_FLOAT_EQ(cv::norm(source, source_copy, cv::NORM_INF), 0.0f);
}

TEST(ImagePipeline, ProducesNhwcAndSupportsMeanStd) {
    const cv::Mat source(2, 2, CV_8UC3, cv::Scalar(255, 0, 127));
    auto result =
        ImagePipeline(source).bgr_to_rgb().to_float().scale(1.0f / 255.0f).mean_std({0.0f, 0.5f, 1.0f}, {1.0f, 0.5f, 0.25f}).nhwc("input");
    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(result.value.tensor.shape, std::vector<int64_t>({1, 2, 2, 3}));

    const auto *data = reinterpret_cast<const float *>(result.value.tensor.buffer.data());
    for (size_t idx = 0; idx < 4; ++idx) {
        EXPECT_NEAR(data[idx * 3 + 0], 127.0f / 255.0f, 1e-6f);
        EXPECT_FLOAT_EQ(data[idx * 3 + 1], -1.0f);
        EXPECT_FLOAT_EQ(data[idx * 3 + 2], 0.0f);
    }
}

TEST(ImagePipeline, ConvertsBgraToRgbAndRejectsOtherChannelCounts) {
    // BGRA(1,2,3,255) -> RGB(3,2,1); the alpha plane is dropped
    const cv::Mat source(2, 2, CV_8UC4, cv::Scalar(1, 2, 3, 255));
    auto result = ImagePipeline(source).bgra_to_rgb().to_float().nhwc("input");
    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(result.value.tensor.shape, std::vector<int64_t>({1, 2, 2, 3}));

    const auto *data = reinterpret_cast<const float *>(result.value.tensor.buffer.data());
    for (size_t idx = 0; idx < 4; ++idx) {
        EXPECT_FLOAT_EQ(data[idx * 3 + 0], 3.0f);
        EXPECT_FLOAT_EQ(data[idx * 3 + 1], 2.0f);
        EXPECT_FLOAT_EQ(data[idx * 3 + 2], 1.0f);
    }

    // a 3-channel image must not silently pass through bgra_to_rgb
    const cv::Mat bgr(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    const auto bad_channels = ImagePipeline(bgr).bgra_to_rgb().nhwc("input");
    ASSERT_FALSE(bad_channels.ok());
    EXPECT_EQ(bad_channels.status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
}

TEST(ImagePipeline, SupportsCenterCropAndRejectsInvalidOperations) {
    const cv::Mat source(4, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    auto cropped = ImagePipeline(source).center_crop({2, 2}).to_float().nchw("images");
    ASSERT_TRUE(cropped.ok()) << cropped.error;
    EXPECT_EQ(cropped.value.tensor.shape, std::vector<int64_t>({1, 3, 2, 2}));

    const auto empty = ImagePipeline(cv::Mat()).nchw("images");
    ASSERT_FALSE(empty.ok());
    EXPECT_EQ(empty.status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);

    const cv::Mat gray(2, 2, CV_8UC1);
    const auto bad_channels = ImagePipeline(gray).bgr_to_rgb().nchw("images");
    ASSERT_FALSE(bad_channels.ok());
    EXPECT_EQ(bad_channels.status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
}

TEST(OutputReader, EnforcesNamedF32Contract) {
    std::vector<NamedTensor> outputs{scores_tensor("scores")};
    auto valid = OutputReader(outputs, "scores").f32().shape({1, 3}).finite().read();
    ASSERT_TRUE(valid.ok()) << valid.error;
    EXPECT_EQ(valid.value.tensor->shape, std::vector<int64_t>({1, 3}));
    EXPECT_FLOAT_EQ(valid.value.data[2], 0.7f);

    EXPECT_EQ(OutputReader(outputs, "missing").f32().shape({1, 3}).finite().read().status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_EQ(OutputReader(outputs, "scores").f32().shape({1, 2}).finite().read().status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    outputs.front().tensor = Tensor::make<int32_t>({1, 3});
    EXPECT_EQ(OutputReader(outputs, "scores").f32().shape({1, 3}).finite().read().status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    outputs.front() = scores_tensor("scores");
    reinterpret_cast<float *>(outputs.front().tensor.buffer.data())[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(OutputReader(outputs, "scores").f32().shape({1, 3}).finite().read().status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(ParamReader, ParsesAndValidatesCommonTypes) {
    const auto params = parse_toml(R"toml(
enabled = true
count = 3
ratio = 0.75
name = "person"
size = [4, 5]
)toml");
    bool enabled = false;
    int32_t count = 0;
    float ratio = 0.0f;
    std::string name;
    cv::Size size;
    ParamReader reader(params, "TEST");
    reader.get("enabled", &enabled)
        .get("count", &count)
        .min(1)
        .max(10)
        .get("ratio", &ratio)
        .min(0.0)
        .max(1.0)
        .get("name", &name)
        .non_empty()
        .get("size", &size);
    ASSERT_TRUE(reader.ok()) << reader.status().error;
    EXPECT_TRUE(enabled);
    EXPECT_EQ(count, 3);
    EXPECT_FLOAT_EQ(ratio, 0.75f);
    EXPECT_EQ(name, "person");
    EXPECT_EQ(size, cv::Size(5, 4));

    auto invalid = parse_toml("ratio = 1.5");
    float bad_ratio = 0.0f;
    ParamReader bad_reader(invalid, "BAD");
    bad_reader.get("ratio", &bad_ratio).min(0.0).max(1.0);
    EXPECT_FALSE(bad_reader.ok());
    EXPECT_EQ(bad_reader.status().status, StatusCode::MODEL_INIT_FAILED);
}

TEST(ParamReader, RejectsUnknownKeys) {
    const auto params = parse_toml("known = 1\nunknown = 2");
    int32_t known = 0;
    ParamReader reader(params, "UNKNOWN_TEST");
    reader.get("known", &known).allow_only_keys({"known"});
    EXPECT_FALSE(reader.ok());
    EXPECT_NE(reader.status().error.find("unknown param 'unknown'"), std::string::npos);
}

TEST(SessionIoValidator, ValidatesNameDtypeShapeAndLayout) {
    FakeSession session({image_info()}, {TensorInfo{"output", DType::F32, {1, 3}, false}});
    auto input = SessionIoValidator(session).input("images").dtype(DType::F32).nchw().channels(3).static_shape().validate();
    ASSERT_TRUE(input.ok()) << input.error;
    EXPECT_EQ(input.value.name, "images");

    auto output = SessionIoValidator(session).output("output").dtype(DType::F32).rank(2).validate();
    ASSERT_TRUE(output.ok()) << output.error;
    EXPECT_EQ(output.value.shape, std::vector<int64_t>({1, 3}));

    EXPECT_EQ(SessionIoValidator(session).input("missing").validate().status, StatusCode::MODEL_INIT_FAILED);
    EXPECT_EQ(SessionIoValidator(session).input("images").dtype(DType::I32).validate().status, StatusCode::MODEL_INIT_FAILED);
    EXPECT_EQ(SessionIoValidator(session).input("images").nhwc().channels(3).validate().status, StatusCode::MODEL_INIT_FAILED);
}

TEST(SessionIoValidator, AllowsDynamicBatchButRejectsDynamicSpatialDims) {
    TensorInfo dynamic = {"images", DType::F32, {-1, 3, 4, 5}, true};
    FakeSession dynamic_batch({dynamic}, {});
    auto valid = SessionIoValidator(dynamic_batch).input().f32().nchw().channels(3).allow_dynamic_batch().validate();
    EXPECT_TRUE(valid.ok()) << valid.error;

    TensorInfo dynamic_spatial = {"images", DType::F32, {-1, 3, -1, 5}, true};
    FakeSession invalid_session({dynamic_spatial}, {});
    auto invalid = SessionIoValidator(invalid_session).input().f32().nchw().allow_dynamic_batch().validate();
    ASSERT_FALSE(invalid.ok());
    EXPECT_EQ(invalid.status, StatusCode::MODEL_INIT_FAILED);
}
