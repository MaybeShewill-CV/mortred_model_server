#include <gtest/gtest.h>

#include <string>

#include "models/object_detection/detection_params.h"

using jinq::models::object_detection::DetectionParams;
using jinq::models::object_detection::parse_model_input_size;

namespace {

toml::table parse_toml(const std::string &content) {
    auto parsed = toml::parse(content);
    if (!parsed) {
        ADD_FAILURE() << "fixture toml parse failed";
        return toml::table{};
    }
    return std::move(parsed).table();
}

} // namespace

TEST(DetectionParams, ParsesValidParamsAndLabels) {
    auto params = parse_toml(R"toml(
model_score_threshold = 0.25
model_nms_threshold = 0.5
model_keep_top_k = 1000
model_class_nums = 2
class_names = ["person", "car"]
)toml");
    DetectionParams result;
    result.score_threshold = 0.9f;
    result.nms_threshold = 0.9f;
    result.keep_top_k = 1;
    result.class_nums = 1;
    std::string error;
    ASSERT_TRUE(DetectionParams::parse(params, &result, &error)) << error;
    EXPECT_FLOAT_EQ(result.score_threshold, 0.25f);
    EXPECT_FLOAT_EQ(result.nms_threshold, 0.5f);
    EXPECT_EQ(result.keep_top_k, 1000);
    EXPECT_EQ(result.class_nums, 2);
    ASSERT_EQ(result.class_names.size(), 2u);
    EXPECT_EQ(result.class_names[0], "person");
}

TEST(DetectionParams, RejectsOutOfRangeValues) {
    const std::vector<std::string> invalid = {
        "model_score_threshold = -0.1", "model_score_threshold = 1.1", "model_nms_threshold = -0.1", "model_nms_threshold = 1.1",
        "model_keep_top_k = 0",         "model_keep_top_k = 10001",    "model_class_nums = 0",       "min_box_area_px = -1",
    };
    for (const auto &line : invalid) {
        DetectionParams result;
        std::string error;
        const auto params = parse_toml(line);
        EXPECT_FALSE(DetectionParams::parse(params, &result, &error)) << line;
        EXPECT_FALSE(error.empty()) << line;
    }
}

TEST(DetectionParams, RejectsInvalidClassNames) {
    DetectionParams result;
    result.class_nums = 2;
    std::string error;
    auto params = parse_toml(R"toml(
model_class_nums = 2
class_names = ["person"]
)toml");
    EXPECT_FALSE(DetectionParams::parse(params, &result, &error));
    EXPECT_NE(error.find("class_names"), std::string::npos);

    params = parse_toml(R"toml(
model_class_nums = 2
class_names = ["person", 3]
)toml");
    EXPECT_FALSE(DetectionParams::parse(params, &result, &error));
    EXPECT_NE(error.find("non-string"), std::string::npos);
}

TEST(DetectionParams, RejectsDeprecatedInputNodeSize) {
    DetectionParams result;
    std::string error;
    const auto params = parse_toml("input_node_size = [640, 640]");
    EXPECT_FALSE(DetectionParams::parse(params, &result, &error));
    EXPECT_NE(error.find("model_input_image_size"), std::string::npos);
}

TEST(DetectionParams, ParsesAndValidatesInputSize) {
    cv::Size size;
    std::string error;
    auto params = parse_toml("model_input_image_size = [480, 640]");
    ASSERT_TRUE(parse_model_input_size(params, &size, &error)) << error;
    EXPECT_EQ(size, cv::Size(640, 480));

    for (const std::string value :
         {"model_input_image_size = [0, 640]", "model_input_image_size = [640]", "model_input_image_size = \"bad\""}) {
        params = parse_toml(value);
        EXPECT_FALSE(parse_model_input_size(params, &size, &error)) << value;
        EXPECT_FALSE(error.empty()) << value;
    }
}
