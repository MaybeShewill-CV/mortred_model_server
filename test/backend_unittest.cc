/************************************************
 * Author: Codex
 * File: backend_unittest.cc
 * Date: 2026-08-20
 *
 * L1 tests of the unified inference session layer. Uses the real vendored
 * weights (no mocks): MNN mobilenetv2 (cpu), ONNX ddpm-unet (cpu) and the
 * TRT yolov8 engine (gpu, skipped when unavailable). GPU/cuda related cases
 * are skipped through GTEST_SKIP so the suite stays green on cpu-only CI.
 ************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <toml/toml.hpp>

#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/backend/backend_config.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"

namespace {

using jinq::common::StatusCode;
using jinq::models::backend::BackendConfig;
using jinq::models::backend::InferenceSession;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorInfo;

constexpr const char* kMnnModel = "weights/classification/mobilenetv2/mobilenetv2_ilsvrc2012.mnn";
constexpr const char* kOnnxModel = "weights/diffusion/ddpm/ddpm_unet_celeba-hq-128x128.onnx";
constexpr const char* kTrtModel = "weights/object_detection/yolov8/yolov8s.engine";
constexpr const char* kLightglueExtractor =
    "weights/feature_point/lightglue/extractor.engine";
constexpr const char* kFeatureImage =
    "demo_data/model_test_input/feature_point/match_test_01.jpg";

bool file_exists(const std::string& path) {
    return jinq::common::FilePathUtil::is_file_exist(path);
}

toml::table parse_toml(const std::string& content) {
    auto parsed = toml::parse(content);
    if (!parsed) {
 ADD_FAILURE() << "fixture toml parse failed: "
   << std::string(parsed.error().description());
        return toml::table{};
    }
    return std::move(parsed).table();
}

BackendConfig make_config(const std::string& type, const std::string& model_path,
                          const std::string& device = "cpu") {
    BackendConfig config;
    config.type = type;
    config.model_file_path = model_path;
    config.device = device;
    config.threads = 2;
    return config;
}

const TensorInfo& find_info(const std::vector<TensorInfo>& infos, const std::string& name) {
    for (const auto& info : infos) {
        if (info.name == name) {
            return info;
        }
    }
    ADD_FAILURE() << "tensor info not found: " << name;
    static TensorInfo empty;
    return empty;
}

std::string info_names(const std::vector<TensorInfo>& infos) {
    std::string out = "[";
    for (const auto& info : infos) {
        out += info.name + ",";
    }
    return out + "]";
}

}  // namespace

TEST(BackendTensor, MakeAndFromMat) {
    auto tensor = Tensor::make<float>({1, 2, 3});
    EXPECT_EQ(tensor.dtype, jinq::models::backend::DType::F32);
    EXPECT_EQ(tensor.element_count(), 6);
    EXPECT_EQ(tensor.byte_size(), 6 * sizeof(float));
    EXPECT_TRUE(tensor.shape_is_concrete());

    cv::Mat image(4, 5, CV_8UC3, cv::Scalar(1, 2, 3));
    bool ok = false;
    auto from_mat = Tensor::from_mat(image, &ok);
    EXPECT_TRUE(ok);
    EXPECT_EQ(from_mat.dtype, jinq::models::backend::DType::U8);
    EXPECT_EQ(from_mat.shape, (std::vector<int64_t>{1, 4, 5, 3}));
    EXPECT_EQ(from_mat.buffer[0], 1);

    cv::Mat bad_depth;
    image.convertTo(bad_depth, CV_64FC3);
    ok = true;
    const auto rejected = Tensor::from_mat(bad_depth, &ok);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(rejected.buffer.empty());
}

TEST(BackendConfig, ParseValidAndInvalidBlocks) {
    const auto valid = parse_toml(R"toml(
[MODEL]
[MODEL.backend]
type = "mnn"
model_file_path = "a.mnn"
device = "cuda"
threads = 3
[MODEL.params]
score = 0.5
)toml");
    BackendConfig config;
    std::string err;
    const toml::table* valid_section = valid["MODEL"].as_table();
    ASSERT_NE(valid_section, nullptr);
    ASSERT_TRUE(jinq::models::backend::parse_backend_config(*valid_section, &config, &err)) << err;
    EXPECT_EQ(config.type, "mnn");
    EXPECT_EQ(config.model_file_path, "a.mnn");
    EXPECT_EQ(config.device, "cuda");
    EXPECT_EQ(config.threads, 3);

    const auto unknown_type = parse_toml(R"toml(
[MODEL]
[MODEL.backend]
type = "ncnn"
model_file_path = "a.mnn"
)toml");
    const toml::table* unknown_section = unknown_type["MODEL"].as_table();
    ASSERT_NE(unknown_section, nullptr);
    EXPECT_FALSE(jinq::models::backend::parse_backend_config(*unknown_section, &config, &err));
    EXPECT_NE(err.find("unknown backend"), std::string::npos) << err;

    const auto bad_device = parse_toml(R"toml(
[MODEL]
[MODEL.backend]
type = "mnn"
model_file_path = "a.mnn"
device = "npu"
)toml");
    const toml::table* device_section = bad_device["MODEL"].as_table();
    ASSERT_NE(device_section, nullptr);
    EXPECT_FALSE(jinq::models::backend::parse_backend_config(*device_section, &config, &err));
    EXPECT_NE(err.find("device"), std::string::npos) << err;

    const auto bad_threads = parse_toml(R"toml(
[MODEL]
[MODEL.backend]
type = "onnx"
model_file_path = "a.onnx"
threads = 0
)toml");
    const toml::table* threads_section = bad_threads["MODEL"].as_table();
    ASSERT_NE(threads_section, nullptr);
    EXPECT_FALSE(jinq::models::backend::parse_backend_config(*threads_section, &config, &err));
    EXPECT_NE(err.find("threads"), std::string::npos) << err;

    const auto no_backend = parse_toml("[MODEL]\nkey = 1\n");
    const toml::table* no_backend_section = no_backend["MODEL"].as_table();
    ASSERT_NE(no_backend_section, nullptr);
    EXPECT_FALSE(jinq::models::backend::parse_backend_config(*no_backend_section, &config, &err));
    EXPECT_NE(err.find("backend"), std::string::npos) << err;
}

TEST(BackendSession, UnknownBackendAndMissingFileFail) {
    std::string err;
    EXPECT_EQ(InferenceSession::create(make_config("ncnn", "x.ncnn"), &err), nullptr);
    EXPECT_NE(err.find("unknown backend"), std::string::npos) << err;

    err.clear();
    EXPECT_EQ(InferenceSession::create(make_config("mnn", "weights/not/exist.mnn"), &err),
              nullptr);
    EXPECT_NE(err.find("not exist"), std::string::npos) << err;

    err.clear();
    EXPECT_EQ(InferenceSession::create(make_config("onnx", "weights/not/exist.onnx"), &err),
              nullptr);
    EXPECT_NE(err.find("not exist"), std::string::npos) << err;

    err.clear();
    EXPECT_EQ(InferenceSession::create(make_config("tensorrt", "weights/not/exist.engine"), &err),
              nullptr);
    EXPECT_NE(err.find("not exist"), std::string::npos) << err;
}

TEST(MnnSession, InitAndRunMobilenetv2) {
    if (!file_exists(kMnnModel)) {
        GTEST_SKIP() << "mnn weights not available";
    }
    std::string err;
    auto session = InferenceSession::create(make_config("mnn", kMnnModel), &err);
    ASSERT_NE(session, nullptr) << err;
    ASSERT_EQ(session->inputs().size(), 1u) << info_names(session->inputs());
    ASSERT_EQ(session->outputs().size(), 1u) << info_names(session->outputs());
    const auto& input_info = session->inputs().front();
    const auto& output_info = session->outputs().front();
    EXPECT_EQ(input_info.dtype, jinq::models::backend::DType::F32);
    EXPECT_FALSE(input_info.dynamic);
    EXPECT_EQ(output_info.dtype, jinq::models::backend::DType::F32);

    std::vector<NamedTensor> inputs;
    NamedTensor input;
    input.name = input_info.name;
    input.tensor = Tensor::make<float>(input_info.shape);
    inputs.push_back(std::move(input));

    std::vector<NamedTensor> outputs;
    ASSERT_EQ(session->run(inputs, outputs), StatusCode::OK);
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs.front().name, output_info.name);
    EXPECT_GT(outputs.front().tensor.element_count(), 0);

    // dtype mismatch must be rejected
    auto bad_inputs = inputs;
    bad_inputs.front().tensor = Tensor::make<int32_t>(input_info.shape);
    EXPECT_EQ(session->run(bad_inputs, outputs), StatusCode::MODEL_RUN_SESSION_FAILED);

    // unknown input names must be rejected
    auto unknown_inputs = inputs;
    unknown_inputs.front().name = "no_such_input";
    EXPECT_EQ(session->run(unknown_inputs, outputs), StatusCode::MODEL_RUN_SESSION_FAILED);

    // missing inputs must be rejected
    std::vector<NamedTensor> empty_inputs;
    EXPECT_EQ(session->run(empty_inputs, outputs), StatusCode::MODEL_RUN_SESSION_FAILED);
}

TEST(OrtSession, InitAndRunDdpmUnet) {
    if (!file_exists(kOnnxModel)) {
        GTEST_SKIP() << "onnx weights not available";
    }
    std::string err;
    auto session = InferenceSession::create(make_config("onnx", kOnnxModel), &err);
    ASSERT_NE(session, nullptr) << err;
    const auto& xt_info = find_info(session->inputs(), "xt");
    const auto& t_info = find_info(session->inputs(), "t");
    EXPECT_EQ(xt_info.dtype, jinq::models::backend::DType::F32);
    EXPECT_EQ(t_info.dtype, jinq::models::backend::DType::I64);
    ASSERT_EQ(session->outputs().size(), 1u);

    std::vector<NamedTensor> inputs;
    NamedTensor xt;
    xt.name = "xt";
    xt.tensor = Tensor::make<float>(xt_info.shape);
    inputs.push_back(std::move(xt));
    NamedTensor timestep;
    timestep.name = "t";
    timestep.tensor = Tensor::make<int64_t>(t_info.shape);
    timestep.tensor.data<int64_t>()[0] = 42;
    inputs.push_back(std::move(timestep));

    std::vector<NamedTensor> outputs;
    ASSERT_EQ(session->run(inputs, outputs), StatusCode::OK);
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs.front().tensor.element_count(), xt_info.shape[1] * xt_info.shape[2] *
                                                            xt_info.shape[3]);

    // int32 timestep must be rejected (the exported model expects int64)
    auto bad_inputs = inputs;
    bad_inputs[1].tensor = Tensor::make<int32_t>(t_info.shape);
    EXPECT_EQ(session->run(bad_inputs, outputs), StatusCode::MODEL_RUN_SESSION_FAILED);
}

TEST(TrtSession, ResolvesLightglueDynamicOutputs) {
    if (!file_exists(kLightglueExtractor) || !file_exists(kFeatureImage)) {
        GTEST_SKIP() << "lightglue weights or test image not available";
    }
    std::string err;
    auto session =
        InferenceSession::create(make_config("tensorrt", kLightglueExtractor, "cuda"), &err);
    if (session == nullptr) {
        GTEST_SKIP() << "tensorrt/gpu unavailable: " << err;
    }

    const auto& image_info = find_info(session->inputs(), "image");
    EXPECT_TRUE(image_info.dynamic);
    cv::Mat image = cv::imread(kFeatureImage, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty());
    cv::resize(image, image, cv::Size(128, 96));
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    image.convertTo(image, CV_32FC1, 1.0 / 255.0);

    NamedTensor named;
    named.name = image_info.name;
    named.tensor = Tensor::make<float>({1, 1, image.rows, image.cols});
    ASSERT_EQ(image.total() * image.elemSize(), named.tensor.byte_size());
    std::memcpy(named.tensor.buffer.data(), image.data, named.tensor.byte_size());
    std::vector<NamedTensor> inputs;
    inputs.push_back(std::move(named));

    std::vector<NamedTensor> outputs;
    ASSERT_EQ(session->run(inputs, outputs), StatusCode::OK);
    const auto find_output = [&outputs](const std::string& name) {
        const auto iter = std::find_if(outputs.begin(), outputs.end(),
                                       [&name](const NamedTensor& item) {
                                           return item.name == name;
                                       });
        return iter == outputs.end() ? nullptr : &*iter;
    };
    const auto* keypoints = find_output("keypoints");
    const auto* scores = find_output("scores");
    const auto* descriptors = find_output("descriptors");
    ASSERT_NE(keypoints, nullptr);
    ASSERT_NE(scores, nullptr);
    ASSERT_NE(descriptors, nullptr);
    ASSERT_GT(scores->tensor.element_count(), 0);
    EXPECT_TRUE(keypoints->tensor.shape_is_concrete());
    EXPECT_TRUE(scores->tensor.shape_is_concrete());
    EXPECT_TRUE(descriptors->tensor.shape_is_concrete());
    EXPECT_EQ(keypoints->tensor.element_count(), scores->tensor.element_count() * 2);
    EXPECT_EQ(descriptors->tensor.element_count(), scores->tensor.element_count() * 256);
}

TEST(TrtSession, InitAndRunYolov8) {
    if (!file_exists(kTrtModel)) {
        GTEST_SKIP() << "tensorrt engine not available";
    }
    std::string err;
    auto session = InferenceSession::create(make_config("tensorrt", kTrtModel, "cuda"), &err);
    if (session == nullptr) {
        GTEST_SKIP() << "tensorrt/gpu unavailable: " << err;
    }
    ASSERT_GE(session->inputs().size(), 1u);
    ASSERT_GE(session->outputs().size(), 1u);
    const auto& input_info = session->inputs().front();
    EXPECT_EQ(input_info.dtype, jinq::models::backend::DType::F32);
    EXPECT_EQ(input_info.shape.size(), 4u);

    std::vector<NamedTensor> inputs;
    NamedTensor input;
    input.name = input_info.name;
    // Tensor::make requires a concrete shape: the yolov8 engine carries a
    // dynamic batch axis (-1), resolve dynamic dims to the smallest valid
    // concrete value (batch 1, inside the profile range)
    auto concrete_shape = input_info.shape;
    for (auto& dim : concrete_shape) {
        if (dim <= 0) {
            dim = 1;
        }
    }
    input.tensor = Tensor::make<float>(concrete_shape);
    inputs.push_back(std::move(input));

    std::vector<NamedTensor> outputs;
    ASSERT_EQ(session->run(inputs, outputs), StatusCode::OK);
    ASSERT_EQ(outputs.size(), session->outputs().size());
    EXPECT_EQ(outputs.front().tensor.dtype, jinq::models::backend::DType::F32);
    EXPECT_GT(outputs.front().tensor.element_count(), 0);

    // a wrong input shape (outside the profile) must be rejected
    auto bad_inputs = inputs;
    bad_inputs.front().tensor = Tensor::make<float>({1, 3, 123, 456});
    const auto status = session->run(bad_inputs, outputs);
    EXPECT_TRUE(status == StatusCode::MODEL_RUN_SESSION_FAILED ||
                status == StatusCode::TRT_CUDA_ERROR)
        << "status: " << static_cast<int>(status);

    // dtype mismatch must be rejected
    auto bad_dtype = inputs;
    bad_dtype.front().tensor = Tensor::make<int32_t>(concrete_shape);
    EXPECT_EQ(session->run(bad_dtype, outputs), StatusCode::MODEL_RUN_SESSION_FAILED);
}

TEST(MultiBackend, CoexistInOneProcess) {
    if (!file_exists(kMnnModel) || !file_exists(kOnnxModel)) {
        GTEST_SKIP() << "weights not available";
    }
    std::string err;
    auto mnn_session = InferenceSession::create(make_config("mnn", kMnnModel), &err);
    ASSERT_NE(mnn_session, nullptr) << err;
    auto ort_session = InferenceSession::create(make_config("onnx", kOnnxModel), &err);
    ASSERT_NE(ort_session, nullptr) << err;

    std::vector<NamedTensor> outputs;
    const auto& mnn_input = mnn_session->inputs().front();
    std::vector<NamedTensor> mnn_inputs;
    NamedTensor mnn_tensor;
    mnn_tensor.name = mnn_input.name;
    mnn_tensor.tensor = Tensor::make<float>(mnn_input.shape);
    mnn_inputs.push_back(std::move(mnn_tensor));
    EXPECT_EQ(mnn_session->run(mnn_inputs, outputs), StatusCode::OK);

    const auto& xt_info = find_info(ort_session->inputs(), "xt");
    const auto& t_info = find_info(ort_session->inputs(), "t");
    std::vector<NamedTensor> ort_inputs;
    NamedTensor xt;
    xt.name = "xt";
    xt.tensor = Tensor::make<float>(xt_info.shape);
    ort_inputs.push_back(std::move(xt));
    NamedTensor timestep;
    timestep.name = "t";
    timestep.tensor = Tensor::make<int64_t>(t_info.shape);
    ort_inputs.push_back(std::move(timestep));
    EXPECT_EQ(ort_session->run(ort_inputs, outputs), StatusCode::OK);

    if (file_exists(kTrtModel)) {
        auto trt_session = InferenceSession::create(make_config("tensorrt", kTrtModel, "cuda"), &err);
        if (trt_session != nullptr) {
            const auto& trt_input = trt_session->inputs().front();
            std::vector<NamedTensor> trt_inputs;
            NamedTensor trt_tensor;
            trt_tensor.name = trt_input.name;
            // the yolov8 engine has a dynamic batch axis: resolve -1 dims
            // to a concrete batch-1 shape before building the host tensor
            auto trt_shape = trt_input.shape;
            for (auto& dim : trt_shape) {
                if (dim <= 0) {
                    dim = 1;
                }
            }
            trt_tensor.tensor = Tensor::make<float>(trt_shape);
            trt_inputs.push_back(std::move(trt_tensor));
            EXPECT_EQ(trt_session->run(trt_inputs, outputs), StatusCode::OK);
        } else {
            std::cout << "trt unavailable in coexistence test: " << err << std::endl;
        }
    }
}
