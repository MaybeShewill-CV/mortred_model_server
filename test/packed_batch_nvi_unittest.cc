/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: packed_batch_nvi_unittest.cc
 * Date: 26-9-8
 ************************************************/

// Packed run_batch: a throw is a session failure. Every item_status must
// become MODEL_RUN_SESSION_FAILED; none may stay OK (HTTP would lie).

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <toml/toml.hpp>

#include "common/status_code.h"
#include "models/backend/backend_cv_model.h"
#include "models/io/common_input.h"

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::InferenceSession;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorInfo;
using jinq::models::io_define::common_io::mat_input;

// tests-only: do not link libmodels (it already defines create).
namespace jinq {
namespace models {
namespace backend {

enum class FakeSessionMode { Echo, Throw };

FakeSessionMode g_fake_session_mode = FakeSessionMode::Echo;

class EchoOrThrowSession final : public InferenceSession {
  public:
    const std::vector<TensorInfo> &inputs() const override {
        static const std::vector<TensorInfo> kInputs{{"input", DType::F32, {1, 1, 1, 3}, true}};
        return kInputs;
    }
    const std::vector<TensorInfo> &outputs() const override {
        static const std::vector<TensorInfo> kOutputs{{"input", DType::F32, {1, 1, 1, 3}, true}};
        return kOutputs;
    }
    StatusCode run(const std::vector<NamedTensor> &inputs, std::vector<NamedTensor> &outputs) override {
        if (g_fake_session_mode == FakeSessionMode::Throw) {
            throw std::runtime_error("synthetic packed session failure");
        }
        outputs = inputs;
        return StatusCode::OK;
    }
};

std::unique_ptr<InferenceSession> InferenceSession::create(const BackendConfig &config, std::string *err) {
    (void)config;
    if (err != nullptr) {
        err->clear();
    }
    return std::make_unique<EchoOrThrowSession>();
}

} // namespace backend
} // namespace models
} // namespace jinq

namespace {

toml::table parse_toml(const std::string &content) {
    auto parsed = toml::parse(content);
    if (!parsed) {
        ADD_FAILURE() << "fixture toml parse failed";
        return {};
    }
    return std::move(parsed).table();
}

toml::table packed_throw_cfg() {
    return parse_toml(R"toml(
[PACKED_THROW]
[PACKED_THROW.backend]
type = "onnx"
model_file_path = "unused.onnx"
device = "cpu"
)toml");
}

class PackedThrowModel : public jinq::models::backend::BackendCvModel<mat_input, int> {
  public:
    PackedThrowModel() : BackendCvModel("PACKED_THROW") {}

    int postprocess_calls = 0;
    bool throw_on_second_postprocess = false;
    bool fail_second_with_status = false;

  protected:
    std::vector<NamedTensor> preprocess(const cv::Mat &image) override {
        (void)image;
        return {NamedTensor{"input", Tensor::make<float>({1, 1, 1, 3})}};
    }

    StatusCode postprocess(const std::vector<NamedTensor> &, const jinq::models::backend::InferenceContext &,
                           int &output) override {
        ++postprocess_calls;
        if (throw_on_second_postprocess && postprocess_calls >= 2) {
            throw std::runtime_error("synthetic packed postprocess failure");
        }
        if (fail_second_with_status && postprocess_calls >= 2) {
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        output = 42;
        return StatusCode::OK;
    }
};

std::vector<mat_input> two_tiny_images() {
    std::vector<mat_input> in(2);
    in[0].input_image = cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    in[1].input_image = cv::Mat(2, 2, CV_8UC3, cv::Scalar(4, 5, 6));
    return in;
}

void expect_all_session_failed(const StatusCode aggregate, const std::vector<StatusCode> &item_status,
                               size_t n) {
    EXPECT_EQ(aggregate, StatusCode::MODEL_RUN_SESSION_FAILED);
    ASSERT_EQ(item_status.size(), n);
    for (size_t idx = 0; idx < item_status.size(); ++idx) {
        EXPECT_EQ(item_status[idx], StatusCode::MODEL_RUN_SESSION_FAILED) << "item " << idx;
        EXPECT_NE(item_status[idx], StatusCode::OK) << "item " << idx << " must not stay OK";
    }
}

} // namespace

TEST(packed_batch_nvi, postprocess_throw_marks_every_item_failed) {
    jinq::models::backend::g_fake_session_mode = jinq::models::backend::FakeSessionMode::Echo;
    PackedThrowModel model;
    model.throw_on_second_postprocess = true;
    ASSERT_EQ(model.init(packed_throw_cfg()), StatusCode::OK);

    auto in = two_tiny_images();
    std::vector<int> out;
    std::vector<StatusCode> item_status;
    const auto status = model.run_batch(in, out, item_status);
    expect_all_session_failed(status, item_status, 2u);
    ASSERT_EQ(out.size(), 2u);
}

TEST(packed_batch_nvi, session_throw_marks_every_item_failed) {
    jinq::models::backend::g_fake_session_mode = jinq::models::backend::FakeSessionMode::Throw;
    PackedThrowModel model;
    ASSERT_EQ(model.init(packed_throw_cfg()), StatusCode::OK);

    auto in = two_tiny_images();
    std::vector<int> out;
    std::vector<StatusCode> item_status;
    const auto status = model.run_batch(in, out, item_status);
    expect_all_session_failed(status, item_status, 2u);
    EXPECT_EQ(model.postprocess_calls, 0);
}

TEST(packed_batch_nvi, postprocess_status_return_stays_isolated) {
    jinq::models::backend::g_fake_session_mode = jinq::models::backend::FakeSessionMode::Echo;
    PackedThrowModel model;
    model.fail_second_with_status = true;
    ASSERT_EQ(model.init(packed_throw_cfg()), StatusCode::OK);

    auto in = two_tiny_images();
    std::vector<int> out;
    std::vector<StatusCode> item_status;
    const auto status = model.run_batch(in, out, item_status);
    EXPECT_EQ(status, StatusCode::MODEL_RUN_SESSION_FAILED);
    ASSERT_EQ(item_status.size(), 2u);
    EXPECT_EQ(item_status[0], StatusCode::OK) << "StatusCode return must not broadcast";
    EXPECT_EQ(item_status[1], StatusCode::MODEL_RUN_SESSION_FAILED);
}
