#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/backend/backend_cv_model.h"
#include "models/backend/inference_context.h"
#include "models/backend/multi_session_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/feature_point/lightglue.h"
#include "models/model_io_define.h"
#include "models/segment_anything/sam_prediction/sam_predictor.h"

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::InferenceSession;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::TensorInfo;

namespace {

/*** stand-in session: reports a fixed IO table and never actually runs ***/
class FakeSession final : public InferenceSession {
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

struct FakeInput {};
struct FakeOutput {};

/*** a two-engine model used to exercise the base class; backend keys match CLIP ***/
class FakeClipModel final : public jinq::models::backend::MultiSessionModel<FakeClipModel, FakeInput, FakeOutput> {
  public:
    FakeClipModel() : MultiSessionModel("FAKE_CLIP") {}

    static std::vector<jinq::models::backend::SessionSpec> sessions() {
        return {
            {"visual", "visual_backend", jinq::models::backend::IoSpec::input("input").f32().rank(4).nchw().channels(3).static_shape(),
             jinq::models::backend::IoSpec::output("output").f32().rank(2).static_shape()},
            {"text", "text_backend", jinq::models::backend::IoSpec::input("input").i32().rank(2).static_shape(),
             jinq::models::backend::IoSpec::output("output").f32().rank(2).static_shape()},
        };
    }

    /*** inject fake engines in declaration order; the base class takes ownership ***/
    void enqueue_session(std::unique_ptr<InferenceSession> session) { pending_.push(std::move(session)); }

    size_t pending_count() const { return pending_.size(); }

  protected:
    // postprocess is the only pure virtual hook; this test only exercises
    // session lifecycle, so it is never called
    StatusCode postprocess(const std::vector<NamedTensor> &, const jinq::models::backend::InferenceContext &, FakeOutput &) override {
        return StatusCode::OK;
    }

    std::unique_ptr<InferenceSession> create_session(const jinq::models::backend::SessionSpec &) const override {
        auto &queue = const_cast<std::queue<std::unique_ptr<InferenceSession>> &>(pending_);
        if (queue.empty()) {
            return nullptr;
        }
        auto session = std::move(queue.front());
        queue.pop();
        return session;
    }

  private:
    std::queue<std::unique_ptr<InferenceSession>> pending_;
};

std::unique_ptr<FakeSession> visual_engine() {
    return std::make_unique<FakeSession>(std::vector<TensorInfo>{{"input", DType::F32, {1, 3, 8, 8}, false}},
                                         std::vector<TensorInfo>{{"output", DType::F32, {1, 4}, false}});
}

std::unique_ptr<FakeSession> text_engine() {
    return std::make_unique<FakeSession>(std::vector<TensorInfo>{{"input", DType::I32, {1, 77}, false}},
                                         std::vector<TensorInfo>{{"output", DType::F32, {1, 4}, false}});
}

toml::table parse_toml(const std::string &content) {
    auto parsed = toml::parse(content);
    if (!parsed) {
        ADD_FAILURE() << "fixture toml parse failed";
        return toml::table{};
    }
    return std::move(parsed).table();
}

StatusCode seed_both_backends(FakeClipModel &model) {
    return model.init(parse_toml(R"toml(
[FAKE_CLIP]
[FAKE_CLIP.visual_backend]
type = "mnn"
model_file_path = "unused.mnn"
[FAKE_CLIP.text_backend]
type = "mnn"
model_file_path = "unused.mnn"
)toml"));
}

std::string capture_init_log(const std::function<StatusCode()> &action, StatusCode *status_out) {
    FLAGS_logtostderr = true;
    testing::internal::CaptureStderr();
    *status_out = action();
    return testing::internal::GetCapturedStderr();
}

} // namespace

TEST(MultiSessionModel, DeclaredSessionsAreCreatedAndAccessible) {
    FakeClipModel model;
    ASSERT_EQ(seed_both_backends(model), StatusCode::OK);
    model.enqueue_session(visual_engine());
    model.enqueue_session(text_engine());

    EXPECT_EQ(model.init_sessions(), StatusCode::OK);
    EXPECT_NE(model.session("visual"), nullptr);
    EXPECT_NE(model.session("text"), nullptr);

    model.reset_sessions();
    EXPECT_EQ(model.session("visual"), nullptr);
    EXPECT_EQ(model.session("text"), nullptr);
}

TEST(MultiSessionModel, IoMismatchFailsInitAndClearsSessions) {
    FakeClipModel model;
    ASSERT_EQ(seed_both_backends(model), StatusCode::OK);
    // wrong dtype: the spec asks for f32, the session reports i32
    model.enqueue_session(std::make_unique<FakeSession>(std::vector<TensorInfo>{{"input", DType::I32, {1, 3, 8, 8}, false}},
                                                        std::vector<TensorInfo>{{"output", DType::F32, {1, 4}, false}}));
    model.enqueue_session(text_engine());

    EXPECT_NE(model.init_sessions(), StatusCode::OK);
    EXPECT_EQ(model.session("visual"), nullptr);
    EXPECT_EQ(model.session("text"), nullptr);
}

TEST(MultiSessionModel, MissingEngineFailsInitAndClearsSessions) {
    FakeClipModel model;
    ASSERT_EQ(seed_both_backends(model), StatusCode::OK);
    model.enqueue_session(visual_engine());
    // the text engine is deliberately not provided

    EXPECT_NE(model.init_sessions(), StatusCode::OK);
    EXPECT_EQ(model.session("visual"), nullptr);
    EXPECT_EQ(model.session("text"), nullptr);
}

TEST(MultiSessionModel, UndeclaredNameIsNullptrNotAnError) {
    FakeClipModel model;
    ASSERT_EQ(seed_both_backends(model), StatusCode::OK);
    model.enqueue_session(visual_engine());
    model.enqueue_session(text_engine());
    ASSERT_EQ(model.init_sessions(), StatusCode::OK);

    EXPECT_EQ(model.session("no_such_engine"), nullptr);
}

TEST(MultiSessionModel, ClipMissingTextBackendFailsNamed) {
    FakeClipModel model;
    ASSERT_EQ(model.init(parse_toml(R"toml(
[FAKE_CLIP]
[FAKE_CLIP.visual_backend]
type = "mnn"
model_file_path = "unused.mnn"
)toml")),
              StatusCode::OK);
    model.enqueue_session(visual_engine());
    model.enqueue_session(text_engine());
    const size_t pending_before = model.pending_count();

    StatusCode status = StatusCode::OK;
    const std::string log = capture_init_log([&] { return model.init_sessions(); }, &status);

    EXPECT_EQ(status, StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_NE(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(log.find("text_backend"), std::string::npos) << log;
    EXPECT_EQ(model.session("visual"), nullptr);
    EXPECT_EQ(model.session("text"), nullptr);
    EXPECT_EQ(model.pending_count(), pending_before);
}

TEST(MultiSessionModel, ClipWrongSessionBindingFailsNamed) {
    FakeClipModel model;
    // text_backend exists but is a string alias to another session, not a table
    ASSERT_EQ(model.init(parse_toml(R"toml(
[FAKE_CLIP]
[FAKE_CLIP.visual_backend]
type = "mnn"
model_file_path = "unused.mnn"
text_backend = "visual_backend"
)toml")),
              StatusCode::OK);
    model.enqueue_session(visual_engine());
    model.enqueue_session(text_engine());
    const size_t pending_before = model.pending_count();

    StatusCode status = StatusCode::OK;
    const std::string log = capture_init_log([&] { return model.init_sessions(); }, &status);

    EXPECT_EQ(status, StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_NE(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(log.find("text_backend"), std::string::npos) << log;
    EXPECT_EQ(model.session("visual"), nullptr);
    EXPECT_EQ(model.session("text"), nullptr);
    EXPECT_EQ(model.pending_count(), pending_before);
}

TEST(MultiSessionModel, LightGlueMissingMatcherBackendFailsNamed) {
    using MatchIn = jinq::models::io_define::common_io::pair_mat_input;
    using MatchOut = jinq::models::io_define::feature_point::std_feature_point_match_output;
    jinq::models::feature_point::LightGlue<MatchIn, MatchOut> model;

    StatusCode status = StatusCode::OK;
    const std::string log = capture_init_log(
        [&] {
            return model.init(parse_toml(R"toml(
[LIGHTGLUE]
[LIGHTGLUE.extractor_backend]
type = "mnn"
model_file_path = "/nonexistent_extractor.mnn"
[LIGHTGLUE.params]
long_side_length = 512
)toml"));
        },
        &status);

    EXPECT_EQ(status, StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_NE(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(log.find("matcher_backend"), std::string::npos) << log;
    EXPECT_FALSE(model.is_successfully_initialized());
}

TEST(MultiSessionModel, LightGlueMissingExtractorBackendFailsNamed) {
    using MatchIn = jinq::models::io_define::common_io::pair_mat_input;
    using MatchOut = jinq::models::io_define::feature_point::std_feature_point_match_output;
    jinq::models::feature_point::LightGlue<MatchIn, MatchOut> model;

    StatusCode status = StatusCode::OK;
    const std::string log = capture_init_log(
        [&] {
            return model.init(parse_toml(R"toml(
[LIGHTGLUE]
[LIGHTGLUE.matcher_backend]
type = "mnn"
model_file_path = "/nonexistent_matcher.mnn"
[LIGHTGLUE.params]
long_side_length = 512
)toml"));
        },
        &status);

    EXPECT_EQ(status, StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(log.find("extractor_backend"), std::string::npos) << log;
    EXPECT_FALSE(model.is_successfully_initialized());
}

TEST(MultiSessionModel, SamMissingDecoderBackendFailsNamed) {
    using SamIn = jinq::models::io_define::segment_anything::sam_prompt_input;
    using SamOut = jinq::models::io_define::segment_anything::std_sam_prompt_output;
    jinq::models::segment_anything::SamPredictor<SamIn, SamOut> model;

    StatusCode status = StatusCode::OK;
    const std::string log = capture_init_log(
        [&] {
            return model.init(parse_toml(R"toml(
[SAM_PREDICTOR]
[SAM_PREDICTOR.encoder_backend]
type = "mnn"
model_file_path = "/nonexistent_encoder.mnn"
)toml"));
        },
        &status);

    EXPECT_EQ(status, StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_NE(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(log.find("decoder_backend"), std::string::npos) << log;
    EXPECT_FALSE(model.is_successfully_initialized());
}

TEST(MultiSessionModel, SamMissingEncoderBackendFailsNamed) {
    using SamIn = jinq::models::io_define::segment_anything::sam_prompt_input;
    using SamOut = jinq::models::io_define::segment_anything::std_sam_prompt_output;
    jinq::models::segment_anything::SamPredictor<SamIn, SamOut> model;

    StatusCode status = StatusCode::OK;
    const std::string log = capture_init_log(
        [&] {
            return model.init(parse_toml(R"toml(
[SAM_PREDICTOR]
[SAM_PREDICTOR.decoder_backend]
type = "mnn"
model_file_path = "/nonexistent_decoder.mnn"
)toml"));
        },
        &status);

    EXPECT_EQ(status, StatusCode::MODEL_INIT_FAILED);
    EXPECT_NE(log.find("encoder_backend"), std::string::npos) << log;
    EXPECT_FALSE(model.is_successfully_initialized());
}
