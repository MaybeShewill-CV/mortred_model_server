#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "common/status_code.h"
#include "models/backend/backend_cv_model.h"
#include "models/backend/inference_context.h"
#include "models/backend/multi_session_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"

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

/*** a two-engine model used to exercise the base class ***/
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

} // namespace

TEST(MultiSessionModel, DeclaredSessionsAreCreatedAndAccessible) {
    FakeClipModel model;
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
    model.enqueue_session(visual_engine());
    // the text engine is deliberately not provided

    EXPECT_NE(model.init_sessions(), StatusCode::OK);
    EXPECT_EQ(model.session("visual"), nullptr);
    EXPECT_EQ(model.session("text"), nullptr);
}

TEST(MultiSessionModel, UndeclaredNameIsNullptrNotAnError) {
    FakeClipModel model;
    model.enqueue_session(visual_engine());
    model.enqueue_session(text_engine());
    ASSERT_EQ(model.init_sessions(), StatusCode::OK);

    EXPECT_EQ(model.session("no_such_engine"), nullptr);
}
