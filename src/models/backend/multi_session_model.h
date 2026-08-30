/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: multi_session_model.h
 * Date: 26-8-31
 *
 * Declarative lifecycle for multi-engine models.
 *
 * BackendCvModel already offers make_session("<key>_backend") and the
 * run_sessions() hook, but every multi-engine model still hand-writes the same
 * sequence: create each session, validate each one, and reset everything on the
 * first failure. This base class owns that sequence so a model only declares
 * what sessions it needs.
 *
 * A model declares its engines through a static sessions() function and
 * inherits from MultiSessionModel<Derived, INPUT, OUTPUT>:
 *
 *   static std::vector<SessionSpec> sessions() {
 *       return {{"visual", "visual_backend",
 *                IoSpec::input("input").f32().rank(4).nchw().channels(3).static_shape(),
 *                IoSpec::output("output").f32().rank(2).static_shape()},
 *               {"text", "text_backend",
 *                IoSpec::input("input").i32().rank(2).static_shape(),
 *                IoSpec::output("output").f32().rank(2).static_shape()}};
 *   }
 *
 *   StatusCode on_init(const toml::table &params) override {
 *       return init_sessions();   // creates + validates + resets on failure
 *   }
 *
 *   auto &visual = session("visual");  // non-null after a successful init
 *
 * SessionSpec holds strings, so sessions() is a plain function rather than a
 * constexpr array. The base class deliberately does NOT orchestrate the runs:
 * the order in which engines execute and how their outputs combine is the
 * model's own logic, and hiding it behind a graph would make debugging harder.
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_MULTI_SESSION_MODEL_H
#define MORTRED_MODELS_BACKEND_MULTI_SESSION_MODEL_H

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "common/status_code.h"
#include "models/backend/backend_cv_model.h"
#include "models/backend/model_runtime.h"
#include "models/backend/session.h"

namespace jinq {
namespace models {
namespace backend {

using jinq::common::StatusCode;

/*** reusable description of one session input or output ***/
struct IoSpec {
    std::string name;
    DType dtype = DType::F32;
    size_t expected_rank = 0;
    bool has_layout = false;
    bool is_nchw = false;
    int64_t channel_count = -1;
    bool require_static = false;

    static IoSpec input(std::string name) { return IoSpec{std::move(name)}; }
    static IoSpec output(std::string name) { return IoSpec{std::move(name)}; }

    IoSpec &&f32() && {
        dtype = DType::F32;
        return std::move(*this);
    }
    IoSpec &&i32() && {
        dtype = DType::I32;
        return std::move(*this);
    }
    IoSpec &&of(DType value) && {
        dtype = value;
        return std::move(*this);
    }
    IoSpec &&rank(size_t value) && {
        expected_rank = value;
        return std::move(*this);
    }
    IoSpec &&nchw() && {
        has_layout = true;
        is_nchw = true;
        return std::move(*this);
    }
    IoSpec &&nhwc() && {
        has_layout = true;
        is_nchw = false;
        return std::move(*this);
    }
    IoSpec &&channels(int64_t value) && {
        channel_count = value;
        return std::move(*this);
    }
    IoSpec &&static_shape() && {
        require_static = true;
        return std::move(*this);
    }
};

/*** one named engine: which config table backs it and what IO it must expose ***/
struct SessionSpec {
    std::string name;
    std::string backend_key;
    IoSpec input;
    IoSpec output;
};

/*** applies an IoSpec through SessionIoValidator ***/
inline RuntimeStatus check_io(const InferenceSession &session, const IoSpec &spec, bool is_output, const std::string &owner) {
    // an empty name means "create the engine but let the model validate its
    // IO": used when the contract is genuinely model-specific (optional or
    // alternative tensors) rather than a fixed input/output pair
    if (spec.name.empty()) {
        return {StatusCode::OK, {}};
    }
    auto validator = is_output ? SessionIoValidator(session).output(spec.name) : SessionIoValidator(session).input(spec.name);
    validator.dtype(spec.dtype);
    if (spec.expected_rank != 0) {
        validator.rank(spec.expected_rank);
    }
    if (spec.has_layout) {
        if (spec.is_nchw) {
            validator.nchw();
        } else {
            validator.nhwc();
        }
    }
    if (spec.channel_count > 0) {
        validator.channels(spec.channel_count);
    }
    if (spec.require_static) {
        validator.static_shape();
    }
    const auto result = validator.validate();
    if (!result.ok()) {
        return {result.status, owner + " " + (is_output ? "output" : "input") + " [" + spec.name + "]: " + result.error};
    }
    return {StatusCode::OK, {}};
}

/***
 * Owns the create / validate / reset-on-failure sequence for a fixed set of
 * engines. Derived declares `static std::vector<SessionSpec> sessions()` and
 * calls init_sessions() from its on_init().
 */
template <typename Derived, typename INPUT, typename OUTPUT> class MultiSessionModel : public BackendCvModel<INPUT, OUTPUT> {
  public:
    ~MultiSessionModel() override = default;

    /*** creates and validates every declared session; resets all on failure ***/
    StatusCode init_sessions() {
        for (const auto &spec : Derived::sessions()) {
            auto created = create_session(spec);
            if (created == nullptr) {
                LOG(ERROR) << "create session [" << spec.name << "] from [" << spec.backend_key << "] failed";
                reset_sessions();
                return StatusCode::MODEL_INIT_FAILED;
            }
            for (const auto &check : {check_io(*created, spec.input, false, spec.name), check_io(*created, spec.output, true, spec.name)}) {
                if (check.status != StatusCode::OK) {
                    LOG(ERROR) << check.error;
                    reset_sessions();
                    return check.status;
                }
            }
            sessions_.emplace(spec.name, std::move(created));
        }
        return StatusCode::OK;
    }

    /*** accessor; nullptr only before init or after a failed init ***/
    InferenceSession *session(const std::string &name) const {
        const auto found = sessions_.find(name);
        return found == sessions_.end() ? nullptr : found->second.get();
    }

    void reset_sessions() { sessions_.clear(); }

  protected:
    explicit MultiSessionModel(std::string section_name) : BackendCvModel<INPUT, OUTPUT>(std::move(section_name)) {}

    /*** session factory; overridable so tests can inject fakes without model files ***/
    virtual std::unique_ptr<InferenceSession> create_session(const SessionSpec &spec) const { return this->make_session(spec.backend_key); }

  private:
    std::map<std::string, std::unique_ptr<InferenceSession>> sessions_;
};

} // namespace backend
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_BACKEND_MULTI_SESSION_MODEL_H
