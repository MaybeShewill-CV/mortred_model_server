/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: session.h
* Date: 26-8-20
************************************************/

#ifndef MORTRED_MODELS_BACKEND_SESSION_H
#define MORTRED_MODELS_BACKEND_SESSION_H

#include <memory>
#include <string>
#include <vector>

#include "common/status_code.h"
#include "models/backend/backend_config.h"
#include "models/backend/tensor.h"

namespace jinq {
namespace models {
namespace backend {
using jinq::common::StatusCode;

/***
 * Metadata of one model io tensor. shape may contain -1 dims for dynamic
 * axes (dynamic == true); concrete shapes are required at run time.
 */
struct TensorInfo {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    bool dynamic = false;

    std::string to_string() const {
        std::string out = name + ":";
        out += dtype_to_string(dtype);
        out += shape_to_string(shape);
        if (dynamic) {
            out += " (dynamic)";
        }
        return out;
    }
};

/***
 * Backend-agnostic inference session. One session per worker (sessions are
 * NOT required to be thread safe), all resources are owned through RAII.
 */
class InferenceSession {
  public:
    virtual ~InferenceSession() = default;

    InferenceSession() = default;
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    /***
     * build a session from a parsed backend config. Returns nullptr and fills
     * err on failure; never throws for expected failures (bad file, unknown
     * io names, unsupported dtype).
     */
    static std::unique_ptr<InferenceSession> create(const BackendConfig& config,
                                                    std::string* err = nullptr);

    virtual const std::vector<TensorInfo>& inputs() const = 0;
    virtual const std::vector<TensorInfo>& outputs() const = 0;

    /***
     * run one inference. Input tensors are validated against the model io
     * (name / dtype / concrete shape); dynamic shapes are set per run. Output
     * buffers are reused inside the session, the returned tensors own a host
     * copy with the concrete run-time shapes.
     */
    virtual StatusCode run(const std::vector<NamedTensor>& inputs,
                                         std::vector<NamedTensor>& outputs) = 0;
};

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_SESSION_H
