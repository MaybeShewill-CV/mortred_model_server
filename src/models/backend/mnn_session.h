/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: backend/mnn_session.h
 * Date: 2026-08-20
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_MNN_SESSION_H
#define MORTRED_MODELS_BACKEND_MNN_SESSION_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "MNN/Interpreter.hpp"

#include "common/status_code.h"
#include "models/backend/backend_config.h"
#include "models/backend/session.h"

namespace jinq {
namespace models {
namespace backend {

/***
 * RAII MNN inference session. Owns the interpreter and session, caches the
 * io tensors by name and re-fetches them after dynamic resizes (MNN may
 * reallocate tensor pointers on resizeSession).
 */
class MnnSession : public InferenceSession {
  public:
    MnnSession() = default;
    ~MnnSession() override;

    MnnSession(const MnnSession&) = delete;
    MnnSession& operator=(const MnnSession&) = delete;

    /*** build the session; returns non-OK status and fills err on failure */
    jinq::common::StatusCode init(const BackendConfig& config, std::string* err = nullptr);

    const std::vector<TensorInfo>& inputs() const override {
        return _m_input_infos;
    }

    const std::vector<TensorInfo>& outputs() const override {
        return _m_output_infos;
    }

    jinq::common::StatusCode run(const std::vector<NamedTensor>& inputs,
                                 std::vector<NamedTensor>& outputs) override;

  private:
    jinq::common::StatusCode refresh_io_tensors();

    std::unique_ptr<MNN::Interpreter> _m_interpreter;
    MNN::Session* _m_session = nullptr;
    std::map<std::string, MNN::Tensor*> _m_input_tensors;
    std::map<std::string, MNN::Tensor*> _m_output_tensors;
    std::map<std::string, MNN::Tensor::DimensionType> _m_input_dim_types;
    std::vector<TensorInfo> _m_input_infos;
    std::vector<TensorInfo> _m_output_infos;
    std::string _m_model_file_path;
};

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_MNN_SESSION_H
