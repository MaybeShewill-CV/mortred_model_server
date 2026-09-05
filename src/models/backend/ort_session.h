/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: ort_session.h
* Date: 26-8-20
************************************************/

#ifndef MORTRED_MODELS_BACKEND_ORT_SESSION_H
#define MORTRED_MODELS_BACKEND_ORT_SESSION_H

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime/onnxruntime_cxx_api.h"

#include "common/status_code.h"
#include "models/backend/backend_config.h"
#include "models/backend/session.h"

namespace jinq {
namespace models {
namespace backend {
using jinq::common::StatusCode;

/***
 * RAII ONNX Runtime inference session. Env/SessionOptions/Session are owned
 * by value/smart pointer; device=gpu enables the CUDA execution provider.
 */
class OrtSession : public InferenceSession {
  public:
    OrtSession() = default;
    ~OrtSession() override = default;

    OrtSession(const OrtSession&) = delete;
    OrtSession& operator=(const OrtSession&) = delete;

    StatusCode init(const BackendConfig& config, std::string* err = nullptr);

    const std::vector<TensorInfo>& inputs() const override {
        return _m_input_infos;
    }

    const std::vector<TensorInfo>& outputs() const override {
        return _m_output_infos;
    }

    StatusCode run(const std::vector<NamedTensor>& inputs,
                                 std::vector<NamedTensor>& outputs) override;

  private:
    Ort::Env _m_env{nullptr};
    Ort::SessionOptions _m_session_options;
    std::unique_ptr<Ort::Session> _m_session;
    std::vector<TensorInfo> _m_input_infos;
    std::vector<TensorInfo> _m_output_infos;
    std::vector<std::string> _m_input_names;
    std::vector<std::string> _m_output_names;
    std::vector<const char*> _m_input_name_ptrs;
    std::vector<const char*> _m_output_name_ptrs;
    std::string _m_model_file_path;
};

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_ORT_SESSION_H
