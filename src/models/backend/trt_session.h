/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: backend/trt_session.h
 * Date: 2026-08-20
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_TRT_SESSION_H
#define MORTRED_MODELS_BACKEND_TRT_SESSION_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "common/status_code.h"
#include "models/backend/backend_config.h"
#include "models/backend/session.h"

namespace jinq {
namespace models {
namespace backend {

namespace trt_detail {

class SessionLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
};

}  // namespace trt_detail

/***
 * RAII TensorRT inference session (tensor-address / enqueueV3 API, compatible
 * with TensorRT 8.6 and 10.x). Owns runtime/engine/context, the cuda stream
 * and per-io device buffers; dynamic shapes are applied per run with
 * setInputShape and output buffers are reallocated when their resolved shape
 * changes.
 */
class TrtSession : public InferenceSession {
  public:
    TrtSession() = default;
    ~TrtSession() override;

    TrtSession(const TrtSession&) = delete;
    TrtSession& operator=(const TrtSession&) = delete;

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
    struct DeviceBuffer {
        void* memory = nullptr;
        size_t bytes = 0;

        DeviceBuffer() = default;
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : memory(other.memory), bytes(other.bytes) {
            other.memory = nullptr;
            other.bytes = 0;
        }
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
        ~DeviceBuffer();

        jinq::common::StatusCode ensure(size_t size_bytes);
    };

    trt_detail::SessionLogger _m_logger;
    nvinfer1::IRuntime* _m_runtime = nullptr;
    nvinfer1::ICudaEngine* _m_engine = nullptr;
    nvinfer1::IExecutionContext* _m_context = nullptr;
    cudaStream_t _m_stream = nullptr;
    std::map<std::string, DeviceBuffer> _m_device_buffers;
    std::vector<TensorInfo> _m_input_infos;
    std::vector<TensorInfo> _m_output_infos;
    std::string _m_model_file_path;
};

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_TRT_SESSION_H
