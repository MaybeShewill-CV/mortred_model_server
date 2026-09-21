/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: trt_session.h
* Date: 26-8-20
************************************************/

#ifndef MORTRED_MODELS_BACKEND_TRT_SESSION_H
#define MORTRED_MODELS_BACKEND_TRT_SESSION_H

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <mutex>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "common/status_code.h"
#include "models/backend/backend_config.h"
#include "models/backend/session.h"

namespace jinq {
namespace models {
namespace backend {
using jinq::common::StatusCode;

namespace trt_detail {

/*** pooled pinned staging slots (see trt_session.cpp for the race this
 * replaces): each run() owns its slot until the stream sync guarantees the
 * queued async copies have executed. */
struct PinnedSlotPool {
    static constexpr size_t k_max_idle = 16;
    std::mutex mu;
    std::deque<std::pair<void*, size_t>> idle;

    void* acquire(size_t bytes) {
        {
            const std::lock_guard<std::mutex> guard(mu);
            for (size_t i = 0; i < idle.size(); ++i) {
                if (idle[i].second >= bytes) {
                    void* p = idle[i].first;
                    idle.erase(idle.begin() + static_cast<long>(i));
                    return p;
                }
            }
        }
        void* p = nullptr;
        return cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
    }

    void release(void* p, size_t bytes) {
        if (p == nullptr) return;
        const std::lock_guard<std::mutex> guard(mu);
        if (idle.size() >= k_max_idle) {
            cudaFreeHost(p);
            return;
        }
        idle.emplace_back(p, bytes);
    }

    void drain() {
        const std::lock_guard<std::mutex> guard(mu);
        for (auto& item : idle) {
            if (item.first) cudaFreeHost(item.first);
        }
        idle.clear();
    }
};


class SessionLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
};

}  // namespace trt_detail

/***
 * RAII TensorRT 10 inference session (tensor-address / enqueueV3 API).
 * Owns runtime/engine/context, the cuda stream
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
    class DynamicOutputAllocator;

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

        StatusCode ensure(size_t size_bytes);
    };

    trt_detail::SessionLogger _m_logger;
    nvinfer1::IRuntime* _m_runtime = nullptr;
    nvinfer1::ICudaEngine* _m_engine = nullptr;
    nvinfer1::IExecutionContext* _m_context = nullptr;
    cudaStream_t _m_stream = nullptr;
    std::map<std::string, DeviceBuffer> _m_device_buffers;
    std::map<std::string, DynamicOutputAllocator*> _m_output_allocators;
    std::vector<TensorInfo> _m_input_infos;
    std::vector<TensorInfo> _m_output_infos;
    std::string _m_model_file_path;

    // pinned host staging for H2D/D2H: pageable std::vector transfers pay a
    // driver-side staging copy (~3x measured vs pinned). Slots come from
    // per-request pools — the old single shared buffer raced under
    // concurrent workers (host memcpy vs queued async copies).
    trt_detail::PinnedSlotPool _m_pinned_h2d_pool;
    trt_detail::PinnedSlotPool _m_pinned_d2h_pool;

    void* pinned_h2d_stage(size_t bytes);
    void pinned_h2d_release(void* slot, size_t bytes);
    void* pinned_d2h_stage(size_t bytes);
    void pinned_d2h_release(void* slot, size_t bytes);
    void release_pinned_staging();
};

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_TRT_SESSION_H
