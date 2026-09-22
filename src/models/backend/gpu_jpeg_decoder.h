/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_jpeg_decoder.h
 * Date: 26-9-20
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H
#define MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>

#include "models/backend/gpu_preprocess_desc.h"

namespace jinq {
namespace models {
namespace backend {
namespace gpu_jpeg {

/*** Decode backends that actually run. nvJPEG was removed with the old ladder. */
enum Backend {
    JPEGGPU = 0,
    CPU_REDUCED,
    CPU_FULL,
    FALLBACK,
    BACKEND_COUNT
};

/*** Global request counters per backend, readable by the metrics layer.
 * Incremented by the decode layer on every request. */
extern std::atomic<uint64_t> g_request_count[BACKEND_COUNT];

/*** Prometheus text for /metrics. */
std::string render_decode_metrics();

/*** Device tensor produced by one worker's slot. The pointers alias that
 * slot's buffers. They stay valid until this same slot decodes again or
 * closes, which the worker lease only allows after the consumer stream has
 * synchronized (TrtSession::run does that before returning). */
struct GpuPipelineResult {
    void* device_input = nullptr;
    void* device_gray = nullptr;
    int out_w = 0, out_h = 0;
    int src_w = 0, src_h = 0;
    void* ready_event = nullptr;
    bool valid = false;
};

/*** One jpeggpu decoder, stream, event and set of buffers. Owned by a single
 * worker. The worker lease is the exclusion: run() does not overlap on the
 * same slot, and the slot is reused only after the consumer synchronizes.
 * No mutex. Not safe to share across workers. */
class GpuDecodeSlot {
  public:
    GpuDecodeSlot();
    ~GpuDecodeSlot();

    GpuDecodeSlot(const GpuDecodeSlot&) = delete;
    GpuDecodeSlot& operator=(const GpuDecodeSlot&) = delete;
    GpuDecodeSlot(GpuDecodeSlot&&) = delete;
    GpuDecodeSlot& operator=(GpuDecodeSlot&&) = delete;

    /*** Probe jpeggpu on a temporary decoder, then keep a private one.
     * Safe to call once from the worker's init thread. */
    bool open();
    void close();
    bool ready() const;

    GpuPipelineResult decode_and_preprocess(
        const unsigned char* data, size_t size,
        int network_w, int network_h,
        const GpuPreprocessDescriptor& desc,
        std::string* err);

  private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H
