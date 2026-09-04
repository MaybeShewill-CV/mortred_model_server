/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: inference_task.h
 * Date: 26-9-4
 ************************************************/

// In-process inference execution types. Shared by the synchronous go-task
// path, the batch collector, and AsyncJobTable. Not part of the HTTP
// envelope contract (that is request_envelope / response_envelope) and not
// unique to async jobs.
//
// ParsedRequest (HTTP bind, may carry 422 violations)
//   → InferenceTask (admitted, deadline attached)
//   → InferenceResult<Output> (typed model outputs)
//   → UnifiedResponse (wire JSON)

#ifndef MORTRED_SERVER_INFERENCE_TASK_H
#define MORTRED_SERVER_INFERENCE_TASK_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "common/status_code.h"
#include "models/backend/param_spec.h"
#include "models/io/common_input.h"
#include "server/output_options.h"

namespace jinq {
namespace server {

using jinq::common::StatusCode;

struct InferenceTask {
    std::string task_id;
    std::vector<jinq::models::io_define::common_io::byte_source> items;
    std::shared_ptr<jinq::models::backend::ParamSet> params;
    jinq::server::OutputOptions options;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();

    size_t item_count() const { return items.size(); }
};

template <typename MODEL_OUTPUT>
struct InferenceResult {
    StatusCode model_run_status = StatusCode::OK;
    std::string task_finished_ts;
    double worker_run_time_consuming = 0;
    double find_worker_time_consuming = 0;
    bool partial = false;
    jinq::server::OutputOptions options;
    std::vector<MODEL_OUTPUT> item_outputs;
    std::vector<StatusCode> item_status;
};

} // namespace server
} // namespace jinq

#endif // MORTRED_SERVER_INFERENCE_TASK_H
