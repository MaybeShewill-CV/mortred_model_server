/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: item_exec.h
* Date: 26-9-9
************************************************/

// Shared per-item inference primitives used by the sync path, async_run_job
// and the batch collector. Namespace-level templates (same style as
// backpressure.h), not a runner class. make_model_input_from_payload was
// dead code and is not migrated.

#ifndef MORTRED_SERVER_ITEM_EXEC_H
#define MORTRED_SERVER_ITEM_EXEC_H

#include <chrono>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "common/response_envelope.h"
#include "common/status_code.h"
#include "models/backend/param_spec.h"
#include "models/io/common_input.h"
#include "server/inference_task.h"
#include "server/output_options.h"

namespace jinq {
namespace server {
using jinq::common::StatusCode;

template <typename INPUT>
inline INPUT make_model_input(models::io_define::common_io::byte_source item,
                              const models::backend::ParamSet* params) {
    INPUT input;
    if constexpr (std::is_same<INPUT, models::io_define::common_io::image_input>::value) {
        input.image = std::move(item);
        input.params = params;
    } else if constexpr (std::is_same<INPUT, models::io_define::common_io::base64_input>::value) {
        input.input_image_content = std::move(item.data);
    } else {
        static_assert(sizeof(INPUT) == 0, "unsupported worker input type for envelope items");
    }
    return input;
}

template <typename MODEL_OUTPUT>
inline void aggregate_item_statuses(InferenceResult<MODEL_OUTPUT>* result) {
    size_t ok_count = 0;
    StatusCode first_error = StatusCode::OK;
    bool any_timeout = false;
    for (const StatusCode status : result->item_status) {
        if (status == StatusCode::OK) {
            ++ok_count;
        } else {
            if (first_error == StatusCode::OK) {
                first_error = status;
            }
            if (status == StatusCode::MODEL_RUN_TIMEOUT) {
                any_timeout = true;
            }
        }
    }
    if (first_error == StatusCode::OK) {
        result->model_run_status = StatusCode::OK;
        result->partial = false;
        return;
    }
    if (any_timeout && ok_count > 0) {
        result->model_run_status = StatusCode::DEADLINE_EXCEEDED_PARTIAL;
        result->partial = true;
        return;
    }
    result->model_run_status = first_error;
    result->partial = false;
}

template <typename WORKER, typename MODEL_OUTPUT>
inline void run_items(WORKER& worker, const InferenceTask& req,
                      InferenceResult<MODEL_OUTPUT>* result) {
    using ModelInput = typename WORKER::element_type::input_type;
    const size_t n_items = req.item_count();
    result->options = req.options;
    result->item_status.assign(n_items, StatusCode::OK);
    result->item_outputs.assign(n_items, MODEL_OUTPUT{});

    for (size_t idx = 0; idx < n_items; ++idx) {
        if (req.deadline != std::chrono::steady_clock::time_point::max() &&
            std::chrono::steady_clock::now() >= req.deadline) {
            for (size_t rest = idx; rest < n_items; ++rest) {
                result->item_status[rest] = StatusCode::MODEL_RUN_TIMEOUT;
            }
            break;
        }
        ModelInput input = make_model_input<ModelInput>(req.items[idx], req.params.get());
        const StatusCode status = worker->run(input, result->item_outputs[idx]);
        result->item_status[idx] = status;
    }
    aggregate_item_statuses(result);
    if (result->model_run_status != StatusCode::OK) {
        LOG(ERROR) << "worker run failed with status "
                   << jinq::common::to_underlying(result->model_run_status);
    }
}

template <typename MODEL_OUTPUT, typename FillFn>
inline jinq::common::UnifiedResponse inference_result_to_unified(
    const std::string& task_id, const std::string& model_name,
    const InferenceResult<MODEL_OUTPUT>& result, FillFn fill) {
    jinq::common::UnifiedResponse unified;
    unified.task_id = task_id;
    unified.model_name = model_name;
    unified.status = jinq::common::to_underlying(result.model_run_status);
    unified.status_str = jinq::common::status_code_to_str(result.model_run_status);
    unified.partial = result.partial;
    unified.server_time_ms =
        result.worker_run_time_consuming + result.find_worker_time_consuming;
    unified.results.reserve(result.item_status.size());
    for (size_t idx = 0; idx < result.item_status.size(); ++idx) {
        const StatusCode item_status = result.item_status[idx];
        jinq::common::ResponseItem item;
        item.status = jinq::common::to_underlying(item_status);
        if (item_status == StatusCode::OK) {
            rapidjson::Document data;
            fill(data.GetAllocator(), data, result.item_outputs[idx], result.options);
            item.data = std::move(data);
        }
        unified.results.push_back(std::move(item));
    }
    return unified;
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_ITEM_EXEC_H
