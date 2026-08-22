/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: backpressure.h
* Date: 26-8-22
************************************************/

// Overload arithmetic shared by the model servers (pure functions, unit
// tested): how long a rejected client should back off before retrying.

#ifndef MORTRED_SERVER_BACKPRESSURE_H
#define MORTRED_SERVER_BACKPRESSURE_H

#include <algorithm>
#include <cmath>

namespace jinq {
namespace server {

/***
 * Retry-After seconds for an overloaded queue: queue depth times the EWMA
 * per-request run time divided by workers, rounded up and clamped to
 * [1, 60]. Degenerate inputs (no workers / no samples / empty queue) fall
 * back to 1s so the header is always a positive hint.
 */
inline int compute_retry_after_seconds(size_t queue_depth, int64_t run_time_ewma_ms,
                                       size_t worker_nums) {
    if (worker_nums == 0 || run_time_ewma_ms <= 0 || queue_depth == 0) {
        return 1;
    }
    const double drain_ms = static_cast<double>(queue_depth) *
                            static_cast<double>(run_time_ewma_ms) /
                            static_cast<double>(worker_nums);
    const int seconds = static_cast<int>(std::ceil(drain_ms / 1000.0));
    return std::min(60, std::max(1, seconds));
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_BACKPRESSURE_H
