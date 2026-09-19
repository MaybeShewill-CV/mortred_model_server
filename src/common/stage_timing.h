/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: stage_timing.h
 * Date: 26-9-19
 ************************************************/

// Opt-in per-request stage tracing for the profiling campaign
// (branch perf/stage-timing). Enabled at runtime by touching the flag file
//   /tmp/mortred_stage_timing
// and disabled by removing it; the check is one stat() per request, so the
// instrumented build behaves identically to the stock build while the flag
// is absent. Each traced request emits one glog line
//   stage_trace svc=<model|gateway> id=<req_id> <stage>=<ms> ... total=<ms>
// where each value is the delta from the previous mark, i.e. a name labels
// the END of its stage. Deep layers (backend session, cv preprocess) mark
// through a thread-local hook that the request owner swaps in for the
// synchronous compute window only, so no cross-thread bookkeeping is needed
// at call sites.

#ifndef MORTRED_COMMON_STAGE_TIMING_H
#define MORTRED_COMMON_STAGE_TIMING_H

#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace jinq {
namespace common {
namespace stage_timing {

inline bool enabled() {
    struct stat probe;
    return ::stat("/tmp/mortred_stage_timing", &probe) == 0;
}

class StageTrace {
  public:
    void mark(const char* name) {
        std::lock_guard<std::mutex> guard(marks_mu_);
        if (marks_.size() < k_max_marks) {
            marks_.push_back(Mark{name, std::chrono::steady_clock::now()});
        }
    }

    void set_id(std::string id) { id_ = std::move(id); }

    std::string to_log_line(const char* svc) const {
        std::string out = "stage_trace svc=";
        out += svc;
        if (!id_.empty()) {
            out += " id=";
            out += id_;
        }
        char buf[64];
        std::lock_guard<std::mutex> guard(marks_mu_);
        for (size_t idx = 1; idx < marks_.size(); ++idx) {
            const double ms = std::chrono::duration<double, std::milli>(
                                  marks_[idx].tp - marks_[idx - 1].tp).count();
            std::snprintf(buf, sizeof(buf), " %s=%.3f", marks_[idx].name, ms);
            out += buf;
        }
        if (marks_.size() >= 2) {
            const double total = std::chrono::duration<double, std::milli>(
                                     marks_.back().tp - marks_.front().tp).count();
            std::snprintf(buf, sizeof(buf), " total=%.3f", total);
            out += buf;
        }
        return out;
    }

    // hook for deep layers: swap this trace in as the active one on the
    // current thread for a synchronous window, then restore the previous
    static StageTrace* active() { return tl_active_; }
    static StageTrace* swap_active(StageTrace* next) {
        StageTrace* prev = tl_active_;
        tl_active_ = next;
        return prev;
    }

  private:
    static constexpr size_t k_max_marks = 32;
    struct Mark {
        const char* name;
        std::chrono::steady_clock::time_point tp;
    };
    mutable std::mutex marks_mu_;
    std::vector<Mark> marks_;
    std::string id_;
    inline static thread_local StageTrace* tl_active_ = nullptr;
};

inline void mark(const char* name) {
    StageTrace* trace = StageTrace::active();
    if (trace != nullptr) {
        trace->mark(name);
    }
}

}  // namespace stage_timing
}  // namespace common
}  // namespace jinq

#endif  // MORTRED_COMMON_STAGE_TIMING_H
