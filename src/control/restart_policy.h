/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: restart_policy.h
* Date: 26-8-22
************************************************/

// Pure restart decision state machine. No threads, no clocks: callers pass
// monotonic milliseconds so every transition is unit-testable.

#ifndef MORTRED_CONTROL_RESTART_POLICY_H
#define MORTRED_CONTROL_RESTART_POLICY_H

#include <cstdint>
#include <algorithm>
#include <string>

namespace mortred {
namespace control {

enum class RestartPolicyKind { kNo = 0, kOnFailure = 1, kAlways = 2 };
enum class SupervisedState { kStopped = 0, kStarting = 1, kRunning = 2, kBackoff = 3, kFailed = 4 };

inline bool parse_restart_policy(const std::string& s, RestartPolicyKind* out) {
    if (s == "no") {
        *out = RestartPolicyKind::kNo;
    } else if (s == "on-failure") {
        *out = RestartPolicyKind::kOnFailure;
    } else if (s == "always") {
        *out = RestartPolicyKind::kAlways;
    } else {
        return false;
    }
    return true;
}

inline const char* to_string(SupervisedState s) {
    switch (s) {
        case SupervisedState::kStopped:
            return "stopped";
        case SupervisedState::kStarting:
            return "starting";
        case SupervisedState::kRunning:
            return "running";
        case SupervisedState::kBackoff:
            return "backoff";
        case SupervisedState::kFailed:
            return "failed";
        default:
            return "unknown";
    }
}

/***
 * Backoff/crash-loop constants (fixed policy, intentionally not configurable):
 * - base 500ms, exponential x2, 30s cap
 * - a run that stayed ready >= 5 minutes resets backoff and the crash window
 * - more than 5 restart decisions inside a 60s window give up -> Failed
 */
struct RestartConstants {
    static constexpr int kBackoffBaseMs = 500;
    static constexpr int kBackoffMaxMs = 30000;
    static constexpr int64_t kStableResetMs = 5 * 60 * 1000;
    static constexpr int64_t kCrashWindowMs = 60 * 1000;
    static constexpr int kMaxRestartsInWindow = 5;
};

class RestartEngine {
  public:
    struct Decision {
        bool restart = false;
        int delay_ms = 0;
        bool gave_up = false;
    };

    explicit RestartEngine(RestartPolicyKind policy)
        : _policy(policy) {}

    /*** process spawned successfully */
    void note_started(int64_t now_ms) {
        _state = SupervisedState::kStarting;
        _started_at_ms = now_ms;
        _ready_at_ms = 0;
    }

    /*** readiness confirmed */
    void note_ready(int64_t now_ms) {
        _state = SupervisedState::kRunning;
        if (_ready_at_ms == 0) {
            _ready_at_ms = now_ms;
        }
    }

    /*** missing engine / config: Failed, no backoff */
    void note_permanent_failure() {
        reset();
        _state = SupervisedState::kFailed;
    }

    /*** external cancel (manual stop while in backoff): back to Stopped */
    void note_cancel() {
        reset();
        _state = SupervisedState::kStopped;
    }

    /***
     * Process exited (or spawn failed: clean=false).
     * @param expected_stop true when a stop() was requested for this exit
     */
    Decision note_exit(int64_t now_ms, bool clean_exit, bool expected_stop) {
        _last_exit_clean = clean_exit;
        if (expected_stop) {
            reset();
            _state = SupervisedState::kStopped;
            return Decision{};
        }
        if (_state == SupervisedState::kFailed) {
            return Decision{};
        }
        // a stable run heals both the backoff ladder and the crash window
        if (_ready_at_ms != 0 && now_ms - _ready_at_ms >= RestartConstants::kStableResetMs) {
            _backoff_next_ms = RestartConstants::kBackoffBaseMs;
            _window_count = 0;
        }

        const bool want_restart = _policy == RestartPolicyKind::kAlways ||
                                  (_policy == RestartPolicyKind::kOnFailure && !clean_exit);
        if (!want_restart) {
            _state = SupervisedState::kStopped;
            return Decision{};
        }

        if (now_ms - _window_start_ms >= RestartConstants::kCrashWindowMs) {
            _window_start_ms = now_ms;
            _window_count = 0;
        }
        ++_window_count;
        if (_window_count > RestartConstants::kMaxRestartsInWindow) {
            reset();
            _state = SupervisedState::kFailed;
            Decision d;
            d.gave_up = true;
            return d;
        }

        Decision d;
        d.restart = true;
        d.delay_ms = _backoff_next_ms;
        _backoff_next_ms = std::min(_backoff_next_ms * 2, RestartConstants::kBackoffMaxMs);
        _state = SupervisedState::kBackoff;
        ++_restart_count;
        return d;
    }

    SupervisedState state() const {
        return _state;
    }
    int restart_count() const {
        return _restart_count;
    }
    bool last_exit_clean() const {
        return _last_exit_clean;
    }

  private:
    void reset() {
        _backoff_next_ms = RestartConstants::kBackoffBaseMs;
        _window_count = 0;
        _window_start_ms = 0;
        _ready_at_ms = 0;
        _started_at_ms = 0;
    }

    RestartPolicyKind _policy;
    SupervisedState _state = SupervisedState::kStopped;
    int _backoff_next_ms = RestartConstants::kBackoffBaseMs;
    int64_t _window_start_ms = 0;
    int _window_count = 0;
    int _restart_count = 0;
    int64_t _started_at_ms = 0;
    int64_t _ready_at_ms = 0;
    bool _last_exit_clean = true;
};

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_RESTART_POLICY_H
