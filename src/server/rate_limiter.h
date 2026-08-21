/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: rate_limiter.h
* Date: 26-8-13
************************************************/

#ifndef MORTRED_SERVER_RATE_LIMITER_H
#define MORTRED_SERVER_RATE_LIMITER_H

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace jinq {
namespace server {

/***
 * Fixed-window rate limiter: caps requests per second per key (e.g. client IP).
 * max_qps <= 0 disables limiting.
 */
class FixedWindowRateLimiter {
public:
    explicit FixedWindowRateLimiter(int max_qps = 0,
                                    int64_t window_ms = 1000)
        : _m_max_qps(max_qps), _m_window_ms(window_ms) {}

    /***
     * Update the per-second cap; <= 0 disables limiting.
     */
    void set_max_qps(int max_qps) {
        std::lock_guard<std::mutex> lock(_m_mutex);
        _m_max_qps = max_qps;
        _m_windows.clear();
    }

    /***
     * Whether the key may make another request in the current window.
     */
    bool allow(const std::string& key) {
        std::lock_guard<std::mutex> lock(_m_mutex);
        if (_m_max_qps <= 0) {
            return true;
        }
        int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();
        int64_t window = now_ms / _m_window_ms;
        auto it = _m_windows.find(key);
        if (it == _m_windows.end()) {
            prune_locked(window);
            _m_windows[key] = {window, 1};
            return true;
        }
        if (it->second.window_start != window) {
            it->second = {window, 1};
            return true;
        }
        if (it->second.count >= _m_max_qps) {
            return false;
        }
        ++it->second.count;
        return true;
    }

private:
    struct WindowState {
        int64_t window_start = 0;
        int count = 0;
    };

    /***
     * Drop stale records from non-current windows to bound memory growth.
     */
    void prune_locked(int64_t current_window) {
        if (_m_windows.size() < k_max_entries) {
            return;
        }
        for (auto it = _m_windows.begin(); it != _m_windows.end();) {
            if (it->second.window_start != current_window) {
                it = _m_windows.erase(it);
            } else {
                ++it;
            }
        }
    }

    static constexpr size_t k_max_entries = 4096;

    std::mutex _m_mutex;
    std::unordered_map<std::string, WindowState> _m_windows;
    int _m_max_qps = 0;
    int64_t _m_window_ms = 1000;
};

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_RATE_LIMITER_H
