/************************************************
 * Author: Codex
 * File: rate_limiter.h
 * Date: 2026-08-13
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
 * 固定窗口限流器：按 key（如客户端 IP）限制每秒最大请求数。
 * max_qps <= 0 表示不限流。
 */
class FixedWindowRateLimiter {
public:
    explicit FixedWindowRateLimiter(int max_qps = 0,
                                    int64_t window_ms = 1000)
        : _m_max_qps(max_qps), _m_window_ms(window_ms) {}

    /***
     * 更新每秒上限；<= 0 表示关闭限流。
     */
    void set_max_qps(int max_qps) {
        std::lock_guard<std::mutex> lock(_m_mutex);
        _m_max_qps = max_qps;
        _m_windows.clear();
    }

    /***
     * 当前窗口内是否允许该 key 再发起一次请求。
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
     * 清理非当前窗口的过期记录，避免内存无限增长。
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
