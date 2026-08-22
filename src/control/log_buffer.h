/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: log_buffer.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_LOG_BUFFER_H
#define MORTRED_CONTROL_LOG_BUFFER_H

#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace mortred {
namespace control {

/***
 * Thread-safe ring buffer for one supervised process's stdout/stderr lines,
 * also persisted to a size-rotated log file (path -> path.1 on overflow).
 */
class LogBuffer {
  public:
    explicit LogBuffer(const std::string& file_path, size_t rotate_bytes = kDefaultRotateBytes)
        : _path(file_path), _rotate_bytes(rotate_bytes == 0 ? kDefaultRotateBytes : rotate_bytes),
          _file(file_path, std::ios::app) {}

    ~LogBuffer() {
        std::lock_guard<std::mutex> lock(_mu);
        if (_file.is_open()) {
            _file.flush();
            _file.close();
        }
    }

    LogBuffer(const LogBuffer&) = delete;
    LogBuffer& operator=(const LogBuffer&) = delete;

    void reset() {
        std::lock_guard<std::mutex> lock(_mu);
        _lines.clear();
        _head = 0;
        _bytes = 0;
        _file_bytes = 0;
        if (_file.is_open()) {
            _file.close();
        }
        _file.open(_path, std::ios::trunc);
    }

    void append(const std::string& line) {
        std::lock_guard<std::mutex> lock(_mu);
        _lines.push_back(line);
        _bytes += line.size() + 1;
        if (_file.is_open()) {
            _file << line << "\n";
            _file.flush();
            _file_bytes += line.size() + 1;
            if (_file_bytes >= _rotate_bytes) {
                rotate_locked();
            }
        }
        while (_lines.size() > k_max_lines || _bytes > k_max_bytes) {
            _bytes -= _lines.front().size() + 1;
            _lines.pop_front();
            ++_head;
        }
    }

    /*** total lines ever appended (including evicted ones) */
    size_t size() const {
        std::lock_guard<std::mutex> lock(_mu);
        return _head + _lines.size();
    }

    /*** lines [offset, offset + limit) by absolute index */
    std::vector<std::string> slice(size_t offset, size_t limit) const {
        std::lock_guard<std::mutex> lock(_mu);
        std::vector<std::string> out;
        if (limit == 0) {
            return out;
        }
        if (offset < _head) {
            offset = _head;
        }
        const size_t idx = offset - _head;
        if (idx >= _lines.size()) {
            return out;
        }
        const size_t n = std::min(limit, _lines.size() - idx);
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.push_back(_lines[idx + i]);
        }
        return out;
    }

    static constexpr size_t kDefaultRotateBytes = 10 * 1024 * 1024;

  private:
    void rotate_locked() {
        if (_file.is_open()) {
            _file.flush();
            _file.close();
        }
        std::error_code ec;
        std::filesystem::rename(_path, _path + ".1", ec);  // overwrite semantics: .1 is replaced
        if (ec) {
            // rename may fail on exotic filesystems; fall back to truncating in place
        }
        _file_bytes = 0;
        _file.open(_path, std::ios::trunc);
    }

    static constexpr size_t k_max_lines = 2000;
    static constexpr size_t k_max_bytes = 1024 * 1024;

    mutable std::mutex _mu;
    std::string _path;
    size_t _rotate_bytes;
    size_t _file_bytes = 0;
    std::deque<std::string> _lines;
    size_t _head = 0;
    size_t _bytes = 0;
    std::ofstream _file;
};

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_LOG_BUFFER_H
