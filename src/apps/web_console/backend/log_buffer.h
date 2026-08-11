/************************************************
 * Author: Codex
 * File: log_buffer.h
 ************************************************/

#ifndef MORTRED_WEB_LOG_BUFFER_H
#define MORTRED_WEB_LOG_BUFFER_H

#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace mortred_web {

/***
 * thread-safe ring buffer for a server's stdout/stderr lines,
 * persisted to a log file as well
 */
class LogBuffer {
  public:
    explicit LogBuffer(const std::string& file_path)
        : _path(file_path), _file(file_path, std::ios::app) {}

    ~LogBuffer() {
        std::lock_guard<std::mutex> lock(_mu);
        if (_file.is_open()) {
            _file.flush();
            _file.close();
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(_mu);
        _lines.clear();
        _head = 0;
        _bytes = 0;
        if (_file.is_open()) {
            _file.close();
            _file.open(_path, std::ios::trunc);
        }
    }

    void append(const std::string& line) {
        std::lock_guard<std::mutex> lock(_mu);
        _lines.push_back(line);
        _bytes += line.size() + 1;
        if (_file.is_open()) {
            _file << line << "\n";
            _file.flush();
        }
        while (_lines.size() > kMaxLines || _bytes > kMaxBytes) {
            _bytes -= _lines.front().size() + 1;
            _lines.pop_front();
            ++_head;
        }
    }

    /***
     * @return total number of lines ever appended (including evicted ones)
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(_mu);
        return _head + _lines.size();
    }

    /***
     * @param offset absolute line index
     * @param limit max lines to return
     * @return lines [offset, offset+limit)
     */
    std::vector<std::string> slice(size_t offset, size_t limit) const {
        std::lock_guard<std::mutex> lock(_mu);
        std::vector<std::string> out;
        if (limit == 0) {
            return out;
        }
        if (offset < _head) {
            offset = _head;
        }
        size_t idx = offset - _head;
        if (idx >= _lines.size()) {
            return out;
        }
        size_t n = std::min(limit, _lines.size() - idx);
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.push_back(_lines[idx + i]);
        }
        return out;
    }

  private:
    static constexpr size_t kMaxLines = 2000;
    static constexpr size_t kMaxBytes = 1024 * 1024;

    mutable std::mutex _mu;
    std::string _path;
    std::deque<std::string> _lines;
    size_t _head = 0;
    size_t _bytes = 0;
    std::ofstream _file;
};

} // namespace mortred_web

#endif // MORTRED_WEB_LOG_BUFFER_H
