/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: log_buffer.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_LOG_BUFFER_H
#define MORTRED_CONTROL_LOG_BUFFER_H

#include <algorithm>
#include <cstdint>
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
 *
 * A respawned process must never destroy the ended run's evidence: rollover()
 * renames the current file to <path>.run<unix_ms> and starts a fresh one,
 * keeping the newest kArchiveKeep archives. The archive name is intentionally
 * distinct from the size-rotation ".1" suffix so retention sweeping cannot
 * touch size-rotated files. Instances are shared via shared_ptr across a
 * respawn: a reader thread straggler holding an old instance appends into the
 * archived inode, never into the new run's file.
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

    /*** Archive the ended run and start fresh: rename the current file to
     * <path>.run<archive_stamp>, delete archives beyond kArchiveKeep (newest
     * kept), clear the in-memory ring, reopen the live path empty. The caller
     * supplies the stamp (supervisor unix_ms) so tests stay deterministic. */
    void rollover(uint64_t archive_stamp) {
        std::lock_guard<std::mutex> lock(_mu);
        if (_file.is_open()) {
            _file.flush();
            _file.close();
        }
        const std::string archive = _path + ".run" + std::to_string(archive_stamp);
        std::error_code ec;
        std::filesystem::rename(_path, archive, ec);
        // rename may fail on exotic filesystems; the truncating reopen below
        // then loses the ended run, same as the pre-rollover behavior
        sweep_archives_locked();
        _lines.clear();
        _head = 0;
        _bytes = 0;
        _file_bytes = 0;
        _unflushed = 0;
        _file.open(_path, std::ios::trunc);
    }

    void append(const std::string& line) {
        std::lock_guard<std::mutex> lock(_mu);
        _lines.push_back(line);
        _bytes += line.size() + 1;
        if (_file.is_open()) {
            // Do not flush every line: a stalled log filesystem would block the
            // pipe reader, fill the child's stdout pipe, and freeze reaping.
            _file << line << "\n";
            _file_bytes += line.size() + 1;
            ++_unflushed;
            if (_file_bytes >= _rotate_bytes) {
                rotate_locked();
            } else if (_unflushed >= k_flush_every_lines) {
                _file.flush();
                _unflushed = 0;
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
    // respawn archives kept per child; disk bound = kArchiveKeep * rotate_bytes
    static constexpr size_t kArchiveKeep = 3;

  private:
    void rotate_locked() {
        if (_file.is_open()) {
            _file.flush();
            _file.close();
        }
        _unflushed = 0;
        std::error_code ec;
        std::filesystem::rename(_path, _path + ".1", ec);  // overwrite semantics: .1 is replaced
        if (ec) {
            // rename may fail on exotic filesystems; fall back to truncating in place
        }
        _file_bytes = 0;
        _file.open(_path, std::ios::trunc);
    }

    /*** delete the oldest .run<stamp> archives beyond kArchiveKeep; ignores
     * every non-archive entry (including the size-rotation ".1") */
    void sweep_archives_locked() {
        std::filesystem::path live(_path);
        const std::filesystem::path dir =
            live.parent_path().empty() ? std::filesystem::path(".") : live.parent_path();
        const std::string prefix = live.filename().string() + ".run";
        struct Archive {
            uint64_t stamp;
            std::filesystem::path path;
        };
        std::vector<Archive> archives;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            const std::string name = entry.path().filename().string();
            if (name.size() <= prefix.size() ||
                name.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            const std::string stamp_text = name.substr(prefix.size());
            if (stamp_text.find_first_not_of("0123456789") != std::string::npos) {
                continue;
            }
            archives.push_back({std::stoull(stamp_text), entry.path()});
        }
        if (archives.size() <= kArchiveKeep) {
            return;
        }
        std::sort(archives.begin(), archives.end(),
                  [](const Archive& a, const Archive& b) { return a.stamp > b.stamp; });
        for (size_t i = kArchiveKeep; i < archives.size(); ++i) {
            std::filesystem::remove(archives[i].path, ec);
        }
    }

    static constexpr size_t k_max_lines = 2000;
    static constexpr size_t k_max_bytes = 1024 * 1024;
    static constexpr size_t k_flush_every_lines = 64;

    mutable std::mutex _mu;
    std::string _path;
    size_t _rotate_bytes;
    size_t _file_bytes = 0;
    std::deque<std::string> _lines;
    size_t _head = 0;
    size_t _bytes = 0;
    size_t _unflushed = 0;
    std::ofstream _file;
};

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_LOG_BUFFER_H
