/************************************************
 * Author: Codex
 * File: server_manager.h
 ************************************************/

#ifndef MORTRED_WEB_SERVER_MANAGER_H
#define MORTRED_WEB_SERVER_MANAGER_H

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "catalog.h"
#include "log_buffer.h"

namespace mortred_web {

class ServerManager {
  public:
    struct Status {
        bool running = false;
        int pid = -1;
        bool healthy = false;
        std::string error;
    };

    void init(const Catalog& catalog, const std::string& project_root, const std::string& logs_dir);

    /***
     * spawn the server process, capture stdout/stderr
     * @return true on success (process spawned)
     */
    bool start(const ServerEntry& entry, std::string& err);

    /***
     * stop the server process (SIGTERM then SIGKILL)
     */
    bool stop(const std::string& id, std::string& err);

    bool is_running(const std::string& id) const;

    /***
     * true when the model server's real /ready endpoint answers 2xx
     * (short-timeout HTTP probe; log-grep heuristic removed)
     */
    bool is_ready(const std::string& id);

    Status status(const std::string& id) const;

    LogBuffer* logs(const std::string& id);

    std::vector<std::string> running_ids() const;

  private:
    struct ProcInfo {
        pid_t pid = -1;
        int out_fd = -1;
        int err_fd = -1;
        std::atomic<bool> running{false};
        std::thread reader_out;
        std::thread reader_err;
        std::thread waiter;
    };

    void spawn_waiters(const std::string& id, ProcInfo* info);
    LogBuffer* get_or_create_log(const std::string& id);

    const Catalog* _catalog = nullptr;
    std::string _bin_dir;
    std::string _lib_dir;
    std::string _libs_dir;
    std::string _logs_dir;

    mutable std::mutex _mu;
    std::condition_variable _cv;
    std::unordered_map<std::string, std::unique_ptr<ProcInfo>> _procs;
    std::unordered_map<std::string, std::unique_ptr<LogBuffer>> _logs;
    std::unordered_map<std::string, int> _ports;
};

} // namespace mortred_web

#endif // MORTRED_WEB_SERVER_MANAGER_H
