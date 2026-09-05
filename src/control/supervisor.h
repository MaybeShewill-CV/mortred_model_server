/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor.h
* Date: 26-8-22
************************************************/

#ifndef MORTRED_CONTROL_SUPERVISOR_H
#define MORTRED_CONTROL_SUPERVISOR_H

#include <atomic>
#include <condition_variable>
#include <sys/types.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "control/catalog.h"
#include "control/control_config.h"
#include "control/log_buffer.h"
#include "control/restart_policy.h"

namespace mortred {
namespace control {

inline constexpr const char* kGatewayId = "__gateway";

/***
 * Linux process supervision core. Owns the whole managed process tree:
 * spawn (fork/exec + PDEATHSIG + env injection), SIGCHLD reaping via one
 * sigwaitinfo thread, restart policies with backoff and crash-loop give-up,
 * readiness probing, ordered autostart (gateway first) and ordered shutdown
 * (models first, gateway last).
 *
 * The caller must invoke block_supervision_signals() before creating any
 * thread (main() start) and must call start_threads() after the catalog is
 * set. REST actions (start/stop/restart/status) are thread-safe.
 */
class ProcessSupervisor {
  public:
    struct Status {
        std::string state;         // stopped/starting/running/backoff/failed
        int pid = -1;
        bool ready = false;
        int restart_count = 0;
        int last_exit_status = 0;
        bool has_last_exit = false;
        int64_t started_at_unix_ms = 0;  // 0 = not running
        std::string error;
    };

    /*** block SIGCHLD/SIGINT/SIGTERM process-wide (call first thing in main) */
    static void block_supervision_signals();

    ProcessSupervisor(const std::string& project_root, const ControlConfig& cfg,
                      const std::string& control_config_path);
    ~ProcessSupervisor();

    ProcessSupervisor(const ProcessSupervisor&) = delete;
    ProcessSupervisor& operator=(const ProcessSupervisor&) = delete;

    /*** build the child table from the catalog; gateway child is implicit */
    void set_catalog(const Catalog& catalog);

    /*** spawn the reaper/signal/monitor threads; false on fatal */
    bool start_threads(std::string* err = nullptr);

    bool start_server(const std::string& id, std::string* err);
    bool stop_server(const std::string& id, std::string* err);
    bool restart_server(const std::string& id, std::string* err);

    bool has_server(const std::string& id) const;
    Status status(const std::string& id) const;
    std::vector<std::pair<std::string, Status>> statuses() const;
    LogBuffer* logs(const std::string& id);

    /*** gateway first (wait for /healthz), then autostart-eligible models */
    void autostart_all();

    /*** signal-thread entry: request the ordered shutdown sequence */
    void request_shutdown();
    /*** block until children are stopped and worker threads joined */
    void wait_shutdown();
    bool shutdown_requested() const {
        return _shutdown_requested.load();
    }

    const GatewayConfig& gateway_config() const {
        return _cfg.gateway;
    }
    const std::string& internal_token() const {
        return _internal_token;
    }

  private:
    struct Child {
        std::string id;
        bool is_gateway = false;
        ServerEntry entry;  // empty for the gateway
        ServerPolicy policy;
        RestartEngine engine{RestartPolicyKind::kOnFailure};
        pid_t pid = -1;
        bool wanted = false;    // desired running
        bool stopping = false;  // stop requested; next exit is expected
        bool ready = false;
        int last_exit_status = 0;
        bool has_last_exit = false;
        int64_t started_at_unix_ms = 0;
        int64_t backoff_due_ms = 0;   // monotonic ms deadline, 0 = none
        int64_t last_probe_ms = 0;
        // read ends of the child's stdout/stderr pipes. Written by
        // spawn_locked before the reader threads start, consumed (and closed)
        // by handle_exit after joining them; -1 while no child is running.
        int out_fd = -1;
        int err_fd = -1;
        std::thread reader_out;
        std::thread reader_err;
        std::unique_ptr<LogBuffer> log;
        std::string error;
    };

    static int64_t monotonic_ms();
    static int64_t unix_ms();
    static std::string generate_internal_token();

    Child* find_locked(const std::string& id);
    const Child* find_locked(const std::string& id) const;
    std::string bin_path(const Child& child) const;
    void apply_exit_decision(Child* child, const RestartEngine::Decision& decision,
                             bool expected_stop);
    void handle_exit(pid_t pid, int wait_status);
    void reaper_loop();
    void signal_loop();
    void monitor_loop();
    void stop_child_signal_locked(Child* child);
    bool wait_child_exit(const std::string& id, int timeout_ms);
    void perform_ordered_shutdown();
    void probe_readiness(Child* child, int64_t now_ms);
    bool probe_path_of(const Child& child, std::string* path) const;
    int probe_port_of(const Child& child) const;
    void log_line(const std::string& id, const std::string& line);

    std::string _project_root;
    ControlConfig _cfg;
    std::string _control_config_path;
    std::string _internal_token;

    mutable std::mutex _mu;
    std::condition_variable _cv;
    std::map<std::string, std::unique_ptr<Child>> _children;
    std::atomic<bool> _shutdown_requested{false};
    std::atomic<bool> _threads_started{false};
    std::atomic<bool> _threads_stop{false};
    bool _shutdown_done = false;
    std::thread _reaper;
    std::thread _signal_watcher;
    std::thread _monitor;
};

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_SUPERVISOR_H
