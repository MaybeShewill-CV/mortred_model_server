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
 * True when pgid's process group holds no process at all (killpg(2) ESRCH).
 * pgid <= 0 means no group was ever recorded, so there is nothing to wait for.
 * It is the primitive stop/restart/shutdown use to confirm that a stop reaped
 * grandchildren - not just the leader - and doubles as a test assertion and an
 * ops self-check. A zombie member still counts: the group disappears only once
 * its corpses are reaped.
 */
bool process_group_gone(pid_t pgid);

/***
 * Linux process supervision core. Owns the whole managed process tree:
 * spawn (fork/exec + own process group + PDEATHSIG + env injection), SIGCHLD
 * reaping via one sigwaitinfo thread, restart policies with backoff and
 * crash-loop give-up, readiness probing, ordered autostart (gateway first) and
 * ordered shutdown (models first, gateway last). Every managed child leads its
 * own process group and every stop/restart/shutdown signal goes to that group.
 *
 * Semantic consequence: a child is no longer in the supervisor's foreground
 * process group, so terminal Ctrl-C no longer reaches children directly. They
 * are stopped only through the supervisor's ordered two-phase shutdown.
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

    /*** Snapshot of the most recent autostart_all gateway gate (read-only). */
    struct AutostartGate {
        bool blocked = false;         // gateway phase failed / not ready
        bool models_skipped = false;  // this wave spawned zero models
        bool models_failed = false;   // model wave spawn fail / models deadline (not skip)
        std::string reason;           // human-readable; scrape / deadline / missing binary
    };

    /*** block SIGCHLD/SIGINT/SIGTERM process-wide (call first thing in main) */
    static void block_supervision_signals();

    ProcessSupervisor(const std::string& project_root, const ControlConfig& cfg,
                      const std::string& control_config_path);
    ~ProcessSupervisor();

    ProcessSupervisor(const ProcessSupervisor&) = delete;
    ProcessSupervisor& operator=(const ProcessSupervisor&) = delete;

    /*** scrape / inference / management tokens used only when spawning __gateway.
     * Empty metrics (the default) permanently fails gateway spawn. Tests set
     * these on the object; the daemon copies them from SupervisorInitOptions. */
    void set_gateway_trust(const std::string& metrics_token, const std::string& infer_token,
                           const std::string& admin_token);

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
    /*** shared_ptr so a caller reading lines outlives a concurrent respawn's
     * buffer swap (the raw-pointer form was a use-after-free window) */
    std::shared_ptr<LogBuffer> logs(const std::string& id);
    AutostartGate autostart_gate() const;

    /*** gateway first (wait for /healthz), then autostart-eligible models.
     *  Both phases honor deadlines so a stuck child cannot block supervisor
     *  startup forever. Gateway binary missing, start failure, or deadline
     *  without running skips the model wave; reason is on status()/autostart_gate(). */
    void autostart_all(int gateway_deadline_ms = 15000, int models_deadline_ms = 60000);

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
        // Every managed child leads its own process group (spawn_locked calls
        // setpgid on both sides of the fork), so pgid == pid while it owns the
        // group. Stop/restart/shutdown deliver signals to the group so a
        // grandchild cannot outlive its leader; pgid <= 0 means no group was
        // recorded and the delivery sites fall back to the single pid.
        pid_t pgid = -1;
        bool wanted = false;    // desired running
        bool stopping = false;  // stop requested; next exit is expected
        bool ready = false;
        int last_exit_status = 0;
        bool has_last_exit = false;
        int64_t started_at_unix_ms = 0;
        int64_t backoff_due_ms = 0;   // monotonic ms deadline, 0 = none
        int64_t last_probe_ms = 0;
        // Log-pipe ownership: spawn_locked creates the pipes and reader
        // threads, then moves the fresh LogBuffer (shared) into child->log.
        // handle_exit closes fds and moves both reader threads into a detached
        // joiner that also holds the ended run's buffer (sole join site —
        // never join on the reaper; a stalled filesystem must not stop
        // reaping). spawn_locked never waits for old readers: it archives the
        // old file and swaps in a brand-new LogBuffer, so a straggler reader
        // can only ever append into the archived inode.
        int out_fd = -1;
        int err_fd = -1;
        std::thread reader_out;
        std::thread reader_err;
        std::shared_ptr<LogBuffer> log;
        std::string error;
    };

    static int64_t monotonic_ms();
    static int64_t unix_ms();
    static std::string generate_internal_token();

    Child* find_locked(const std::string& id);
    const Child* find_locked(const std::string& id) const;
    std::string bin_path(const Child& child) const;
    bool spawn_locked(Child* child, std::string* err, std::unique_lock<std::mutex>& lock);
    void apply_exit_decision(Child* child, const RestartEngine::Decision& decision,
                             bool expected_stop);
    void handle_exit(pid_t pid, int wait_status);
    void reaper_loop();
    void signal_loop();
    void monitor_loop();
    void stop_child_signal_locked(Child* child);
    bool wait_child_exit(const std::string& id, int timeout_ms);
    /*** One kStopGraceMs budget for the leader *and* its group: wait for the
     * leader, spend the rest of the budget on the group, then SIGKILL whatever
     * is left of the group. True when the whole group is gone. Caller must not
     * hold _mu. */
    bool reap_group_with_grace(const std::string& id);
    void perform_ordered_shutdown();
    void probe_readiness(Child* child, int64_t now_ms);
    bool probe_path_of(const Child& child, std::string* path) const;
    int probe_port_of(const Child& child) const;
    void log_line(const std::string& id, const std::string& line);
    void mark_gateway_autostart_blocked_locked(const std::string& reason, bool permanent_fail);
    void mark_models_failed_locked(const std::string& reason);
    /*** Non-gateway empty server_uri: permanent fail + optional models_failed. */
    void fail_missing_server_uri_locked(Child* child);

    std::string _project_root;
    ControlConfig _cfg;
    std::string _control_config_path;
    std::string _internal_token;
    std::string _gateway_metrics_token;
    std::string _gateway_infer_token;
    std::string _gateway_admin_token;
    AutostartGate _autostart_gate;
    /*** True while autostart_all is in the model wave (not gateway phase). */
    std::atomic<bool> _autostart_models_active{false};

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
