/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor.cpp
* Date: 26-8-22
************************************************/

#include "control/supervisor.h"

#include <dirent.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>

#include "control/ready_probe.h"
#include "control/trt_spawn_gate.h"

namespace mortred {
namespace control {

namespace {

constexpr int kStopGraceMs = 5000;
constexpr int kStartingProbeIntervalMs = 500;
constexpr int kRunningProbeIntervalMs = 10000;
constexpr int kProbeTimeoutMs = 1000;
constexpr int64_t kAutostartReadyTimeoutMs = 10 * 60 * 1000;

void close_all_fds_except_stdio() {
    DIR* d = opendir("/proc/self/fd");
    if (d == nullptr) {
        return;
    }
    const int dir_fd = dirfd(d);
    struct dirent* e = nullptr;
    while ((e = readdir(d)) != nullptr) {
        char* end = nullptr;
        const long fd = std::strtol(e->d_name, &end, 10);
        if (*end == '\0' && fd > 2 && fd != dir_fd) {
            ::close(static_cast<int>(fd));
        }
    }
    closedir(d);
}

void read_pipe_loop(int fd, LogBuffer* buffer) {
    char buf[4096];
    std::string line;
    while (true) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n <= 0) {
            break;
        }
        for (ssize_t i = 0; i < n; ++i) {
            if (buf[i] == '\n') {
                if (!line.empty()) {
                    buffer->append(line);
                    line.clear();
                }
            } else {
                line.push_back(buf[i]);
            }
        }
    }
    if (!line.empty()) {
        buffer->append(line);
    }
}

}  // namespace

void ProcessSupervisor::block_supervision_signals() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGCHLD);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
}

int64_t ProcessSupervisor::monotonic_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t ProcessSupervisor::unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string ProcessSupervisor::generate_internal_token() {
    const char* env = std::getenv("MORTRED_INTERNAL_TOKEN");
    if (env != nullptr && *env != '\0') {
        return env;
    }
    std::random_device rd;
    std::ostringstream ss;
    for (int i = 0; i < 32; ++i) {
        ss << std::hex << (rd() & 0xF);
    }
    return ss.str();
}

ProcessSupervisor::ProcessSupervisor(const std::string& project_root, const ControlConfig& cfg,
                                     const std::string& control_config_path)
    : _project_root(project_root), _cfg(cfg), _control_config_path(control_config_path),
      _internal_token(generate_internal_token()) {
    // the spawned gateway and the supervisor must agree on the resolved address
    if (const char* env = std::getenv("MORTRED_GATEWAY_HOST"); env != nullptr && *env != '\0') {
        _cfg.gateway.host = env;
    }
    if (const char* env = std::getenv("MORTRED_GATEWAY_PORT"); env != nullptr && *env != '\0') {
        const int port = std::atoi(env);
        if (port > 0 && port <= 65535) {
            _cfg.gateway.port = port;
        }
    }
    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(project_root) / _cfg.supervisor.log_dir, ec);
}

ProcessSupervisor::~ProcessSupervisor() {
    if (_threads_started.load()) {
        request_shutdown();
        wait_shutdown();
    }
}

ProcessSupervisor::Child* ProcessSupervisor::find_locked(const std::string& id) {
    const auto it = _children.find(id);
    return it == _children.end() ? nullptr : it->second.get();
}

const ProcessSupervisor::Child* ProcessSupervisor::find_locked(const std::string& id) const {
    const auto it = _children.find(id);
    return it == _children.end() ? nullptr : it->second.get();
}

std::string ProcessSupervisor::bin_path(const Child& child) const {
    return (std::filesystem::path(_project_root) / _cfg.supervisor.bin_dir / child.entry.exe)
        .string();
}

void ProcessSupervisor::set_catalog(const Catalog& catalog) {
    std::lock_guard<std::mutex> lock(_mu);
    _children.clear();

    auto gateway = std::make_unique<Child>();
    gateway->id = kGatewayId;
    gateway->is_gateway = true;
    gateway->policy.enabled = true;
    gateway->policy.restart_policy = "always";
    RestartPolicyKind kind = RestartPolicyKind::kAlways;
    parse_restart_policy(gateway->policy.restart_policy, &kind);
    gateway->engine = RestartEngine(kind);
    _children[kGatewayId] = std::move(gateway);

    for (const auto& entry : catalog.entries()) {
        const ServerPolicy policy = _cfg.effective_policy(entry.id);
        if (!policy.enabled) {
            continue;
        }
        auto child = std::make_unique<Child>();
        child->id = entry.id;
        child->entry = entry;
        child->policy = policy;
        RestartPolicyKind kind = RestartPolicyKind::kOnFailure;
        parse_restart_policy(policy.restart_policy, &kind);
        child->engine = RestartEngine(kind);
        _children[entry.id] = std::move(child);
    }
}

bool ProcessSupervisor::start_threads(std::string* err) {
    if (_threads_started.exchange(true)) {
        return true;
    }
    try {
        _reaper = std::thread([this]() { reaper_loop(); });
        _signal_watcher = std::thread([this]() { signal_loop(); });
        _monitor = std::thread([this]() { monitor_loop(); });
    } catch (const std::system_error& e) {
        if (err != nullptr) {
            *err = std::string("cannot start supervision threads: ") + e.what();
        }
        _threads_stop = true;
        return false;
    }
    return true;
}

void ProcessSupervisor::log_line(const std::string& id, const std::string& line) {
    std::lock_guard<std::mutex> lock(_mu);
    Child* child = find_locked(id);
    if (child != nullptr && child->log != nullptr) {
        child->log->append("[supervisor] " + line);
    }
}

bool ProcessSupervisor::spawn_locked(Child* child, std::string* err) {
    if (!child->is_gateway) {
        if (!trt_engines_ready_for_spawn(_project_root, _cfg.supervisor.bin_dir,
                                         child->entry.config, child->policy.model_config, err)) {
            return false;
        }
    }
    const std::string exe_path = child->is_gateway
                                     ? (std::filesystem::path(_project_root) /
                                        _cfg.supervisor.bin_dir / "mortred-gateway.out")
                                           .string()
                                     : bin_path(*child);
    if (!std::filesystem::exists(exe_path)) {
        if (err != nullptr) {
            *err = "executable not found: " + exe_path;
        }
        return false;
    }

    // a previous incarnation's pipe readers must be gone before new pipes land
    if (child->reader_out.joinable()) {
        child->reader_out.join();
    }
    if (child->reader_err.joinable()) {
        child->reader_err.join();
    }

    int out_pipe[2];
    int err_pipe[2];
    if (::pipe(out_pipe) != 0) {
        if (err != nullptr) {
            *err = "pipe() failed";
        }
        return false;
    }
    if (::pipe(err_pipe) != 0) {
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        if (err != nullptr) {
            *err = "pipe() failed";
        }
        return false;
    }

    const pid_t parent_pid = ::getpid();
    const std::string ld_path = (std::filesystem::path(_project_root) / _cfg.supervisor.lib_dir)
                                    .string() +
                                ":" +
                                (std::filesystem::path(_project_root) / _cfg.supervisor.libs_dir)
                                    .string();
    const std::string bin_dir =
        (std::filesystem::path(_project_root) / _cfg.supervisor.bin_dir).string();
    const std::string internal_token = _internal_token;
    const std::string project_root = _project_root;
    const std::string control_config = _control_config_path;
    const std::string gateway_host = _cfg.gateway.host;
    const int gateway_port = _cfg.gateway.port;
    const bool is_gateway = child->is_gateway;
    const std::string exe_name = child->is_gateway ? "mortred-gateway.out" : child->entry.exe;
    const std::string config_arg = child->is_gateway ? std::string() : child->entry.config;
    const std::string model_arg = child->is_gateway ? std::string() : child->entry.model;
    const bool inject_workers = !child->is_gateway && child->policy.has_worker_nums;
    const std::string worker_str =
        inject_workers ? std::to_string(child->policy.worker_nums) : std::string();
    const std::string model_config_override =
        child->is_gateway ? std::string() : child->policy.model_config;

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);
        if (err != nullptr) {
            *err = "fork() failed";
        }
        return false;
    }
    if (pid == 0) {
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);
        close_all_fds_except_stdio();
        // die with the supervisor (graceful SIGINT path) instead of orphaning
        ::prctl(PR_SET_PDEATHSIG, SIGINT);
        if (::getppid() != parent_pid) {
            ::_exit(0);
        }
        // nohup-style parents leave SIGINT/SIGQUIT ignored; reset for exec
        ::signal(SIGINT, SIG_DFL);
        ::signal(SIGQUIT, SIG_DFL);
        // the supervisor blocks SIGCHLD/SIGINT/SIGTERM process-wide and the
        // child INHERITS that mask: without unblocking, SIG_DFL alone leaves
        // stop signals (and PDEATHSIG) pending forever and the child unkillable
        sigset_t empty_set;
        ::sigemptyset(&empty_set);
        ::sigprocmask(SIG_SETMASK, &empty_set, nullptr);
        ::setenv("LD_LIBRARY_PATH", ld_path.c_str(), 1);
        ::setenv("MORTRED_PROJECT_ROOT", project_root.c_str(), 1);
        ::setenv("MORTRED_CONTROL_CONFIG", control_config.c_str(), 1);
        if (is_gateway) {
            ::setenv("MORTRED_INTERNAL_TOKEN", internal_token.c_str(), 1);
            ::setenv("MORTRED_GATEWAY_HOST", gateway_host.c_str(), 1);
            std::string port_str = std::to_string(gateway_port);
            ::setenv("MORTRED_GATEWAY_PORT", port_str.c_str(), 1);
        } else {
            // managed children are loopback-only and protected by the internal
            // token regardless of what their TOML declares
            ::setenv("MORTRED_LISTEN_HOST", "127.0.0.1", 1);
            ::setenv("MORTRED_AUTH_TOKEN", internal_token.c_str(), 1);
            if (inject_workers) {
                ::setenv("MORTRED_WORKER_NUMS", worker_str.c_str(), 1);
            }
            if (!model_config_override.empty()) {
                ::setenv("MORTRED_MODEL_CONFIG_FILE", model_config_override.c_str(), 1);
            }
        }
        if (::chdir(bin_dir.c_str()) != 0) {
            ::_exit(127);
        }
        if (config_arg.empty()) {
            ::execl(exe_path.c_str(), exe_name.c_str(), static_cast<char*>(nullptr));
        } else if (!model_arg.empty()) {
            ::setenv("MORTRED_MODEL", model_arg.c_str(), 1);
            ::execl(exe_path.c_str(), exe_name.c_str(), "--model", model_arg.c_str(),
                    config_arg.c_str(), static_cast<char*>(nullptr));
        } else {
            ::execl(exe_path.c_str(), exe_name.c_str(), config_arg.c_str(),
                    static_cast<char*>(nullptr));
        }
        ::_exit(127);
    }

    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    const std::string log_path =
        (std::filesystem::path(_project_root) / _cfg.supervisor.log_dir /
         (child->id + ".log"))
            .string();
    if (child->log == nullptr) {
        child->log = std::make_unique<LogBuffer>(
            log_path, static_cast<size_t>(_cfg.supervisor.log_rotate_mb) * 1024 * 1024);
    } else {
        child->log->reset();
    }
    child->log->append("[supervisor] spawn pid " + std::to_string(pid));
    child->pid = pid;
    child->ready = false;
    child->started_at_unix_ms = unix_ms();
    child->last_probe_ms = 0;
    child->error.clear();
    child->engine.note_started(monotonic_ms());

    // the read ends of the log pipes are owned by the reader threads; the
    // Child fields transfer them to handle_exit, which joins the threads and
    // closes the fds after the readers observed EOF (see handle_exit)
    child->out_fd = out_pipe[0];
    child->err_fd = err_pipe[0];
    LogBuffer* buffer = child->log.get();
    child->reader_out = std::thread([fd = child->out_fd, buffer]() { read_pipe_loop(fd, buffer); });
    child->reader_err = std::thread([fd = child->err_fd, buffer]() { read_pipe_loop(fd, buffer); });
    return true;
}

bool ProcessSupervisor::start_server(const std::string& id, std::string* err) {
    std::lock_guard<std::mutex> lock(_mu);
    Child* child = find_locked(id);
    if (child == nullptr) {
        if (err != nullptr) {
            *err = "unknown server id: " + id;
        }
        return false;
    }
    if (!child->policy.enabled) {
        if (err != nullptr) {
            *err = "server disabled in mortred.toml: " + id;
        }
        return false;
    }
    if (child->pid > 0 || (child->wanted && child->engine.state() != SupervisedState::kFailed)) {
        if (err != nullptr) {
            *err = "server already running or starting: " + id;
        }
        return false;
    }
    child->wanted = true;
    child->stopping = false;
    std::string local_err;
    std::string* spawn_err = err != nullptr ? err : &local_err;
    if (!spawn_locked(child, spawn_err)) {
        if (is_trt_gate_error(*spawn_err)) {
            child->wanted = false;
            child->error = *spawn_err;
            child->engine.note_permanent_failure();
            if (child->log != nullptr) {
                child->log->append("[supervisor] " + *spawn_err);
            }
            return false;
        }
        // fork/pipe level failure: treat as an unclean exit so the restart
        // policy machinery schedules the retry
        child->engine.note_exit(monotonic_ms(), false, false);
        return false;
    }
    return true;
}

bool ProcessSupervisor::wait_child_exit(const std::string& id, int timeout_ms) {
    std::unique_lock<std::mutex> lock(_mu);
    return _cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, &id]() {
        const Child* child = find_locked(id);
        return child == nullptr || child->pid < 0;
    });
}

bool ProcessSupervisor::stop_server(const std::string& id, std::string* err) {
    {
        std::lock_guard<std::mutex> lock(_mu);
        Child* child = find_locked(id);
        if (child == nullptr) {
            if (err != nullptr) {
                *err = "unknown server id: " + id;
            }
            return false;
        }
        if (child->pid < 0) {
            if (child->engine.state() == SupervisedState::kStopped) {
                if (err != nullptr) {
                    *err = "server not running: " + id;
                }
                return false;
            }
            // cancel a pending backoff/starting restart
            child->wanted = false;
            child->stopping = false;
            child->backoff_due_ms = 0;
            child->engine.note_cancel();
            _cv.notify_all();
            return true;
        }
        child->wanted = false;
        child->stopping = true;
        child->backoff_due_ms = 0;
        ::kill(child->pid, SIGINT);
    }
    if (!wait_child_exit(id, kStopGraceMs)) {
        std::lock_guard<std::mutex> lock(_mu);
        Child* child = find_locked(id);
        if (child != nullptr && child->pid > 0) {
            ::kill(child->pid, SIGKILL);
        }
        wait_child_exit(id, 2000);
    }
    log_line(id, "stop requested via api");
    return true;
}

bool ProcessSupervisor::restart_server(const std::string& id, std::string* err) {
    std::string stop_err;
    stop_server(id, &stop_err);  // "not running" is fine
    return start_server(id, err);
}

bool ProcessSupervisor::has_server(const std::string& id) const {
    std::lock_guard<std::mutex> lock(_mu);
    return find_locked(id) != nullptr;
}

ProcessSupervisor::Status ProcessSupervisor::status(const std::string& id) const {
    Status s;
    std::lock_guard<std::mutex> lock(_mu);
    const Child* child = find_locked(id);
    if (child == nullptr) {
        s.state = "unknown";
        return s;
    }
    s.state = to_string(child->engine.state());
    s.pid = child->pid > 0 ? static_cast<int>(child->pid) : -1;
    s.ready = child->ready;
    s.restart_count = child->engine.restart_count();
    s.last_exit_status = child->last_exit_status;
    s.has_last_exit = child->has_last_exit;
    s.started_at_unix_ms = child->pid > 0 ? child->started_at_unix_ms : 0;
    s.error = child->error;
    return s;
}

std::vector<std::pair<std::string, ProcessSupervisor::Status>> ProcessSupervisor::statuses()
    const {
    std::vector<std::pair<std::string, Status>> out;
    std::lock_guard<std::mutex> lock(_mu);
    for (const auto& [id, child] : _children) {
        Status s;
        s.state = to_string(child->engine.state());
        s.pid = child->pid > 0 ? static_cast<int>(child->pid) : -1;
        s.ready = child->ready;
        s.restart_count = child->engine.restart_count();
        s.last_exit_status = child->last_exit_status;
        s.has_last_exit = child->has_last_exit;
        s.started_at_unix_ms = child->pid > 0 ? child->started_at_unix_ms : 0;
        s.error = child->error;
        out.emplace_back(id, std::move(s));
    }
    return out;
}

LogBuffer* ProcessSupervisor::logs(const std::string& id) {
    std::lock_guard<std::mutex> lock(_mu);
    Child* child = find_locked(id);
    return child == nullptr ? nullptr : child->log.get();
}

int ProcessSupervisor::probe_port_of(const Child& child) const {
    return child.is_gateway ? _cfg.gateway.port : child.entry.port;
}

bool ProcessSupervisor::probe_path_of(const Child& child, std::string* path) const {
    if (child.is_gateway) {
        *path = "/healthz";
        return true;
    }
    if (child.entry.uri.empty()) {
        return false;
    }
    *path = "/ready";
    return true;
}

void ProcessSupervisor::probe_readiness(Child* child, int64_t now_ms) {
    std::string path;
    const int port = probe_port_of(*child);
    if (port <= 0 || !probe_path_of(*child, &path)) {
        return;
    }
    const bool ready = endpoint_ready(port, path.c_str(), kProbeTimeoutMs);

    std::lock_guard<std::mutex> lock(_mu);
    if (child->pid < 0) {
        return;  // exited while probing
    }
    child->ready = ready;
    if (ready && child->engine.state() == SupervisedState::kStarting) {
        child->engine.note_ready(now_ms);
    }
}

void ProcessSupervisor::handle_exit(pid_t pid, int wait_status) {
    bool expected = false;
    bool restart = false;
    int delay_ms = 0;
    Child* child = nullptr;
    int out_fd = -1;
    int err_fd = -1;
    {
        std::lock_guard<std::mutex> lock(_mu);
        for (auto& [id, c] : _children) {
            (void)id;
            if (c->pid == pid) {
                child = c.get();
                break;
            }
        }
        if (child == nullptr) {
            return;  // not ours (already reaped via waitpid)
        }
        child->pid = -1;
        child->ready = false;
        if (WIFEXITED(wait_status)) {
            child->last_exit_status = WEXITSTATUS(wait_status);
        } else if (WIFSIGNALED(wait_status)) {
            child->last_exit_status = 128 + WTERMSIG(wait_status);
        }
        child->has_last_exit = true;
        expected = child->stopping;
        child->stopping = false;
        const bool clean = WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0;
        const auto decision = child->engine.note_exit(monotonic_ms(), clean, expected);
        restart = decision.restart && child->wanted && !_shutdown_requested.load();
        delay_ms = decision.delay_ms;
        if (decision.gave_up) {
            child->error = "crash loop detected: >5 restarts within 60s; manual start required";
            child->wanted = false;
            child->log->append("[supervisor] " + child->error);
        } else if (restart) {
            child->backoff_due_ms = monotonic_ms() + delay_ms;
        } else {
            child->backoff_due_ms = 0;
            if (!expected) {
                child->wanted = false;
            }
        }
        out_fd = child->out_fd;
        err_fd = child->err_fd;
        child->out_fd = -1;
        child->err_fd = -1;
    }

    // join readers without holding the table lock, then release the pipe fds
    if (child->reader_out.joinable()) {
        child->reader_out.join();
    }
    if (child->reader_err.joinable()) {
        child->reader_err.join();
    }
    if (out_fd >= 0) {
        ::close(out_fd);
    }
    if (err_fd >= 0) {
        ::close(err_fd);
    }
    _cv.notify_all();
}

void ProcessSupervisor::reaper_loop() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGCHLD);
    while (!_threads_stop.load()) {
        timespec ts{0, 200 * 1000 * 1000};
        siginfo_t info{};
        const int rc = ::sigtimedwait(&set, &info, &ts);
        if (rc < 0 && errno != EAGAIN) {
            continue;
        }
        while (true) {
            int status = 0;
            const pid_t pid = ::waitpid(-1, &status, WNOHANG);
            if (pid <= 0) {
                break;
            }
            handle_exit(pid, status);
        }
    }
}

void ProcessSupervisor::signal_loop() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    while (!_threads_stop.load()) {
        timespec ts{0, 200 * 1000 * 1000};
        siginfo_t info{};
        const int rc = ::sigtimedwait(&set, &info, &ts);
        if (rc == SIGINT || rc == SIGTERM) {
            request_shutdown();
            return;
        }
        if (rc < 0 && errno != EAGAIN) {
            continue;
        }
    }
}

void ProcessSupervisor::request_shutdown() {
    _shutdown_requested.store(true);
    std::lock_guard<std::mutex> lock(_mu);
    _cv.notify_all();
}

void ProcessSupervisor::perform_ordered_shutdown() {
    // phase 1: model servers (all at once, they are independent)
    std::vector<std::string> model_ids;
    {
        std::lock_guard<std::mutex> lock(_mu);
        for (auto& [id, child] : _children) {
            if (child->is_gateway) {
                continue;
            }
            model_ids.push_back(id);
            child->wanted = false;
            child->backoff_due_ms = 0;
            if (child->pid < 0) {
                child->engine.note_cancel();
            } else {
                child->stopping = true;
                ::kill(child->pid, SIGINT);
            }
        }
    }
    for (const auto& id : model_ids) {
        if (!wait_child_exit(id, kStopGraceMs)) {
            std::lock_guard<std::mutex> lock(_mu);
            Child* child = find_locked(id);
            if (child != nullptr && child->pid > 0) {
                ::kill(child->pid, SIGKILL);
            }
        }
        wait_child_exit(id, 2000);
    }
    // phase 2: gateway last
    {
        std::lock_guard<std::mutex> lock(_mu);
        Child* gateway = find_locked(kGatewayId);
        if (gateway != nullptr) {
            gateway->wanted = false;
            gateway->backoff_due_ms = 0;
            if (gateway->pid < 0) {
                gateway->engine.note_cancel();
            } else {
                gateway->stopping = true;
                ::kill(gateway->pid, SIGINT);
            }
        }
    }
    if (!wait_child_exit(kGatewayId, kStopGraceMs)) {
        std::lock_guard<std::mutex> lock(_mu);
        Child* gateway = find_locked(kGatewayId);
        if (gateway != nullptr && gateway->pid > 0) {
            ::kill(gateway->pid, SIGKILL);
        }
    }
    wait_child_exit(kGatewayId, 2000);

    std::lock_guard<std::mutex> lock(_mu);
    _shutdown_done = true;
    _cv.notify_all();
}

void ProcessSupervisor::monitor_loop() {
    while (!_threads_stop.load()) {
        if (_shutdown_requested.load()) {
            perform_ordered_shutdown();
            return;
        }
        const int64_t now = monotonic_ms();
        int64_t wake_at = now + 500;
        std::vector<Child*> to_probe;
        std::vector<Child*> to_restart;
        {
            std::lock_guard<std::mutex> lock(_mu);
            for (auto& [id, child] : _children) {
                (void)id;
                if (child->pid > 0) {
                    const SupervisedState st = child->engine.state();
                    const int interval =
                        st == SupervisedState::kStarting ? kStartingProbeIntervalMs
                                                         : kRunningProbeIntervalMs;
                    if (st == SupervisedState::kStarting || st == SupervisedState::kRunning) {
                        if (now - child->last_probe_ms >= interval) {
                            child->last_probe_ms = now;
                            to_probe.push_back(child.get());
                        }
                        if (child->last_probe_ms + interval < wake_at) {
                            wake_at = child->last_probe_ms + interval;
                        }
                    }
                } else if (child->backoff_due_ms > 0 && child->wanted &&
                           !child->stopping && !_shutdown_requested.load()) {
                    if (now >= child->backoff_due_ms) {
                        to_restart.push_back(child.get());
                    } else if (child->backoff_due_ms < wake_at) {
                        wake_at = child->backoff_due_ms;
                    }
                }
            }
        }
        for (Child* child : to_probe) {
            probe_readiness(child, monotonic_ms());
        }
        if (!to_restart.empty()) {
            std::lock_guard<std::mutex> lock(_mu);
            for (Child* child : to_restart) {
                child->backoff_due_ms = 0;
                std::string err;
                if (!spawn_locked(child, &err)) {
                    if (is_trt_gate_error(err)) {
                        child->wanted = false;
                        child->error = err;
                        child->engine.note_permanent_failure();
                        if (child->log != nullptr) {
                            child->log->append("[supervisor] " + err);
                        }
                    } else {
                        child->engine.note_exit(monotonic_ms(), false, false);
                        if (child->log != nullptr) {
                            child->log->append("[supervisor] respawn failed: " + err);
                        }
                    }
                } else if (child->log != nullptr) {
                    child->log->append("[supervisor] respawned after backoff");
                }
            }
        }
        std::unique_lock<std::mutex> lock(_mu);
        const int64_t wait_ms = std::max<int64_t>(1, wake_at - monotonic_ms());
        _cv.wait_for(lock, std::chrono::milliseconds(std::min<int64_t>(wait_ms, 500)));
    }
}

void ProcessSupervisor::wait_shutdown() {
    if (!_threads_started.load()) {
        return;
    }
    if (_monitor.joinable()) {
        std::unique_lock<std::mutex> lock(_mu);
        _cv.wait(lock, [this]() { return _shutdown_done; });
    }
    _threads_stop.store(true);
    _cv.notify_all();
    if (_monitor.joinable()) {
        _monitor.join();
    }
    if (_reaper.joinable()) {
        _reaper.join();
    }
    if (_signal_watcher.joinable()) {
        _signal_watcher.join();
    }
}

void ProcessSupervisor::autostart_all() {
    // gateway first: routing must exist before clients can reach models
    {
        const std::string gateway_bin =
            (std::filesystem::path(_project_root) / _cfg.supervisor.bin_dir / "mortred-gateway.out")
                .string();
        if (std::filesystem::exists(gateway_bin)) {
            std::string err;
            start_server(kGatewayId, &err);
            const int64_t deadline = monotonic_ms() + 15000;
            while (monotonic_ms() < deadline && !_shutdown_requested.load()) {
                const Status s = status(kGatewayId);
                if (s.state == "running" || s.pid < 0) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
    }

    // models with concurrency width; a slot frees on ready or terminal state
    std::vector<std::string> queue;
    for (const auto& [id, s] : statuses()) {
        (void)s;
        if (id == kGatewayId) {
            continue;
        }
        const ServerPolicy policy = _cfg.effective_policy(id);
        if (policy.autostart) {
            queue.push_back(id);
        }
    }
    const size_t width = static_cast<size_t>(_cfg.supervisor.start_concurrency);
    auto unresolved_count = [this, &queue](size_t launched) {
        size_t n = 0;
        for (size_t i = 0; i < launched && i < queue.size(); ++i) {
            const Status s = status(queue[i]);
            if (s.pid >= 0 && s.state != "running") {
                ++n;
            }
        }
        return n;
    };
    size_t launched = 0;
    while (launched < queue.size() || unresolved_count(launched) > 0) {
        if (_shutdown_requested.load()) {
            return;
        }
        while (launched < queue.size() && unresolved_count(launched) < width) {
            const std::string& id = queue[launched++];
            std::string err;
            start_server(id, &err);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

}  // namespace control
}  // namespace mortred
