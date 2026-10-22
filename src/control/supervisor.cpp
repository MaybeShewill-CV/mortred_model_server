/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor.cpp
* Date: 26-8-22
************************************************/

#include "control/supervisor.h"

#include <signal.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "control/occupancy_gate.h"
#include "control/gpu_probe.h"
#include "control/ready_probe.h"
#include "control/trust_tokens.h"
#include "control/trt_spawn_gate.h"

extern "C" char** environ;

namespace mortred {
namespace control {

bool process_group_gone(pid_t pgid) {
    if (pgid <= 0) {
        return true;  // no group was ever recorded: nothing to wait for
    }
    return ::killpg(pgid, 0) == -1 && errno == ESRCH;
}

namespace {

constexpr int kStopGraceMs = 5000;
constexpr int kStartingProbeIntervalMs = 500;
constexpr int kRunningProbeIntervalMs = 10000;
constexpr int kProbeTimeoutMs = 1000;

void override_env_entry(std::vector<std::string>& block, const char* key,
                        const std::string& value) {
    const std::string prefix = std::string(key) + "=";
    for (std::string& entry : block) {
        if (entry.compare(0, prefix.size(), prefix) == 0) {
            entry = prefix + value;
            return;
        }
    }
    block.emplace_back(prefix + value);
}

// Copy one non-secret supervisor env var into the child block. Empty and
// unset values stay absent so the child keeps its own default.
void forward_supervisor_env(std::vector<std::string> &block, const char *key) {
    const char *value = std::getenv(key);
    if (value == nullptr || *value == '\0') {
        return;
    }
    override_env_entry(block, key, value);
}

// Remove every entry whose name matches key exactly. Exact match matters: a
// prefix test would also drop unrelated keys such as MORTRED_API_TOKEN_FILE.
// Deliver sig to a managed child's whole process group so that a grandchild
// dies with its leader. pgid <= 0 means no group was recorded (a genuine
// setpgid failure): fall back to the single pid, never to the supervisor's own
// group.
void signal_process_group(pid_t pgid, pid_t pid, int sig) {
    if (pgid > 0) {
        ::killpg(pgid, sig);
        return;
    }
    if (pid > 0) {
        ::kill(pid, sig);
    }
}

// Bounded poll of process_group_gone: no busy spin, no lock held.
bool wait_process_group_gone(pid_t pgid, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 0);
    while (!process_group_gone(pgid)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
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

bool is_permanent_spawn_error(const std::string& err) {
    return is_trt_gate_error(err) || is_scrape_token_error(err) || is_occupancy_gate_error(err);
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

void ProcessSupervisor::set_gateway_trust(const std::string& metrics_token,
                                          const std::string& infer_token,
                                          const std::string& admin_token) {
    std::lock_guard<std::mutex> lock(_mu);
    _gateway_metrics_token = metrics_token;
    _gateway_infer_token = infer_token;
    _gateway_admin_token = admin_token;
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

void ProcessSupervisor::apply_exit_decision(Child* child,
                                            const RestartEngine::Decision& decision,
                                            bool expected_stop) {
    const bool restart = decision.restart && child->wanted && !_shutdown_requested.load();
    if (decision.gave_up) {
        child->error = "crash loop detected: >5 restarts within 60s; manual start required";
        child->wanted = false;
        child->backoff_due_ms = 0;
        if (child->log != nullptr) {
            child->log->append("[supervisor] " + child->error);
        }
    } else if (restart) {
        child->backoff_due_ms = monotonic_ms() + decision.delay_ms;
    } else {
        child->backoff_due_ms = 0;
        if (!expected_stop) {
            child->wanted = false;
        }
    }
}

bool ProcessSupervisor::spawn_locked(Child* child, std::string* err) {
    if (child->is_gateway) {
        const auto listen = tokens_usable_for_listen(_gateway_metrics_token, _gateway_infer_token,
                                                     _gateway_admin_token);
        if (!listen.ok) {
            if (err != nullptr) {
                *err = scrape_token_spawn_error(listen);
            }
            return false;
        }
    } else if (!trt_engines_ready_for_spawn(_project_root, _cfg.supervisor.bin_dir,
                                            child->entry.config, child->policy.model_config,
                                            err)) {
        return false;
    } else {
        OccupancyCheckInput occ;
        occ.pack_active = _cfg.supervisor.pack_active;
        occ.pack_path = _cfg.supervisor.pack_file;
        occ.occupancy_policy = _cfg.occupancy.policy;
        occ.gpu_reserve_pct = _cfg.occupancy.gpu_reserve_pct;
        occ.gpu_name = _cfg.occupancy.gpu_name;
        occ.gpu_memory_total_mib = _cfg.occupancy.gpu_memory_total_mib;
        occ.candidate_id = child->id;
        occ.candidate_is_tensorrt =
            model_uses_tensorrt(_project_root, _cfg.supervisor.bin_dir, child->entry.config,
                                child->policy.model_config);
        occ.worker_nums = child->policy.has_worker_nums ? child->policy.worker_nums : 0;
        occ.has_gpu_mem_mib = child->policy.has_gpu_mem_mib;
        occ.gpu_mem_mib = child->policy.gpu_mem_mib;
        occ.gpu_mem_at_workers = child->policy.gpu_mem_at_workers;
        occ.has_rss_mib = child->policy.has_rss_mib;
        occ.rss_mib = child->policy.rss_mib;
        occ.host_memory_total_mib = _cfg.occupancy.host_memory_total_mib;
        occ.ram_reserve_pct = _cfg.occupancy.ram_reserve_pct;
        for (const auto& [id, other] : _children) {
            if (other == nullptr || other->is_gateway || other->id == child->id) {
                continue;
            }
            if (other->pid > 0 &&
                ((other->policy.has_gpu_mem_mib && other->policy.gpu_mem_mib > 0) ||
                 (other->policy.has_rss_mib && other->policy.rss_mib > 0))) {
                OccupancySibling sib;
                sib.id = other->id;
                sib.gpu_mem_mib = other->policy.gpu_mem_mib;
                sib.rss_mib = other->policy.rss_mib;
                occ.running_siblings.push_back(sib);
            }
        }
        occ.live = query_live_gpu_via_shared_probe();
        occ.live_mem_available_mib = query_mem_available_mib();
        // Fail closed on a missing GPU probe unless the operator opts out
        // explicitly. Without live telemetry the fingerprint and free-memory
        // checks cannot run, so the gate would otherwise "verify" nothing.
        if (const char* allow = std::getenv("MORTRED_ALLOW_UNKNOWN_GPU");
            allow != nullptr && (std::strcmp(allow, "1") == 0 || std::strcmp(allow, "true") == 0 ||
                                 std::strcmp(allow, "TRUE") == 0 ||
                                 std::strcmp(allow, "yes") == 0)) {
            occ.allow_unknown_gpu = true;
        }
        if (!occupancy_ready_for_spawn(occ, err)) {
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

    // No wait for previous readers: handle_exit moved them into a detached
    // joiner that holds the ended run's buffer, and the swap below hands this
    // spawn a brand-new LogBuffer over an archived file. Old and new readers
    // can therefore never touch the same buffer, so spawn never stalls on a
    // straggler draining a pipe over a stalled filesystem.

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
    std::string control_config = _control_config_path;
    {
        std::filesystem::path cfg_path(control_config);
        if (!control_config.empty() && !cfg_path.is_absolute()) {
            cfg_path = std::filesystem::path(project_root) / cfg_path;
            control_config = cfg_path.lexically_normal().string();
        }
    }
    const std::string gateway_host = _cfg.gateway.host;
    const int gateway_port = _cfg.gateway.port;
    const std::string gateway_metrics_token = _gateway_metrics_token;
    const std::string gateway_infer_token = _gateway_infer_token;
    const std::string gateway_admin_token = _gateway_admin_token;
    const bool is_gateway = child->is_gateway;
    const std::string exe_name = child->is_gateway ? "mortred-gateway.out" : child->entry.exe;
    const std::string config_arg = child->is_gateway ? std::string() : child->entry.config;
    const std::string model_arg = child->is_gateway ? std::string() : child->entry.model;
    const bool inject_workers = !child->is_gateway && child->policy.has_worker_nums;
    const std::string worker_str =
        inject_workers ? std::to_string(child->policy.worker_nums) : std::string();
    const std::string model_config_override =
        child->is_gateway ? std::string() : child->policy.model_config;
    const std::string decode_auto =
        child->is_gateway ? std::string() : child->policy.decode_auto;
    const std::string gateway_port_str = std::to_string(gateway_port);

    int max_fd = 1024;
    struct rlimit nofile_limit {};
    if (::getrlimit(RLIMIT_NOFILE, &nofile_limit) == 0) {
        const rlim_t cur = nofile_limit.rlim_cur;
        if (cur >= static_cast<rlim_t>(3) && cur <= static_cast<rlim_t>(1048576)) {
            max_fd = static_cast<int>(cur);
        }
    }

    // Allowlist child environ : never copy the supervisor's full
    // environ. Only pass a fixed set of process/runtime keys, then inject
    // the role-specific MORTRED_* credentials below. A drop-list cannot
    // keep up with new secret names; an allowlist can.
    static const char* kChildEnvAllow[] = {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
        "TZ", "TERM", "TMPDIR", "TMP", "TEMP",
        "LD_LIBRARY_PATH", "LD_PRELOAD",
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES", "CUDA_MODULE_LOADING",
        nullptr,
    };
    std::vector<std::string> env_storage;
    if (environ != nullptr) {
        for (char** ep = environ; *ep != nullptr; ++ep) {
            const char* entry = *ep;
            const char* eq = std::strchr(entry, '=');
            if (eq == nullptr) {
                continue;
            }
            const size_t key_len = static_cast<size_t>(eq - entry);
            bool allowed = false;
            for (const char** a = kChildEnvAllow; *a != nullptr; ++a) {
                if (std::strlen(*a) == key_len && std::strncmp(entry, *a, key_len) == 0) {
                    allowed = true;
                    break;
                }
            }
            if (allowed) {
                env_storage.emplace_back(entry);
            }
        }
    }
    override_env_entry(env_storage, "LD_LIBRARY_PATH", ld_path);
    override_env_entry(env_storage, "MORTRED_PROJECT_ROOT", project_root);
    override_env_entry(env_storage, "MORTRED_CONTROL_CONFIG", control_config);
    if (is_gateway) {
        override_env_entry(env_storage, "MORTRED_INTERNAL_TOKEN", internal_token);
        override_env_entry(env_storage, "MORTRED_GATEWAY_HOST", gateway_host);
        override_env_entry(env_storage, "MORTRED_GATEWAY_PORT", gateway_port_str);
        if (!gateway_metrics_token.empty()) {
            override_env_entry(env_storage, "MORTRED_METRICS_TOKEN", gateway_metrics_token);
        }
        if (!gateway_infer_token.empty()) {
            override_env_entry(env_storage, "MORTRED_GATEWAY_AUTH_TOKEN", gateway_infer_token);
        }
        if (!gateway_admin_token.empty()) {
            override_env_entry(env_storage, "MORTRED_API_TOKEN", gateway_admin_token);
        }
        // Compose and docker_entrypoint set these on the supervisor process.
        // The allowlist above does not copy them, and the gateway does not
        // inherit the supervisor environ. Without MORTRED_EXPOSE the injected
        // MORTRED_GATEWAY_HOST (0.0.0.0 in containers) fails listen_host_permitted
        // and mortred-gateway.out exits 1 before listen(). Without
        // MORTRED_PROFILE the gateway catalog defaults to gpu and drops
        // profile=cpu routes (the CI pack). Model children stay off this path:
        // they are forced onto 127.0.0.1 and must not inherit a docker expose.
        forward_supervisor_env(env_storage, "MORTRED_EXPOSE");
        forward_supervisor_env(env_storage, "MORTRED_PROFILE");
    } else {
        // managed children are loopback-only and protected by the internal
        // token regardless of what their TOML declares
        override_env_entry(env_storage, "MORTRED_LISTEN_HOST", "127.0.0.1");
        override_env_entry(env_storage, "MORTRED_AUTH_TOKEN", internal_token);
        // Allow Prometheus scrapers to use the same MORTRED_METRICS_TOKEN on
        // model :port/metrics (not only gateway). Still distinct from infer token.
        if (!gateway_metrics_token.empty()) {
            override_env_entry(env_storage, "MORTRED_METRICS_TOKEN", gateway_metrics_token);
        }
        if (inject_workers) {
            override_env_entry(env_storage, "MORTRED_WORKER_NUMS", worker_str);
        }
        if (!decode_auto.empty()) {
            override_env_entry(env_storage, "MORTRED_DECODE_AUTO", decode_auto);
        }
        if (!model_config_override.empty()) {
            override_env_entry(env_storage, "MORTRED_MODEL_CONFIG_FILE", model_config_override);
        }
    }

    const std::string model_flag = "--model";
    char* child_argv[5] = {};
    int argi = 0;
    child_argv[argi++] = const_cast<char*>(exe_name.c_str());
    if (config_arg.empty()) {
        // executable name only
    } else if (!model_arg.empty()) {
        child_argv[argi++] = const_cast<char*>(model_flag.c_str());
        child_argv[argi++] = const_cast<char*>(model_arg.c_str());
        child_argv[argi++] = const_cast<char*>(config_arg.c_str());
        override_env_entry(env_storage, "MORTRED_MODEL", model_arg);
    } else {
        child_argv[argi++] = const_cast<char*>(config_arg.c_str());
    }
    child_argv[argi] = nullptr;

    std::vector<char*> env_ptrs;
    env_ptrs.reserve(env_storage.size() + 1);
    for (std::string& entry : env_storage) {
        env_ptrs.push_back(entry.data());
    }
    env_ptrs.push_back(nullptr);
    const char* child_exe = exe_path.c_str();
    const char* child_bin_dir = bin_dir.c_str();
    char** child_envp = env_ptrs.data();

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
    pid_t child_pgid = -1;
    if (pid > 0) {
        // Parent side of the process-group handshake: the child leads its own
        // group (pgid == pid) so the stop/restart/shutdown sites can reach a
        // grandchild that the leader would otherwise orphan. Both sides call
        // setpgid, so no ordering can leave the child in our group. EACCES =
        // the child already exec'd (its own call ran), EPERM/ESRCH = already a
        // group leader or already gone; in every tolerated case pgid == pid is
        // still the truth.
        if (::setpgid(pid, pid) != 0 && errno != EACCES && errno != EPERM && errno != ESRCH) {
            child_pgid = -1;  // no group: delivery falls back to the single pid
        } else {
            child_pgid = pid;
        }
    }
    if (pid == 0) {
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);
        for (int fd = 3; fd < max_fd; ++fd) {
            ::close(fd);
        }
        // Child side of the same handshake. It must stay inside this
        // post-fork/pre-exec block: setpgid(2) is async-signal-safe (integer
        // arguments only, no libc state), so completion-standard row ST5 still
        // holds - nothing non-async-signal-safe runs between fork and execve.
        ::setpgid(0, 0);
        // die with the supervisor (graceful SIGINT path) instead of orphaning.
        // syscall, not libc prctl: this stays after the fd close and before execve.
        ::syscall(SYS_prctl, PR_SET_PDEATHSIG /* orphan self-exit when supervisor dies */, SIGINT);
        if (::getppid() != parent_pid) {
            ::_exit(0);
        }
        // nohup-style parents leave SIGINT/SIGQUIT ignored; reset for exec
        struct sigaction sa {};
        ::sigemptyset(&sa.sa_mask);
        sa.sa_handler = SIG_DFL;
        sa.sa_flags = 0;
        ::sigaction(SIGINT, &sa, nullptr);
        ::sigaction(SIGQUIT, &sa, nullptr);
        // the supervisor blocks SIGCHLD/SIGINT/SIGTERM process-wide and the
        // child INHERITS that mask: without unblocking, SIG_DFL alone leaves
        // stop signals (and PDEATHSIG) pending forever and the child unkillable
        sigset_t empty_set;
        ::sigemptyset(&empty_set);
        ::sigprocmask(SIG_SETMASK, &empty_set, nullptr);
        if (::chdir(child_bin_dir) != 0) {
            ::_exit(127);
        }
        ::execve(child_exe, child_argv, child_envp);
        ::_exit(127);
    }

    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    const std::string log_path =
        (std::filesystem::path(_project_root) / _cfg.supervisor.log_dir /
         (child->id + ".log"))
            .string();
    if (child->log != nullptr) {
        // archive the ended run (rename + retention sweep); a reader straggler
        // still holding the old buffer appends into the archived inode only
        child->log->rollover(static_cast<uint64_t>(unix_ms()));
    }
    child->log = std::make_shared<LogBuffer>(
        log_path, static_cast<size_t>(_cfg.supervisor.log_rotate_mb) * 1024 * 1024);
    child->log->append("[supervisor] spawn pid " + std::to_string(pid));
    child->pid = pid;
    child->pgid = child_pgid;
    child->ready = false;
    child->started_at_unix_ms = unix_ms();
    child->last_probe_ms = 0;
    child->error.clear();
    child->engine.note_started(monotonic_ms());

    // Readers + fds are owned until handle_exit joins/closes them. The reader
    // threads hold this run's buffer by value: they outlive any later swap.
    child->out_fd = out_pipe[0];
    child->err_fd = err_pipe[0];
    std::shared_ptr<LogBuffer> buffer = child->log;
    child->reader_out = std::thread([fd = child->out_fd, buffer]() { read_pipe_loop(fd, buffer.get()); });
    child->reader_err = std::thread([fd = child->err_fd, buffer]() { read_pipe_loop(fd, buffer.get()); });
    return true;
}

bool ProcessSupervisor::start_server(const std::string& id, std::string* err) {
    std::unique_lock<std::mutex> lock(_mu);
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
        if (is_permanent_spawn_error(*spawn_err)) {
            child->wanted = false;
            child->error = *spawn_err;
            child->engine.note_permanent_failure();
            if (child->log != nullptr) {
                child->log->append("[supervisor] " + *spawn_err);
            }
            return false;
        }
        const auto decision = child->engine.note_exit(monotonic_ms(), false, false);
        apply_exit_decision(child, decision, false);
        if (child->log != nullptr) {
            child->log->append("[supervisor] spawn failed: " + *spawn_err);
        }
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

bool ProcessSupervisor::reap_group_with_grace(const std::string& id) {
    pid_t pgid = -1;
    {
        std::lock_guard<std::mutex> lock(_mu);
        Child* child = find_locked(id);
        if (child != nullptr) {
            pgid = child->pgid;
        }
    }
    // One grace budget for the leader *and* its group. The leader's exit alone
    // does not end the stop: a grandchild that ignores SIGINT keeps the group
    // alive after the leader is reaped, so the rest of the budget is spent on
    // the group and whatever is left is SIGKILLed as a group. The leader's own
    // timeout still escalates on its own, which keeps the pgid <= 0 fallback
    // (setpgid failed, so there is no group to inspect) as safe as before.
    const int64_t deadline = monotonic_ms() + kStopGraceMs;
    const bool leader_exited = wait_child_exit(id, kStopGraceMs);
    const int64_t remaining = deadline - monotonic_ms();
    const bool group_gone =
        wait_process_group_gone(pgid, remaining > 0 ? static_cast<int>(remaining) : 0);
    if (!leader_exited || !group_gone) {
        {
            std::lock_guard<std::mutex> lock(_mu);
            Child* child = find_locked(id);
            if (child != nullptr) {
                signal_process_group(child->pgid, child->pid, SIGKILL);
            }
        }
        wait_child_exit(id, 2000);
        return wait_process_group_gone(pgid, 2000);
    }
    return true;
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
        // whole group: helpers a child forked must not outlive the child
        signal_process_group(child->pgid, child->pid, SIGINT);
    }
    const bool group_gone = reap_group_with_grace(id);
    log_line(id, "stop requested via api");
    if (!group_gone) {
        // process_group_gone could not confirm an empty group: report it rather
        // than pretend the whole tree stopped.
        log_line(id, "stop: process group survived grace+2s; check for surviving descendants");
    }
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

std::shared_ptr<LogBuffer> ProcessSupervisor::logs(const std::string& id) {
    std::lock_guard<std::mutex> lock(_mu);
    Child* child = find_locked(id);
    return child == nullptr ? nullptr : child->log;
}

ProcessSupervisor::AutostartGate ProcessSupervisor::autostart_gate() const {
    std::lock_guard<std::mutex> lock(_mu);
    return _autostart_gate;
}

void ProcessSupervisor::mark_gateway_autostart_blocked_locked(const std::string& reason,
                                                             bool permanent_fail) {
    _autostart_gate.blocked = true;
    _autostart_gate.models_skipped = true;
    _autostart_gate.models_failed = false;
    _autostart_gate.reason = reason;
    Child* gateway = find_locked(kGatewayId);
    if (gateway == nullptr) {
        return;
    }
    if (gateway->error.empty()) {
        gateway->error = reason;
    }
    if (permanent_fail && gateway->engine.state() != SupervisedState::kFailed) {
        gateway->wanted = false;
        gateway->engine.note_permanent_failure();
    }
    if (gateway->log != nullptr) {
        gateway->log->append("[supervisor] autostart blocked: " + reason);
    }
}

void ProcessSupervisor::mark_models_failed_locked(const std::string& reason) {
    // Gateway already cleared this wave; model-wave failure must not look like a skip.
    _autostart_gate.models_failed = true;
    _autostart_gate.models_skipped = false;
    if (_autostart_gate.reason.empty()) {
        _autostart_gate.reason = reason;
    } else {
        _autostart_gate.reason += "; " + reason;
    }
}

void ProcessSupervisor::fail_missing_server_uri_locked(Child* child) {
    if (child == nullptr || child->is_gateway) {
        return;
    }
    if (child->engine.state() == SupervisedState::kFailed &&
        child->error.find("server_uri") != std::string::npos) {
        return;
    }
    // Point at missing server_uri and that HTTP /ready was never attempted.
    const std::string reason = "missing server_uri; HTTP readiness probe skipped";
    child->error = reason;
    child->ready = false;
    child->wanted = false;
    child->backoff_due_ms = 0;
    child->engine.note_permanent_failure();
    if (child->log != nullptr) {
        child->log->append("[supervisor] " + reason);
    }
    if (_autostart_models_active.load()) {
        mark_models_failed_locked("model failed: " + child->id + ": " + reason);
    }
    // Drop the whole process group without stopping=true so note_exit keeps
    // Failed: the child may already have forked helpers of its own.
    signal_process_group(child->pgid, child->pid, SIGKILL);
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
    // Empty server_uri cannot form a /ready probe: fail immediately.
    // Hot path: no HTTP round-trip, no image/preprocess work — config gate only.
    if (!child->is_gateway && child->entry.uri.empty()) {
        std::lock_guard<std::mutex> lock(_mu);
        fail_missing_server_uri_locked(child);
        return;
    }

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
        // Child stderr is still in the log pipe at reap time, so a refuse-to-start
        // message is not visible yet. Record the wait status before the next
        // spawn clears error; a crash-loop give-up overwrites this below.
        if (!expected && child->error.empty()) {
            std::string reason;
            if (WIFEXITED(wait_status) && WEXITSTATUS(wait_status) != 0) {
                reason = "exited " + std::to_string(WEXITSTATUS(wait_status));
            } else if (WIFSIGNALED(wait_status)) {
                reason = "killed by signal " + std::to_string(WTERMSIG(wait_status));
            }
            if (!reason.empty()) {
                child->error = reason;
                if (child->log != nullptr) {
                    child->log->append("[supervisor] " + child->error);
                }
            }
        }
        const bool clean = WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0;
        const auto decision = child->engine.note_exit(monotonic_ms(), clean, expected);
        apply_exit_decision(child, decision, expected);
        out_fd = child->out_fd;
        err_fd = child->err_fd;
        child->out_fd = -1;
        child->err_fd = -1;
    }
    // wake wait_child_exit() and every other _cv predicate that polls the
    // child state updated above. This notify used to come indirectly from the
    // log-reader joiner (reader EOF -> join -> notify), which made a prompt
    // stop depend on pipe EOF timing; a stalled reader would have burned the
    // whole stop grace budget.
    _cv.notify_all();

    // Close pipe fds first so readers blocked in read() wake up. Never join
    // readers on the reaper thread: a stalled LogBuffer write must not stop
    // SIGCHLD reaping for other children. Detach a joiner that is the sole
    // join site; it also holds the ended run's buffer, keeping it alive until
    // both readers finish. spawn_locked does not wait for this joiner: it
    // archives the old file and swaps in a new LogBuffer, so a straggler can
    // only append into the archived inode.
    if (out_fd >= 0) {
        ::close(out_fd);
    }
    if (err_fd >= 0) {
        ::close(err_fd);
    }
    std::thread out = std::move(child->reader_out);
    std::thread err = std::move(child->reader_err);
    std::shared_ptr<LogBuffer> ended_run = child->log;
    std::thread([out = std::move(out), err = std::move(err),
                 ended_run = std::move(ended_run)]() mutable {
        if (out.joinable()) {
            out.join();
        }
        if (err.joinable()) {
            err.join();
        }
    }).detach();
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
                // whole group, models first (unchanged order)
                signal_process_group(child->pgid, child->pid, SIGINT);
            }
        }
    }
    for (const auto& id : model_ids) {
        if (!reap_group_with_grace(id)) {
            log_line(id, "shutdown: process group survived grace+2s");
        }
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
                // whole group, gateway last (unchanged order)
                signal_process_group(gateway->pgid, gateway->pid, SIGINT);
            }
        }
    }
    if (!reap_group_with_grace(kGatewayId)) {
        log_line(kGatewayId, "shutdown: process group survived grace+2s");
    }

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
            std::unique_lock<std::mutex> lock(_mu);
            for (Child* child : to_restart) {
                // Re-check under the spawn lock: the collection pass above ran
                // under an earlier lock hold, and a REST stop_server in between
                // clears wanted/backoff without touching this list. Spawning
                // anyway would revive a process the operator just stopped.
                // (_children is insert-only, so the collected pointers stay valid.)
                if (!child->wanted || child->stopping || _shutdown_requested.load()) {
                    continue;
                }
                child->backoff_due_ms = 0;
                std::string err;
                if (!spawn_locked(child, &err)) {
                    if (is_permanent_spawn_error(err)) {
                        child->wanted = false;
                        child->error = err;
                        child->engine.note_permanent_failure();
                        if (child->log != nullptr) {
                            child->log->append("[supervisor] " + err);
                        }
                    } else {
                        const auto decision =
                            child->engine.note_exit(monotonic_ms(), false, false);
                        apply_exit_decision(child, decision, false);
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
    constexpr int kShutdownWaitMs = 30000;
    if (_monitor.joinable()) {
        std::unique_lock<std::mutex> lock(_mu);
        if (!_cv.wait_for(lock, std::chrono::milliseconds(kShutdownWaitMs),
                          [this]() { return _shutdown_done; })) {
            // Bounded wait: continue teardown even if monitor did not signal.
        }
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

void ProcessSupervisor::autostart_all(int gateway_deadline_ms, int models_deadline_ms) {
    if (gateway_deadline_ms <= 0) {
        gateway_deadline_ms = 15000;
    }
    if (models_deadline_ms <= 0) {
        models_deadline_ms = 60000;
    }

    {
        std::lock_guard<std::mutex> lock(_mu);
        _autostart_gate = AutostartGate{};
    }

    // gateway first: routing must exist before clients can reach models.
    // Any gateway-phase failure blocks this autostart wave (zero model spawns).
    bool skip_models = false;
    {
        const std::string gateway_bin =
            (std::filesystem::path(_project_root) / _cfg.supervisor.bin_dir / "mortred-gateway.out")
                .string();
        if (!std::filesystem::exists(gateway_bin)) {
            const std::string reason = "gateway binary missing: " + gateway_bin;
            {
                std::lock_guard<std::mutex> lock(_mu);
                mark_gateway_autostart_blocked_locked(reason, /*permanent_fail=*/true);
            }
            std::fprintf(stderr, "mortred-supervisor: autostart blocked: %s\n", reason.c_str());
            skip_models = true;
        } else {
            std::string err;
            if (!start_server(kGatewayId, &err)) {
                const std::string reason =
                    err.empty() ? std::string("gateway start failed") : err;
                {
                    std::lock_guard<std::mutex> lock(_mu);
                    // permanent scrape/TRT failures already marked failed; still record gate
                    mark_gateway_autostart_blocked_locked(reason, /*permanent_fail=*/true);
                }
                std::fprintf(stderr, "mortred-supervisor: autostart gateway failed: %s\n",
                             reason.c_str());
                skip_models = true;
            } else {
                const int64_t deadline = monotonic_ms() + gateway_deadline_ms;
                while (monotonic_ms() < deadline && !_shutdown_requested.load()) {
                    const Status s = status(kGatewayId);
                    if (s.state == "running" || s.pid < 0) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                const Status s = status(kGatewayId);
                if (s.state != "running") {
                    std::ostringstream oss;
                    oss << "gateway not ready within deadline (" << gateway_deadline_ms
                        << " ms); state=" << s.state << " pid=" << s.pid;
                    if (!s.error.empty()) {
                        oss << " error=" << s.error;
                    }
                    const std::string reason = oss.str();
                    {
                        std::lock_guard<std::mutex> lock(_mu);
                        mark_gateway_autostart_blocked_locked(reason, /*permanent_fail=*/false);
                    }
                    std::fprintf(stderr,
                                 "mortred-supervisor: autostart gateway deadline (%d ms) "
                                 "elapsed (state=%s pid=%d error=%s)\n",
                                 gateway_deadline_ms, s.state.c_str(), s.pid, s.error.c_str());
                    skip_models = true;
                }
            }
        }
    }

    if (skip_models) {
        return;
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
    if (queue.empty()) {
        return;
    }
    _autostart_models_active.store(true);
    struct ClearModelsActive {
        std::atomic<bool>& flag;
        ~ClearModelsActive() {
            flag.store(false);
        }
    } clear_models_active{_autostart_models_active};

    const size_t width = static_cast<size_t>(std::max(1, _cfg.supervisor.start_concurrency));
    auto unresolved_count = [this, &queue](size_t launched) {
        size_t n = 0;
        for (size_t i = 0; i < launched && i < queue.size(); ++i) {
            const Status s = status(queue[i]);
            // failed is terminal for this wave (e.g. empty server_uri); do not
            // wait for models deadline.
            if (s.state == "failed") {
                continue;
            }
            if (s.pid >= 0 && s.state != "running") {
                ++n;
            }
        }
        return n;
    };
    const int64_t deadline = monotonic_ms() + models_deadline_ms;
    size_t launched = 0;
    while (launched < queue.size() || unresolved_count(launched) > 0) {
        if (_shutdown_requested.load()) {
            return;
        }
        if (monotonic_ms() >= deadline) {
            std::ostringstream oss;
            oss << "models not ready within deadline (" << models_deadline_ms << " ms):";
            std::fprintf(stderr,
                         "mortred-supervisor: autostart models deadline (%d ms) elapsed "
                         "with unresolved children:\n",
                         models_deadline_ms);
            for (size_t i = 0; i < launched && i < queue.size(); ++i) {
                const Status s = status(queue[i]);
                if (s.pid >= 0 && s.state != "running" && s.state != "failed") {
                    oss << " " << queue[i];
                    std::fprintf(stderr, "  - %s state=%s pid=%d error=%s\n", queue[i].c_str(),
                                 s.state.c_str(), s.pid, s.error.c_str());
                }
            }
            {
                // Same critical section as the gate write. Judge by lock state:
                // only a launched non-gateway child still starting becomes
                // failed. Leave running and not-yet-launched stopped children
                // alone. Do not signal; ordered shutdown still owns SIGINT,
                // kStopGraceMs, then SIGKILL.
                std::lock_guard<std::mutex> lock(_mu);
                for (size_t i = 0; i < launched && i < queue.size(); ++i) {
                    Child* child = find_locked(queue[i]);
                    if (child == nullptr || child->is_gateway || child->pid <= 0 ||
                        child->engine.state() != SupervisedState::kStarting) {
                        continue;
                    }
                    child->engine.note_permanent_failure();
                    child->wanted = false;
                    child->ready = false;
                    child->backoff_due_ms = 0;
                    const std::string sentence = "not ready within deadline: " + child->id;
                    if (child->error.empty()) {
                        child->error = sentence;
                    }
                    if (child->log != nullptr) {
                        child->log->append("[supervisor] " + sentence);
                    }
                }
                mark_models_failed_locked(oss.str());
            }
            return;
        }
        while (launched < queue.size() && unresolved_count(launched) < width) {
            const std::string& id = queue[launched++];
            std::string err;
            if (!start_server(id, &err)) {
                std::ostringstream oss;
                oss << "model start failed: " << id;
                if (!err.empty()) {
                    oss << ": " << err;
                }
                {
                    std::lock_guard<std::mutex> lock(_mu);
                    mark_models_failed_locked(oss.str());
                }
                std::fprintf(stderr, "mortred-supervisor: autostart %s failed: %s\n", id.c_str(),
                             err.c_str());
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

}  // namespace control
}  // namespace mortred
