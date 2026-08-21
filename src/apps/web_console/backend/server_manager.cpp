/************************************************
 * Author: Codex
 * File: server_manager.cpp
 ************************************************/

#include "server_manager.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <utility>
#include <dirent.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "ready_probe.h"

namespace mortred_web {

namespace {

void read_pipe_loop(int fd, LogBuffer* buffer) {
    char buf[4096];
    std::string line;
    while (true) {
        ssize_t n = read(fd, buf, sizeof(buf));
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

void close_all_fds() {
    // close every fd except stdin/stdout/stderr so the exec'd model server
    // does not inherit the app_server's listening socket / epoll fds
    DIR* d = opendir("/proc/self/fd");
    if (d == nullptr) {
        return;
    }
    int dir_fd = dirfd(d);
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        char* end = nullptr;
        long fd = std::strtol(e->d_name, &end, 10);
        if (*end == '\0' && fd > 2 && fd != dir_fd) {
            ::close(static_cast<int>(fd));
        }
    }
    closedir(d);
}

} // namespace

void ServerManager::init(const Catalog& catalog, const std::string& project_root, const std::string& logs_dir) {
    _catalog = &catalog;
    // 目录名可配置：安装树部署（Docker/systemd）通过 APP_BIN_DIR / APP_LIB_DIR /
    // APP_LIBS_DIR 注入 bin / lib / lib；源码树直跑保持默认 _bin / _lib /
    // 3rd_party/libs（兼容既有布局与测试）。修复：CMake 安装树是 bin/lib，
    // 原硬编码 _bin/_lib/3rd_party/libs 使部署环境下 spawn 模型服务必然 execl 127。
    const char* bin_dir  = getenv("APP_BIN_DIR");
    const char* lib_dir  = getenv("APP_LIB_DIR");
    const char* libs_dir = getenv("APP_LIBS_DIR");
    _bin_dir  = project_root + "/" + (bin_dir  && *bin_dir  ? bin_dir  : "_bin");
    _lib_dir  = project_root + "/" + (lib_dir  && *lib_dir  ? lib_dir  : "_lib");
    _libs_dir = project_root + "/" + (libs_dir && *libs_dir ? libs_dir : "3rd_party/libs");
    _logs_dir = logs_dir;
    std::error_code ec;
    std::filesystem::create_directories(_logs_dir, ec);
}

LogBuffer* ServerManager::get_or_create_log(const std::string& id) {
    std::lock_guard<std::mutex> lock(_mu);
    auto it = _logs.find(id);
    if (it != _logs.end()) {
        return it->second.get();
    }
    auto buf = std::make_unique<LogBuffer>(_logs_dir + "/" + id + ".log");
    auto* raw = buf.get();
    _logs[id] = std::move(buf);
    return raw;
}

void ServerManager::spawn_waiters(const std::string& id, ProcInfo* info) {
    auto* buffer = get_or_create_log(id);
    int out_fd = info->out_fd;
    int err_fd = info->err_fd;
    pid_t pid = info->pid;

    info->reader_out = std::thread([out_fd, buffer]() { read_pipe_loop(out_fd, buffer); });
    info->reader_err = std::thread([err_fd, buffer]() { read_pipe_loop(err_fd, buffer); });

    info->waiter = std::thread([this, id, pid, out_fd, err_fd, info, buffer]() {
        int status = 0;
        waitpid(pid, &status, 0);
        buffer->append("[app] 进程已退出 (pid " + std::to_string(pid) +
                       ", status " + std::to_string(status) + ")");
        ::close(out_fd);
        ::close(err_fd);
        if (info->reader_out.joinable()) {
            info->reader_out.join();
        }
        if (info->reader_err.joinable()) {
            info->reader_err.join();
        }
        std::lock_guard<std::mutex> lock(_mu);
        info->running = false;
        _cv.notify_all();
    });
}

bool ServerManager::start(const ServerEntry& entry, std::string& err) {
    {
        std::lock_guard<std::mutex> lock(_mu);
        auto it = _procs.find(entry.id);
        if (it != _procs.end() && it->second->running.load()) {
            err = "server already running";
            return false;
        }
        // port conflict with another managed server
        for (const auto& [id, port] : _ports) {
            auto pit = _procs.find(id);
            if (id != entry.id && pit != _procs.end() && pit->second->running.load() && port == entry.port) {
                err = "port " + std::to_string(entry.port) + " already used by " + id;
                return false;
            }
        }
    }

    int out_pipe[2];
    int err_pipe[2];
    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        err = "create pipe failed";
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        err = "fork failed";
        return false;
    }
    if (pid == 0) {
        // child
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);
        close_all_fds();
        // When app_server is launched in the background (e.g. via start.sh with
        // nohup ... &), bash sets SIGINT/SIGQUIT to SIG_IGN in the child and the
        // disposition survives fork+exec. Reset them to the default so model
        // servers actually terminate on SIGINT (our stop signal) instead of
        // requiring a SIGKILL fallback.
        ::signal(SIGINT, SIG_DFL);
        ::signal(SIGQUIT, SIG_DFL);
        std::string ld = _lib_dir + ":" + _libs_dir;
        setenv("LD_LIBRARY_PATH", ld.c_str(), 1);
        if (chdir(_bin_dir.c_str()) != 0) {
            _exit(127);
        }
        execl((_bin_dir + "/" + entry.exe).c_str(), entry.exe.c_str(), entry.config.c_str(), (char*)nullptr);
        _exit(127);
    }

    // parent
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    auto info = std::make_unique<ProcInfo>();
    info->pid = pid;
    info->out_fd = out_pipe[0];
    info->err_fd = err_pipe[0];
    info->running.store(true);
    {
        std::lock_guard<std::mutex> lock(_mu);
        _ports[entry.id] = entry.port;
        _procs[entry.id] = std::move(info);
    }
    get_or_create_log(entry.id)->reset();
    get_or_create_log(entry.id)->append("[app] 启动请求已发送 (pid " + std::to_string(pid) + ")");
    spawn_waiters(entry.id, _procs[entry.id].get());

    // 快速失败检测（消除"假启动"）：子进程 spawn 后立即退出（如 execl 失败
    // _exit(127)、依赖库缺失）时，waiter 线程会很快置 running=false 并通知 _cv。
    // 短超时探测：健康进程（正在加载模型）在窗口内 running 仍为 true，继续视为已启动；
    // 已退出则返回失败，退出原因可从该服务日志缓冲查看。
    // 注意：此处不 erase 死条目——waiter/reader 线程可能仍在收尾，销毁 joinable
    // std::thread 会 terminate；死条目 running=false，is_running/重复 start/
    // 端口冲突检查都会正确跳过。
    constexpr int k_spawn_probe_ms = 200;
    {
        std::unique_lock<std::mutex> lock(_mu);
        _cv.wait_for(lock, std::chrono::milliseconds(k_spawn_probe_ms), [this, &entry]() {
            auto it = _procs.find(entry.id);
            return it == _procs.end() || !it->second->running.load();
        });
        auto it = _procs.find(entry.id);
        if (it != _procs.end() && !it->second->running.load()) {
            err = "model server exited immediately after start (see logs: " + entry.id + ")";
            return false;
        }
    }
    return true;
}

bool ServerManager::stop(const std::string& id, std::string& err) {
    pid_t pid = -1;
    {
        std::lock_guard<std::mutex> lock(_mu);
        auto it = _procs.find(id);
        if (it == _procs.end() || !it->second->running.load()) {
            err = "server not running";
            return false;
        }
        pid = it->second->pid;
    }
    // model servers install glog's InstallFailureSignalHandler(), which dumps a
    // scary "SIGTERM received ... stack trace" to stderr on SIGTERM. glog does
    // not handle SIGINT, so sending SIGINT terminates the process silently and
    // behaves identically for our workflow-based servers.
    ::kill(pid, SIGINT);

    // wait up to 5s for graceful exit
    {
        std::unique_lock<std::mutex> lock(_mu);
        auto pit = _procs.find(id);
        if (pit != _procs.end() && pit->second->running.load()) {
            _cv.wait_for(lock, std::chrono::seconds(5), [&]() {
                auto it = _procs.find(id);
                return it == _procs.end() || !it->second->running.load();
            });
        }
    }
    {
        std::lock_guard<std::mutex> lock(_mu);
        auto pit = _procs.find(id);
        if (pit != _procs.end() && pit->second->running.load()) {
            ::kill(pid, SIGKILL);
        }
    }
    return true;
}

bool ServerManager::is_running(const std::string& id) const {
    std::lock_guard<std::mutex> lock(_mu);
    auto it = _procs.find(id);
    return it != _procs.end() && it->second->running.load();
}

bool ServerManager::is_ready(const std::string& id) {
    int port = -1;
    {
        std::lock_guard<std::mutex> lock(_mu);
        auto it = _procs.find(id);
        if (it == _procs.end() || !it->second->running.load()) {
            return false;
        }
        auto port_it = _ports.find(id);
        if (port_it == _ports.end()) {
            return false;
        }
        port = port_it->second;
    }
    // 探测真实 /ready 端点（2xx 即就绪），取代日志字符串匹配：
    // 日志文案一旦调整 grep 即失效，而 /ready 是受 e2e 契约测试保护的稳定接口
    return endpoint_ready(port, "/ready", 1000);
}

ServerManager::Status ServerManager::status(const std::string& id) const {
    Status s;
    std::lock_guard<std::mutex> lock(_mu);
    auto it = _procs.find(id);
    if (it != _procs.end()) {
        s.running = it->second->running.load();
        s.pid = static_cast<int>(it->second->pid);
    }
    return s;
}

LogBuffer* ServerManager::logs(const std::string& id) {
    std::lock_guard<std::mutex> lock(_mu);
    auto it = _logs.find(id);
    if (it != _logs.end()) {
        return it->second.get();
    }
    return nullptr;
}

std::vector<std::string> ServerManager::running_ids() const {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lock(_mu);
    for (const auto& [id, p] : _procs) {
        if (p && p->running.load()) {
            out.push_back(id);
        }
    }
    return out;
}

} // namespace mortred_web
