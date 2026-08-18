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
    _bin_dir = project_root + "/_bin";
    _lib_dir = project_root + "/_lib";
    _libs_dir = project_root + "/3rd_party/libs";
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
    if (!is_running(id)) {
        return false;
    }
    auto* buf = logs(id);
    if (buf == nullptr) {
        return false;
    }
    auto lines = buf->slice(0, 2000);
    for (const auto& line : lines) {
        if (line.find("server init successfully") != std::string::npos ||
            line.find("initialization complete") != std::string::npos) {
            return true;
        }
    }
    return false;
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
