/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: gpu_sampler.h
* Date: 26-9-16
************************************************/

// Background GPU sampler for the supervisor HUD.
// Round 7 P1-1: shared_gpu_probe() is the primary metrics source
// (NVML dlopen → smi chain). Local smi fork is fallback-only; never waitpid(-1).
// dedicated thread (never on the HTTP handler path) at a configurable
// interval, stores a bounded ring buffer of full-field samples, and serves
// them to /api/v1/gpu as JSON. The probe's fork/poll/deadline discipline
// follows occupancy_gate.h's query_nvidia_smi (proven against hung drivers).
//
// Failure policy: first probe failure logs once and retries with backoff
// (min(interval, 30s)); available=false in the payload lets the UI show
// a graceful "gpu n/a" instead of a dead panel.

#ifndef MORTRED_CONTROL_GPU_SAMPLER_H
#define MORTRED_CONTROL_GPU_SAMPLER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include "control/occupancy_gate.h"
#include "control/gpu_probe.h"

namespace mortred {
namespace control {

struct GpuSample {
    int64_t t_unix_ms = 0;
    int32_t util = -1;            // %, -1 unknown
    int32_t mem_used_mib = -1;
    int32_t mem_total_mib = -1;
    int32_t temp_c = -1;          // WSL often [N/A]
    double power_w = -1.0;
    int32_t clocks_sm_mhz = -1;
    int32_t clocks_mem_mhz = -1;
    int32_t fan_pct = -1;
    int32_t pcie_gen = -1;
    int32_t pcie_width = -1;
};

inline int32_t parse_i32_or(const std::string& s, int32_t fallback) {
    if (s.empty() || s == "[N/A]" || s == "N/A") return fallback;
    try { return static_cast<int32_t>(std::stod(s)); } catch (...) { return fallback; }
}

inline double parse_f64_or(const std::string& s, double fallback) {
    if (s.empty() || s == "[N/A]" || s == "N/A") return fallback;
    try { return std::stod(s); } catch (...) { return fallback; }
}

class GpuSampler {
  public:
    GpuSampler(int interval_ms = 2000, size_t ring_capacity = 300)
        : interval_ms_(interval_ms), capacity_(ring_capacity) {}

    ~GpuSampler() { stop(); }

    void start() {
        if (running_.exchange(true)) return;
        thread_ = std::thread([this]() { loop(); });
    }
    void stop() {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
    }

    bool available() const {
        std::lock_guard<std::mutex> lock(mu_);
        return available_;
    }
    std::string gpu_name() const {
        std::lock_guard<std::mutex> lock(mu_);
        return gpu_name_;
    }
    int interval_ms() const { return interval_ms_; }

    std::vector<GpuSample> snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        return {ring_.begin(), ring_.end()};
    }

  private:
    static int64_t unix_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    // Probe nvidia-smi with a hard deadline (poll-based, non-blocking read).
    // Returns the raw CSV stdout line, or "" on failure/timeout.
    static std::string probe_nvidia_smi(int timeout_ms, std::string* gpu_name_out) {
        std::string smi_path;
        if (!resolve_nvidia_smi_path(&smi_path)) return "";
        char* smi_file = smi_path.data();
        int fds[2];
        if (::pipe(fds) != 0) return "";
        const pid_t pid = ::fork();
        if (pid < 0) { ::close(fds[0]); ::close(fds[1]); return ""; }
        if (pid == 0) {
            ::close(fds[0]);
            if (::dup2(fds[1], STDOUT_FILENO) < 0) ::_exit(127);
            ::close(fds[1]);
            const int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0) { ::dup2(devnull, STDERR_FILENO); ::close(devnull); }
            char a1[] = "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                        "temperature.gpu,power.draw,clocks.sm,clocks.mem,fan.speed,"
                        "pcie.link.gen.current,pcie.link.width.current";
            char a2[] = "--format=csv,noheader,nounits";
            char* argv[] = {smi_file, a1, a2, nullptr};
            ::execve(smi_file, argv, ::environ);
            ::_exit(127);
        }
        ::close(fds[1]);
        const int flags = ::fcntl(fds[0], F_GETFL, 0);
        if (flags >= 0) ::fcntl(fds[0], F_SETFL, flags | O_NONBLOCK);
        std::string out;
        char buf[512];
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        bool timed_out = false;
        bool eof = false;
        while (!eof && out.size() <= 8192) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) { timed_out = true; break; }
            pollfd pfd{};
            pfd.fd = fds[0]; pfd.events = POLLIN;
            const auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    deadline - now).count();
            const int pr = ::poll(&pfd, 1, static_cast<int>(remain));
            if (pr < 0) { if (errno == EINTR) continue; timed_out = true; break; }
            if (pr == 0) { timed_out = true; break; }
            const ssize_t n = ::read(fds[0], buf, sizeof(buf));
            if (n > 0) { out.append(buf, static_cast<size_t>(n)); continue; }
            if (n == 0) { eof = true; break; }
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
            timed_out = true;
            break;
        }
        ::close(fds[0]);
        int status = 0;
        auto reap = [&]() -> bool {
            for (int i = 0; i < 50; ++i) {
                const pid_t r = ::waitpid(pid, &status, WNOHANG);
                if (r == pid) return true;
                if (r < 0 && errno != EINTR) return false;  // ECHILD: stolen
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            return false;
        };
        // Signal only a child that still belongs to this process and is RUNNING
        // (waitid WNOWAIT peek). Never kill a stolen/recycled pid.
        const auto signal_if_running = [&pid]() {
            siginfo_t live{};
            if (::waitid(P_PID, static_cast<id_t>(pid), &live, WEXITED | WNOHANG | WNOWAIT) != 0) {
                return;  // ECHILD
            }
            if (live.si_pid != 0) {
                return;  // exited, unreaped
            }
            ::kill(pid, SIGKILL);
        };
        if (timed_out || !eof) {
            signal_if_running();
        }
        if (!reap()) {
            // Stolen by process-wide reaper: trust fully-read EOF output only;
            // otherwise fail closed (empty -> sampler offline path).
            if (timed_out || !eof || out.empty() || out.size() > 8192) {
                return "";
            }
        } else if (timed_out || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            return "";
        }
        if (out.size() > 8192 || out.empty()) return "";
        if (gpu_name_out != nullptr) {
            const auto comma = out.find(',');
            if (comma != std::string::npos) {
                *gpu_name_out = out.substr(0, comma);
                auto b = gpu_name_out->find_first_not_of(" \t");
                auto e = gpu_name_out->find_last_not_of(" \t\r\n");
                if (b != std::string::npos) *gpu_name_out = gpu_name_out->substr(b, e - b + 1);
            }
        }
        return out;
    }

    static GpuSample parse_sample(const std::string& csv, int name_len,
                                   std::string* util_col = nullptr) {
        (void)name_len;  // fields[0] (the name) is skipped unconditionally below
        GpuSample s;
        s.t_unix_ms = unix_ms();
        std::vector<std::string> fields;
        size_t start = 0;
        for (size_t i = 0; i <= csv.size(); ++i) {
            if (i == csv.size() || csv[i] == ',') {
                auto f = csv.substr(start, i - start);
                auto b = f.find_first_not_of(" \t");
                auto e = f.find_last_not_of(" \t\r\n");
                fields.push_back(b != std::string::npos ? f.substr(b, e - b + 1) : "");
                start = i + 1;
            }
        }
        // fields[0] = name (skipped), then the 10 queried metrics
        if (fields.size() >= 11) {
            if (util_col != nullptr) *util_col = fields[1];
            s.util = parse_i32_or(fields[1], -1);
            s.mem_used_mib = parse_i32_or(fields[2], -1);
            s.mem_total_mib = parse_i32_or(fields[3], -1);
            s.temp_c = parse_i32_or(fields[4], -1);
            s.power_w = parse_f64_or(fields[5], -1.0);
            s.clocks_sm_mhz = parse_i32_or(fields[6], -1);
            s.clocks_mem_mhz = parse_i32_or(fields[7], -1);
            s.fan_pct = parse_i32_or(fields[8], -1);
            s.pcie_gen = parse_i32_or(fields[9], -1);
            s.pcie_width = parse_i32_or(fields[10], -1);
        }
        return s;
    }

    void loop() {
        bool first = true;
        int backoff_ms = interval_ms_;
        while (running_.load()) {
            // Prefer shared GpuProbe (NVML→smi). Avoid a second smi fork on the
            // happy path; probe_nvidia_smi remains as a local fallback only if
            // the shared probe returns unavailable.
            const GpuProbeSample probed = shared_gpu_probe().sample(3000);
            bool got = false;
            GpuSample sample;
            std::string name;
            bool util_na = false;
            if (probed.available) {
                sample.t_unix_ms = unix_ms();
                sample.util = probed.util;
                sample.mem_used_mib = probed.mem_used_mib;
                sample.mem_total_mib = probed.mem_total_mib;
                sample.temp_c = probed.temp_c;
                sample.power_w = probed.power_w;
                name = probed.name;
                util_na = probed.util < 0;
                got = true;
            } else {
                const std::string csv = probe_nvidia_smi(3000, &name);
                if (!csv.empty()) {
                    std::string util_col;
                    sample = parse_sample(csv, 0, &util_col);
                    util_na = util_col == "N/A" || util_col == "[N/A]";
                    got = true;
                }
            }
            if (got) {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    available_ = !util_na;
                    if (!name.empty()) gpu_name_ = name;
                    ring_.push_back(sample);
                    while (ring_.size() > capacity_) ring_.pop_front();
                }
                backoff_ms = interval_ms_;
                if (first) {
                    if (util_na) {
                        std::fprintf(stderr,
                                     "mortred-supervisor: gpu sampler offline (utilization N/A)\n");
                    } else {
                        std::fprintf(stderr, "mortred-supervisor: gpu sampler online (%s)\n",
                                     name.c_str());
                    }
                    first = false;
                }
            } else {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    available_ = false;
                }
                if (first) {
                    std::fprintf(stderr, "mortred-supervisor: gpu sampler offline (no probe / no GPU)\n");
                    first = false;
                }
                backoff_ms = std::min(backoff_ms * 2, 30000);
            }
            // sleep in small chunks so stop() is responsive
            int slept = 0;
            while (running_.load() && slept < backoff_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    std::min(200, backoff_ms - slept)));
                slept += 200;
            }
        }
    }

    int interval_ms_;
    size_t capacity_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    mutable std::mutex mu_;
    std::deque<GpuSample> ring_;
    bool available_ = false;
    std::string gpu_name_;
};

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_GPU_SAMPLER_H
