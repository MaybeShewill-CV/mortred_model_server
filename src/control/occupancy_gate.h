/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: occupancy_gate.h
* Date: 26-9-10
************************************************/

// Fail-closed spawn check: TensorRT ids in an active pack need a calibrated
// gpu_mem_mib stamp. Joint budget and GPU fingerprint apply when those pack
// fields are present. Round 7 P1-1: prefer shared_gpu_probe (NVML→smi).
// query_nvidia_smi remains the SmiGpuProbe backend; callers should use
// query_live_gpu_via_shared_probe() for a single metrics source.
// or an injected snapshot. occupancy_policy=off / MORTRED_OCCUPANCY_ENFORCE=0
// skip the gate (unsafe; doctor --strict still fails).

#ifndef MORTRED_CONTROL_OCCUPANCY_GATE_H
#define MORTRED_CONTROL_OCCUPANCY_GATE_H

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <poll.h>
#include <chrono>
#include <thread>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// libc environ. nvidia-smi children pass this pointer to execve; they do not
// build an environment block after fork.
extern "C" char** environ;

namespace mortred {
namespace control {

struct LiveGpuSnapshot {
    bool available = false;
    std::string name;
    int total_mib = 0;
    int free_mib = 0;
};

struct OccupancySibling {
    std::string id;
    int gpu_mem_mib = 0;
    int rss_mib = 0;
};

struct OccupancyCheckInput {
    bool pack_active = false;
    std::string pack_path;
    std::string occupancy_policy = "enforce";
    int gpu_reserve_pct = 15;
    std::string gpu_name;
    int gpu_memory_total_mib = 0;
    std::string candidate_id;
    bool candidate_is_tensorrt = false;
    int worker_nums = 0;
    bool has_gpu_mem_mib = false;
    int gpu_mem_mib = 0;
    int gpu_mem_at_workers = 0;
    // Optional operator-supplied process occupancy (MiB). When set, spawn fails
    // if |declared - measured| exceeds max(64, declared/100). Supervisor does
    // not sample this; production wiring is pack_occupancy.py --measured-json.
    bool has_measured_gpu_mem_mib = false;
    int measured_gpu_mem_mib = 0;
    // Host RAM dimension (Round 14).
    bool has_rss_mib = false;
    int rss_mib = 0;
    int host_memory_total_mib = 0;
    int ram_reserve_pct = 15;
    int live_mem_available_mib = -1;  // <0 = unknown
    // Explicit operator override: allow a TensorRT id to start even though the
    // live GPU probe is unavailable (nothing can be verified in that state).
    bool allow_unknown_gpu = false;
    std::vector<OccupancySibling> running_siblings;
    LiveGpuSnapshot live;
};

namespace occupancy_detail {

inline std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

}  // namespace occupancy_detail

inline bool occupancy_enforcement_disabled(const std::string& policy) {
    // P0-3: soft off alone must not skip the gate. Also require
    // MORTRED_OCCUPANCY_ALLOW_DISABLE=1|true|yes so production spawn cannot
    // silently overcommit when only occupancy_policy=off / ENFORCE=0 is set.
    bool soft_off = false;
    if (const char* env = std::getenv("MORTRED_OCCUPANCY_ENFORCE"); env != nullptr && *env != '\0') {
        const std::string v = occupancy_detail::lower_copy(env);
        if (v == "0" || v == "false" || v == "off") {
            soft_off = true;
        }
    }
    if (!soft_off) {
        soft_off = occupancy_detail::lower_copy(policy) == "off";
    }
    if (!soft_off) {
        return false;
    }
    if (const char* allow = std::getenv("MORTRED_OCCUPANCY_ALLOW_DISABLE");
        allow != nullptr && *allow != '\0') {
        const std::string v = occupancy_detail::lower_copy(allow);
        if (v == "1" || v == "true" || v == "yes") {
            return true;
        }
    }
    return false;
}

inline std::string occupancy_pack_arg(const std::string& pack_path) {
    return pack_path.empty() ? std::string("$MORTRED_PACK") : pack_path;
}

inline std::string occupancy_calibrate_hint(const std::string& pack_path) {
    return "stop the supervisor, then: mortredctl calibrate --pack " +
           occupancy_pack_arg(pack_path) + " --write-pack";
}

inline std::string occupancy_gate_error(const std::string& detail, const std::string& pack_path) {
    return "occupancy gate: " + detail + "; " + occupancy_calibrate_hint(pack_path);
}

inline bool is_occupancy_gate_error(const std::string& err) {
    return err.find("occupancy gate:") != std::string::npos;
}

// Declared stamp vs measured process occupancy. Same rule as
// scripts/pack_occupancy.py --check --measured-json.
inline int occupancy_declared_measured_tol_mib(int declared_mib) {
    if (declared_mib <= 0) {
        return 64;
    }
    return std::max(64, declared_mib / 100);
}

inline std::string occupancy_declared_measured_detail(const std::string& catalog_id,
                                                      int declared_mib, int measured_mib) {
    return "declared gpu_mem_mib=" + std::to_string(declared_mib) +
           " disagrees with measured " + std::to_string(measured_mib) + " for " + catalog_id;
}

// Parent-side PATH search shared by both nvidia-smi probes. Order matches
// this machine's glibc execvp: unset PATH uses confstr(_CS_PATH) (the same
// directories as glibc's CS_PATH), an empty PATH field is the current
// directory, and the first access(X_OK) success wins. A relative hit is
// turned into an absolute path here. The caller owns *out. No shared buffer,
// so the sampler thread and the spawn thread can resolve at the same time.
// Not found: false, and the caller must not fork.
inline bool resolve_nvidia_smi_path(std::string* out) {
    if (out == nullptr) {
        return false;
    }
    out->clear();
    constexpr const char* kFile = "nvidia-smi";

    std::string path;
    const char* env_path = std::getenv("PATH");
    if (env_path == nullptr) {
        const std::size_t n = ::confstr(_CS_PATH, nullptr, 0);
        if (n == 0) {
            return false;
        }
        path.assign(n, '\0');
        if (::confstr(_CS_PATH, path.data(), path.size()) == 0) {
            return false;
        }
        const std::size_t nul = path.find('\0');
        if (nul != std::string::npos) {
            path.resize(nul);
        }
    } else {
        path.assign(env_path);
    }

    std::string cwd;
    const auto ensure_cwd = [&cwd]() -> bool {
        if (!cwd.empty()) {
            return true;
        }
        std::string buf(256, '\0');
        for (;;) {
            if (::getcwd(buf.data(), buf.size()) != nullptr) {
                const std::size_t nul = buf.find('\0');
                if (nul != std::string::npos) {
                    buf.resize(nul);
                }
                if (buf.empty() || buf.front() != '/') {
                    return false;
                }
                cwd = std::move(buf);
                return true;
            }
            if (errno != ERANGE || buf.size() >= (static_cast<std::size_t>(1) << 20)) {
                return false;
            }
            buf.resize(buf.size() * 2, '\0');
        }
    };

    std::size_t begin = 0;
    for (;;) {
        const std::size_t colon = path.find(':', begin);
        const std::size_t end = colon == std::string::npos ? path.size() : colon;
        const std::string dir = path.substr(begin, end - begin);
        std::string candidate;
        if (dir.empty()) {
            // glibc execvp overwrites the joining slash, leaving the bare name.
            candidate = kFile;
        } else {
            candidate.reserve(dir.size() + 1 + std::strlen(kFile));
            candidate = dir;
            candidate.push_back('/');
            candidate += kFile;
        }
        if (::access(candidate.c_str(), X_OK) == 0) {
            if (candidate.front() != '/') {
                if (!ensure_cwd()) {
                    return false;
                }
                std::string absolute;
                absolute.reserve(cwd.size() + 1 + candidate.size());
                absolute = cwd;
                if (absolute.back() != '/') {
                    absolute.push_back('/');
                }
                absolute += candidate;
                candidate = std::move(absolute);
            }
            if (candidate.empty() || candidate.front() != '/') {
                return false;
            }
            *out = std::move(candidate);
            return true;
        }
        if (colon == std::string::npos) {
            break;
        }
        begin = colon + 1;
    }
    return false;
}

// Strict parse of the first nvidia-smi CSV line ("name, total MiB, free
// MiB"). Returns the snapshot with available=true only when every field is
// present and sane; anything else keeps available=false.
inline LiveGpuSnapshot parse_smi_snapshot(const std::string& out, LiveGpuSnapshot snap) {
    const auto nl = out.find(static_cast<char>(10));
    std::string line = occupancy_detail::trim_copy(nl == std::string::npos ? out : out.substr(0, nl));
    const auto last = line.rfind(',');
    if (last == std::string::npos) {
        return snap;
    }
    const std::string free_s = occupancy_detail::trim_copy(line.substr(last + 1));
    line = line.substr(0, last);
    const auto mid = line.rfind(',');
    if (mid == std::string::npos) {
        return snap;
    }
    const std::string total_s = occupancy_detail::trim_copy(line.substr(mid + 1));
    const std::string name = occupancy_detail::trim_copy(line.substr(0, mid));
    try {
        snap.total_mib = std::stoi(total_s);
        snap.free_mib = std::stoi(free_s);
    } catch (...) {
        return snap;
    }
    if (name.empty() || snap.total_mib <= 0) {
        return snap;
    }
    snap.name = name;
    snap.available = true;
    return snap;
}
// Probe nvidia-smi with a hard deadline. A hung driver/tool must not block
// supervisor spawn forever; on timeout/failure returns available=false so
// fingerprint/free-VRAM checks are skipped (stamp budget still applies).
inline LiveGpuSnapshot query_nvidia_smi(int timeout_ms = 3000) {
    LiveGpuSnapshot snap;
    if (timeout_ms <= 0) {
        timeout_ms = 3000;
    }
    std::string smi_path;
    if (!resolve_nvidia_smi_path(&smi_path)) {
        return snap;
    }
    int fds[2];
    if (::pipe(fds) != 0) {
        return snap;
    }
    char* smi_file = smi_path.data();
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(fds[0]);
        ::close(fds[1]);
        return snap;
    }
    if (pid == 0) {
        ::close(fds[0]);
        if (::dup2(fds[1], STDOUT_FILENO) < 0) {
            ::_exit(127);
        }
        ::close(fds[1]);
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDERR_FILENO);
            ::close(devnull);
        }
        char arg1[] = "--query-gpu=name,memory.total,memory.free";
        char arg2[] = "--format=csv,noheader,nounits";
        char* argv[] = {smi_file, arg1, arg2, nullptr};
        ::execve(smi_file, argv, ::environ);
        ::_exit(127);
    }
    ::close(fds[1]);
    const int flags = ::fcntl(fds[0], F_GETFL, 0);
    if (flags >= 0) {
        ::fcntl(fds[0], F_SETFL, flags | O_NONBLOCK);
    }
    std::string out;
    char buf[256];
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    bool timed_out = false;
    bool eof = false;
    while (!eof && out.size() <= 4096) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            timed_out = true;
            break;
        }
        const auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        pollfd pfd{};
        pfd.fd = fds[0];
        pfd.events = POLLIN;
        const int pr = ::poll(&pfd, 1, static_cast<int>(remain.count()));
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
            timed_out = true;
            break;
        }
        if (pr == 0) {
            timed_out = true;
            break;
        }
        const ssize_t n = ::read(fds[0], buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, static_cast<size_t>(n));
            continue;
        }
        if (n == 0) {
            eof = true;
            break;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            continue;
        }
        timed_out = true;
        break;
    }
    ::close(fds[0]);
    int status = 0;
    auto reap = [&]() -> bool {
        for (int i = 0; i < 50; ++i) {
            const pid_t r = ::waitpid(pid, &status, WNOHANG);
            if (r == pid) {
                return true;
            }
            if (r < 0 && errno != EINTR) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return false;
    };
    // Signal only a child that provably still belongs to this process and is
    // still RUNNING: waitid(WNOWAIT) peeks without consuming the exit state,
    // so a stolen (already reaped, possibly recycled) pid is never signalled
    // and an exited child needs no signal at all.
    const auto signal_if_running = [&pid]() {
        siginfo_t live{};
        if (::waitid(P_PID, static_cast<id_t>(pid), &live, WEXITED | WNOHANG | WNOWAIT) != 0) {
            return;  // ECHILD: reaped by someone else, never signal again
        }
        if (live.si_pid != 0) {
            return;  // exited but not yet reaped: nothing to signal
        }
        ::kill(pid, SIGKILL);
    };
    // Always reap; kill if the probe hung, hit the read cap with the child
    // still alive, or waitpid would otherwise block past the deadline.
    if (timed_out || !eof) {
        signal_if_running();
    }
    if (!reap()) {
        // The child was reaped out from under us by a process-wide reaper.
        // Its exit status is unknowable, but a fully read (EOF) output is
        // still trustworthy evidence: hand it to the strict parser instead
        // of failing open to the no-nvidia-smi path.
        if (timed_out || !eof) {
            return snap;
        }
        return parse_smi_snapshot(out, snap);
    }
    if (timed_out || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return snap;
    }
    return parse_smi_snapshot(out, snap);
}

inline int query_mem_available_mib() {
    FILE* f = ::fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return -1;
    }
    char line[256];
    long available_kb = -1;
    long free_kb = -1;
    while (::fgets(line, sizeof(line), f) != nullptr) {
        long val = 0;
        if (std::sscanf(line, "MemAvailable: %ld kB", &val) == 1) {
            available_kb = val;
        } else if (std::sscanf(line, "MemFree: %ld kB", &val) == 1) {
            free_kb = val;
        }
    }
    ::fclose(f);
    const long kb = available_kb >= 0 ? available_kb : free_kb;
    if (kb < 0) {
        return -1;
    }
    return static_cast<int>(kb / 1024);
}

inline bool occupancy_ready_for_spawn(const OccupancyCheckInput& in, std::string* err) {
    if (!in.pack_active || occupancy_enforcement_disabled(in.occupancy_policy)) {
        return true;
    }
    const auto fail = [&](const std::string& detail) {
        if (err != nullptr) {
            *err = occupancy_gate_error(detail, in.pack_path);
        }
        return false;
    };
    if (in.candidate_is_tensorrt && (!in.has_gpu_mem_mib || in.gpu_mem_mib <= 0)) {
        return fail("stamp missing for " + in.candidate_id + " (gpu_mem_mib>0 required for TensorRT)");
    }
    // Live telemetry is what turns the static stamps into a real capacity gate:
    // it supplies the GPU fingerprint and the free-memory budget. When the probe
    // is unavailable both checks below silently disappear, so a TensorRT id used
    // to start "verified" by a gate that verified nothing. Fail closed instead;
    // an operator who really wants to start unverified must say so explicitly.
    if (in.candidate_is_tensorrt && in.has_gpu_mem_mib && in.gpu_mem_mib > 0 &&
        !in.live.available && !in.allow_unknown_gpu) {
        return fail("GPU telemetry unavailable: cannot verify the " + in.candidate_id +
                    " stamp (gpu_mem_mib=" + std::to_string(in.gpu_mem_mib) +
                    " MiB); fix the GPU probe or set MORTRED_ALLOW_UNKNOWN_GPU=1 to start unverified");
    }
    if (in.has_measured_gpu_mem_mib && in.has_gpu_mem_mib && in.gpu_mem_mib > 0) {
        const int tol = occupancy_declared_measured_tol_mib(in.gpu_mem_mib);
        const int delta = in.gpu_mem_mib > in.measured_gpu_mem_mib
                              ? in.gpu_mem_mib - in.measured_gpu_mem_mib
                              : in.measured_gpu_mem_mib - in.gpu_mem_mib;
        if (delta > tol) {
            return fail(occupancy_declared_measured_detail(in.candidate_id, in.gpu_mem_mib,
                                                           in.measured_gpu_mem_mib));
        }
    }
    if (in.gpu_mem_at_workers > 0) {
        const int effective_w = in.worker_nums > 0 ? in.worker_nums : 1;
        if (effective_w != in.gpu_mem_at_workers) {
            return fail("stamp stale for " + in.candidate_id + " (worker_nums=" +
                        std::to_string(effective_w) + " but gpu_mem_at_workers=" +
                        std::to_string(in.gpu_mem_at_workers) + ")");
        }
    }
    if (in.live.available) {
        if (!in.gpu_name.empty() && in.live.name != in.gpu_name) {
            return fail("GPU fingerprint mismatch (pack gpu_name='" + in.gpu_name +
                        "', this card is '" + in.live.name + "')");
        }
        if (in.gpu_memory_total_mib > 0 && in.live.total_mib > 0) {
            const int tol = std::max(64, in.gpu_memory_total_mib / 100);
            const int delta = in.gpu_memory_total_mib > in.live.total_mib
                                  ? in.gpu_memory_total_mib - in.live.total_mib
                                  : in.live.total_mib - in.gpu_memory_total_mib;
            if (delta > tol) {
                return fail("GPU fingerprint mismatch (pack gpu_memory_total_mib=" +
                            std::to_string(in.gpu_memory_total_mib) + ", this card is " +
                            std::to_string(in.live.total_mib) + ")");
            }
        }
    }
    if (in.has_gpu_mem_mib && in.gpu_mem_mib > 0 && in.gpu_memory_total_mib > 0) {
        long long used = in.gpu_mem_mib;
        for (const auto& sib : in.running_siblings) {
            if (sib.gpu_mem_mib > 0) {
                used += sib.gpu_mem_mib;
            }
        }
        int reserve = in.gpu_reserve_pct;
        if (reserve < 0) {
            reserve = 0;
        }
        if (reserve > 90) {
            reserve = 90;
        }
        const long long cap =
            static_cast<long long>(in.gpu_memory_total_mib) * (100 - reserve) / 100;
        if (used > cap) {
            return fail("pack occupancy exceeds GPU budget for " + in.candidate_id + " (need " +
                        std::to_string(used) + " MiB, budget " + std::to_string(cap) +
                        " MiB); remove other pack ids or lower worker_nums");
        }
    }
    if (in.live.available && in.has_gpu_mem_mib && in.gpu_mem_mib > 0 &&
        in.live.free_mib < in.gpu_mem_mib) {
        if (err != nullptr) {
            *err = "occupancy gate: GPU free memory below " + in.candidate_id + " stamp (need " +
                   std::to_string(in.gpu_mem_mib) + " MiB free); stop other GPU processes, then retry start";
        }
        return false;
    }
    if (in.host_memory_total_mib > 0 && in.has_rss_mib && in.rss_mib > 0) {
        long long used = in.rss_mib;
        for (const auto& sib : in.running_siblings) {
            if (sib.rss_mib > 0) {
                used += sib.rss_mib;
            }
        }
        int reserve = in.ram_reserve_pct;
        if (reserve < 0) { reserve = 0; }
        if (reserve > 90) { reserve = 90; }
        const long long cap =
            static_cast<long long>(in.host_memory_total_mib) * (100 - reserve) / 100;
        if (used > cap) {
            return fail("pack host RAM occupancy exceeds budget for " + in.candidate_id +
                        " (need " + std::to_string(used) + " MiB RSS, budget " +
                        std::to_string(cap) + " MiB); remove other pack ids or lower handler_threads");
        }
    }
    if (in.live_mem_available_mib >= 0 && in.has_rss_mib && in.rss_mib > 0 &&
        in.live_mem_available_mib < in.rss_mib) {
        if (err != nullptr) {
            *err = "occupancy gate: host MemAvailable below " + in.candidate_id +
                   " rss_mib stamp (need " + std::to_string(in.rss_mib) +
                   " MiB); free RAM or reduce concurrent ids";
        }
        return false;
    }
    return true;
}

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_OCCUPANCY_GATE_H
