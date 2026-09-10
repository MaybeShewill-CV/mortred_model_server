/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: occupancy_gate.h
* Date: 26-9-10
************************************************/

// Fail-closed spawn check: TensorRT ids in an active pack need a calibrated
// gpu_mem_mib stamp. Joint budget and GPU fingerprint apply when those pack
// fields are present. Does not link NVML; live free/name come from nvidia-smi
// or an injected snapshot. occupancy_policy=off / MORTRED_OCCUPANCY_ENFORCE=0
// skip the gate (unsafe; doctor --strict still fails).

#ifndef MORTRED_CONTROL_OCCUPANCY_GATE_H
#define MORTRED_CONTROL_OCCUPANCY_GATE_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

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
    if (const char* env = std::getenv("MORTRED_OCCUPANCY_ENFORCE"); env != nullptr && *env != '\0') {
        const std::string v = occupancy_detail::lower_copy(env);
        if (v == "0" || v == "false" || v == "off") {
            return true;
        }
    }
    return occupancy_detail::lower_copy(policy) == "off";
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

inline LiveGpuSnapshot query_nvidia_smi() {
    LiveGpuSnapshot snap;
    int fds[2];
    if (::pipe(fds) != 0) {
        return snap;
    }
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
        char arg0[] = "nvidia-smi";
        char arg1[] = "--query-gpu=name,memory.total,memory.free";
        char arg2[] = "--format=csv,noheader,nounits";
        char* argv[] = {arg0, arg1, arg2, nullptr};
        ::execvp(argv[0], argv);
        ::_exit(127);
    }
    ::close(fds[1]);
    std::string out;
    char buf[256];
    ssize_t n = 0;
    while ((n = ::read(fds[0], buf, sizeof(buf))) > 0) {
        out.append(buf, static_cast<size_t>(n));
        if (out.size() > 4096) {
            break;
        }
    }
    ::close(fds[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return snap;
    }
    const auto nl = out.find('\n');
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
    return true;
}

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_OCCUPANCY_GATE_H
