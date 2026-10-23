/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor_unittest.cc
* Date: 26-8-22
************************************************/

// End-to-end supervision tests against the real ProcessSupervisor core and a
// fake model server (fork/exec, readiness probes, restart backoff, crash-loop
// give-up, ordered shutdown). Runs in the tests-only CI path (no workflow).

#include <gtest/gtest.h>

#include <dirent.h>
#include <unistd.h>
#include <signal.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <thread>

#include "control/catalog.h"
#include "control/control_config.h"
#include "control/supervisor.h"

namespace fs = std::filesystem;
using mortred::control::Catalog;
using mortred::control::ControlConfig;
using mortred::control::ProcessSupervisor;
using mortred::control::kGatewayId;
using mortred::control::process_group_gone;

namespace {

#ifndef MORTRED_FAKE_BIN_DEFAULT
#define MORTRED_FAKE_BIN_DEFAULT ""
#endif

int free_test_port() {
    return 38000 + (::getpid() % 20000);
}

int count_open_fds() {
    DIR* d = ::opendir("/proc/self/fd");
    if (d == nullptr) {
        return -1;
    }
    int count = 0;
    while (::readdir(d) != nullptr) {
        ++count;
    }
    ::closedir(d);
    return count;
}

// /proc/<pid>/environ is a NUL-separated copy of the child's initial envp,
// taken by the kernel at execve, so it proves exactly what spawn_locked handed
// to execve rather than what the child later mutated.
std::string read_child_environ(pid_t pid) {
    std::ifstream in("/proc/" + std::to_string(static_cast<long long>(pid)) + "/environ",
                     std::ios::in | std::ios::binary);
    if (!in.good()) {
        return std::string();
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Exact key match on the NUL-separated block: a substring search would also hit
// keys that merely start with the same text.
int count_env_key(const std::string& environ_block, const std::string& key) {
    const std::string needle = key + "=";
    int count = 0;
    size_t pos = 0;
    while (pos < environ_block.size()) {
        size_t end = environ_block.find('\0', pos);
        if (end == std::string::npos) {
            end = environ_block.size();
        }
        const size_t len = end - pos;
        if (len >= needle.size() && environ_block.compare(pos, needle.size(), needle) == 0) {
            ++count;
        }
        pos = end + 1;
    }
    return count;
}

class SupervisorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() /
                ("mortred_supervisor_test_" + std::to_string(::getpid()));
        fs::remove_all(root_, ec_);
        fs::create_directories(root_ / "conf" / "server" / "test", ec_);
        fs::create_directories(root_ / "bin", ec_);
        fs::create_directories(root_ / "lib", ec_);
        fs::create_directories(root_ / "logs", ec_);

        const char* fake_bin_env = std::getenv("MORTRED_FAKE_BIN");
        if (fake_bin_env == nullptr) {
            fake_bin_env = MORTRED_FAKE_BIN_DEFAULT;
        }
        ASSERT_NE(fake_bin_env, nullptr) << "MORTRED_FAKE_BIN env must point at fake_model_server";
        std::error_code copy_ec;
        fs::copy_file(fake_bin_env, root_ / "bin" / "fake_model_server.out",
                      fs::copy_options::overwrite_existing, copy_ec);
        ASSERT_FALSE(copy_ec) << copy_ec.message();

        port_ = free_test_port();
        write_fake_config("ready", 0);
        write_catalog();
    }

    void TearDown() override {
        ::unsetenv("MORTRED_ARGV_FILE");
        ::unsetenv("MORTRED_EXPOSE");
        ::unsetenv("MORTRED_PROFILE");
        fs::remove_all(root_, ec_);
    }

    void write_fake_config(const std::string& mode, int exit_after_ms, int exit_code = 0,
                           bool spawn_grandchild = false,
                           bool grandchild_ignores_sigint = false) {
        std::ofstream out(root_ / "conf" / "server" / "test" / "fake.toml");
        out << "[FAKE_SERVER]\n"
            << "port=" << port_ << "\n"
            << "host=\"localhost\"\n"
            << "server_uri=\"/mortred_ai_server_v1/test/fake\"\n"
            << "server_exe=\"fake_model_server.out\"\n"
            << "fake_port=" << port_ << "\n"
            << "fake_mode=\"" << mode << "\"\n"
            << "fake_exit_after_ms=" << exit_after_ms << "\n"
            << "fake_exit_code=" << exit_code << "\n"
            << "fake_spawn_grandchild=" << (spawn_grandchild ? 1 : 0) << "\n"
            << "fake_grandchild_ignores_sigint=" << (grandchild_ignores_sigint ? 1 : 0) << "\n";
    }

    void write_catalog() {
        ASSERT_TRUE(catalog_.init(root_.string(), &catalog_err_)) << catalog_err_;
    }

    std::unique_ptr<ProcessSupervisor> make_supervisor() {
        ControlConfig cfg;
        cfg.supervisor.bin_dir = "bin";
        cfg.supervisor.lib_dir = "lib";
        cfg.supervisor.libs_dir = "lib";
        cfg.supervisor.log_dir = "logs";
        cfg.supervisor.log_rotate_mb = 1;
        return std::make_unique<ProcessSupervisor>(
            root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    }

    // Gateway child is spawned with no argv; wrap fake_model_server so /healthz
    // answers on MORTRED_GATEWAY_PORT (ready) or sleep forever (never ready).
    void install_gateway_stub(const std::string& mode) {
        const fs::path stub = root_ / "bin" / "mortred-gateway.out";
        std::ofstream out(stub);
        if (mode == "never-ready") {
            out << "#!/bin/bash\nexec sleep 3600\n";
        } else {
            out << "#!/bin/bash\n"
                << "exec \"" << (root_ / "bin" / "fake_model_server.out").string()
                << "\" --port \"${MORTRED_GATEWAY_PORT:-8080}\" --mode ready\n";
        }
        out.close();
        fs::permissions(stub, fs::perms::owner_all | fs::perms::group_read |
                                  fs::perms::group_exec | fs::perms::others_read |
                                  fs::perms::others_exec);
    }

    // P1-3: the fake's grandchild announces "fake grandchild pid=<n> pgid=<n>
    // ignore_sigint=<0|1>" on stderr before dropping its log pipes; the
    // supervisor's reader thread copies that line into the child's LogBuffer.
    static std::string grandchild_log_line(ProcessSupervisor* sup) {
        auto log = sup->logs("fake_model_server");
        if (log == nullptr) {
            return std::string();
        }
        for (const auto& line : log->slice(0, 400)) {
            if (line.find("grandchild pid=") != std::string::npos) {
                return line;
            }
        }
        return std::string();
    }

    static pid_t pid_from_grandchild_line(const std::string& line) {
        const std::string needle = "grandchild pid=";
        const auto pos = line.find(needle);
        if (pos == std::string::npos) {
            return -1;
        }
        return static_cast<pid_t>(std::strtol(line.c_str() + pos + needle.size(), nullptr, 10));
    }

    // P1-3 setup: start the fake with a grandchild in the same process group and
    // prove the group topology the stop path relies on (child leads its own
    // group; the grandchild shares it) before anyone signals anything.
    void start_fake_with_grandchild(ProcessSupervisor* sup, bool grandchild_ignores_sigint,
                                    pid_t* leader, pid_t* grandchild) {
        std::string err;
        ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
        ASSERT_TRUE(wait_for([sup]() {
            const auto s = sup->status("fake_model_server");
            return s.state == "running" && s.ready;
        }, 10000)) << "fake server did not become ready";

        const auto st = sup->status("fake_model_server");
        ASSERT_GT(st.pid, 0);
        *leader = static_cast<pid_t>(st.pid);
        ASSERT_EQ(::getpgid(*leader), *leader) << "child did not get its own process group";
        ASSERT_NE(::getpgid(*leader), ::getpgid(::getpid()))
            << "child is still in the supervisor's process group";

        std::string line;
        ASSERT_TRUE(wait_for([&]() {
            line = grandchild_log_line(sup);
            return !line.empty();
        }, 5000)) << "grandchild never announced itself in the child log";
        EXPECT_NE(line.find(grandchild_ignores_sigint ? "ignore_sigint=1" : "ignore_sigint=0"),
                  std::string::npos)
            << line;
        *grandchild = pid_from_grandchild_line(line);
        ASSERT_GT(*grandchild, 0) << line;
        ASSERT_EQ(::getpgid(*grandchild), *leader)
            << "grandchild left the child's process group: " << line;
    }

    ControlConfig base_cfg() {
        ControlConfig cfg;
        cfg.supervisor.bin_dir = "bin";
        cfg.supervisor.lib_dir = "lib";
        cfg.supervisor.libs_dir = "lib";
        cfg.supervisor.log_dir = "logs";
        cfg.supervisor.log_rotate_mb = 1;
        cfg.gateway.port = free_test_port() + 1;
        return cfg;
    }

    static bool wait_for(const std::function<bool()>& pred, int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (pred()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return pred();
    }

    // wait until the fake server is ready; on failure dump its log + status
    // so supervision regressions are diagnosable from the test output alone
    static void expect_ready(ProcessSupervisor* sup, int timeout_ms = 10000) {
        if (wait_for([sup]() {
                const auto s = sup->status("fake_model_server");
                return s.state == "running" && s.ready;
            }, timeout_ms)) {
            SUCCEED();
            return;
        }
        const auto s = sup->status("fake_model_server");
        ADD_FAILURE() << "fake server not ready: state=" << s.state << " pid=" << s.pid
                      << " restarts=" << s.restart_count
                      << " last_exit=" << s.last_exit_status << " error=" << s.error;
        auto log = sup->logs("fake_model_server");
        if (log != nullptr) {
            for (const auto& line : log->slice(0, 80)) {
                ADD_FAILURE() << "[log] " << line;
            }
        }
    }

    fs::path root_;
    std::error_code ec_;
    int port_ = 0;
    Catalog catalog_;
    std::string catalog_err_;
};

}  // namespace

TEST_F(SupervisorTest, spawn_failure_schedules_backoff_retry) {
    fs::remove(root_ / "bin" / "fake_model_server.out");
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    EXPECT_FALSE(sup->start_server("fake_model_server", &err));
    EXPECT_NE(err.find("executable not found"), std::string::npos) << err;
    const auto first = sup->status("fake_model_server");
    EXPECT_EQ(first.state, "backoff") << first.error;
    EXPECT_GE(first.restart_count, 1);
    ASSERT_TRUE(wait_for([&sup, n = first.restart_count]() {
                    return sup->status("fake_model_server").restart_count > n;
                },
                4000))
        << "spawn failure must schedule Decision backoff so the monitor retries";
}

TEST_F(SupervisorTest, start_reaches_ready_state) {
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    expect_ready(sup.get());
}

TEST_F(SupervisorTest, unexpected_kill_triggers_backoff_restart) {
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));

    const int pid = sup->status("fake_model_server").pid;
    ASSERT_GT(pid, 0);
    ::kill(pid, SIGKILL);

    EXPECT_TRUE(wait_for([&sup]() {
        const auto s = sup->status("fake_model_server");
        return s.restart_count >= 1 && s.state == "running";
    }, 15000)) << "server was not restarted after SIGKILL";
}

TEST_F(SupervisorTest, crash_loop_gives_up_into_failed_state) {
    write_fake_config("exit-now", 0, 1);
    write_catalog();
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    // 5 restart decisions inside the crash window (delays 0.5+1+2+4+8s)
    EXPECT_TRUE(wait_for([&sup]() {
        return sup->status("fake_model_server").state == "failed";
    }, 30000));

    // manual start after give-up works again
    EXPECT_TRUE(sup->start_server("fake_model_server", &err)) << err;
}

TEST_F(SupervisorTest, manual_stop_is_expected_and_not_restarted) {
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));

    ASSERT_TRUE(sup->stop_server("fake_model_server", &err)) << err;
    const auto s = sup->status("fake_model_server");
    EXPECT_EQ(s.state, "stopped");
    EXPECT_LT(s.pid, 0);
    EXPECT_FALSE(s.ready);

    // a second stop reports "not running"
    EXPECT_FALSE(sup->stop_server("fake_model_server", &err));
}

TEST_F(SupervisorTest, stop_during_backoff_window_never_revives_the_process) {
    // P0-3b: the monitor collects respawn candidates under one lock hold and
    // spawns them under a later one. A stop_server landing between the two
    // must win: after stop returns, the process stays stopped even though a
    // respawn was pending when the stop was issued. The pre-fix interleaving
    // is microsecond-narrow, so this is an invariant guard looped across many
    // kill/backoff/stop rounds rather than a deterministic red/green repro.
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    for (int iter = 0; iter < 30; ++iter) {
        std::string err;
        ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
        ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000))
            << "iter=" << iter;

        const int pid = sup->status("fake_model_server").pid;
        ASSERT_GT(pid, 0);
        ::kill(pid, SIGKILL);
        // stop inside the backoff window: a respawn is pending but not yet
        // spawned (handle_exit scheduled the retry, the monitor has not run it)
        ASSERT_TRUE(wait_for([&sup]() {
            return sup->status("fake_model_server").state == "backoff";
        }, 10000)) << "iter=" << iter << ": child did not enter backoff";
        ASSERT_TRUE(sup->stop_server("fake_model_server", &err)) << err;

        // backoff delay starts at 500ms and the probe cycle is ~500ms-1s, so
        // a buggy revive would surface as running/backoff within this window
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(3500);
        while (std::chrono::steady_clock::now() < deadline) {
            const auto s = sup->status("fake_model_server");
            ASSERT_EQ(s.state, "stopped")
                << "iter=" << iter << ": stopped server was revived (state=" << s.state << ")";
            ASSERT_LT(s.pid, 0) << "iter=" << iter << ": stopped server has a pid again";
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}


TEST_F(SupervisorTest, restart_action_recovers_running_state) {
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));
    ASSERT_TRUE(sup->restart_server("fake_model_server", &err)) << err;
    EXPECT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));
}

TEST_F(SupervisorTest, restarts_do_not_leak_log_pipe_fds) {
    // regression: the child log pipe read ends were never closed (the Child
    // out_fd/err_fd fields were dead), leaking 2 fds per process lifetime.
    // Both the expected (restart_server) and the unexpected (SIGKILL -> backoff)
    // exit paths must keep the fd count flat.
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));

    int prev = count_open_fds();
    ASSERT_GT(prev, 0) << "cannot count /proc/self/fd";

    for (int round = 0; round < 5; ++round) {
        ASSERT_TRUE(sup->restart_server("fake_model_server", &err)) << err;
        ASSERT_TRUE(wait_for([&sup]() {
            const auto s = sup->status("fake_model_server");
            return s.state == "running" && s.ready;
        }, 15000)) << "restart round " << round << " did not recover";
        // handle_exit must have joined readers and closed fds before the count;
        // 200ms is generous for pipe EOF + join
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const int fds = count_open_fds();
        EXPECT_LE(fds, prev + 1) << "fd count grew after restart round " << round
                                 << " (before " << prev << ", after " << fds << ")";
        prev = fds;
    }

    const int pid = sup->status("fake_model_server").pid;
    ASSERT_GT(pid, 0);
    ::kill(pid, SIGKILL);
    ASSERT_TRUE(wait_for([&sup]() {
        const auto s = sup->status("fake_model_server");
        return s.state == "running" && s.ready && s.restart_count >= 1;
    }, 15000)) << "server did not recover after SIGKILL";
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const int fds = count_open_fds();
    EXPECT_LE(fds, prev + 1) << "fd count grew after unexpected exit (before " << prev
                             << ", after " << fds << ")";
}


TEST_F(SupervisorTest, autostart_models_respects_deadline) {
    // P1-B3: never-ready children used to pin autostart_all forever.
    // Gateway must be ready first or the model wave is skipped (P0-2).
    // P1-1 AC1: models deadline leaves models_failed with deadline + id in reason.
    write_fake_config("never-ready", 0);
    catalog_ = Catalog{};
    write_catalog();
    install_gateway_stub("ready");

    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    const auto t0 = std::chrono::steady_clock::now();
    sup->autostart_all(/*gateway_deadline_ms=*/3000, /*models_deadline_ms=*/800);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(ms, 5000) << "autostart_all must return after models deadline, took " << ms
                        << "ms";
    const auto gate = sup->autostart_gate();
    EXPECT_FALSE(gate.blocked);
    EXPECT_FALSE(gate.models_skipped);
    EXPECT_TRUE(gate.models_failed);
    EXPECT_NE(gate.reason.find("deadline"), std::string::npos) << gate.reason;
    EXPECT_NE(gate.reason.find("fake_model_server"), std::string::npos) << gate.reason;
    const auto s = sup->status("fake_model_server");
    EXPECT_EQ(s.state, "failed");
    EXPECT_NE(s.state, "starting");
    EXPECT_NE(s.state, "running");
    EXPECT_NE(s.error.find("deadline"), std::string::npos) << s.error;
    EXPECT_NE(s.error.find("fake_model_server"), std::string::npos) << s.error;
    EXPECT_GT(s.pid, 0) << "never-ready child should still be alive at deadline";
    const int pid_at_return = s.pid;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    const auto later = sup->status("fake_model_server");
    EXPECT_EQ(later.state, "failed");
    EXPECT_EQ(later.pid, pid_at_return);
}

TEST_F(SupervisorTest, autostart_per_model_start_timeout_overrides_default) {
    // [servers.<id>] start_timeout_ms is this model's own readiness budget; the
    // autostart_all models_deadline_ms is only the fallback. A 400 ms budget
    // must win over a 5 s default, otherwise a big engine loads past 60 s and
    // gets marked failed while it is still starting.
    write_fake_config("never-ready", 0);
    catalog_ = Catalog{};
    write_catalog();
    install_gateway_stub("ready");

    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    mortred::control::ServerPolicy policy;
    policy.start_timeout_ms = 400;
    cfg.servers["fake_model_server"] = policy;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    const auto t0 = std::chrono::steady_clock::now();
    sup->autostart_all(/*gateway_deadline_ms=*/3000, /*models_deadline_ms=*/5000);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(ms, 3000) << "per-model start_timeout_ms must win over the 5 s default, took "
                        << ms << " ms";
    const auto s = sup->status("fake_model_server");
    EXPECT_EQ(s.state, "failed");
    EXPECT_NE(s.error.find("deadline"), std::string::npos) << s.error;
}

TEST_F(SupervisorTest, autostart_running_sibling_stays_running_on_peer_deadline) {
    // Two model children in one wave. ready_sibling reaches running before the
    // models deadline; fake_model_server never becomes ready and trips it.
    write_fake_config("never-ready", 0);
    const int sibling_port = port_ + 2;
    {
        std::ofstream out(root_ / "conf" / "server" / "test" / "ready_sibling.toml");
        out << "[FAKE_SERVER]\n"
            << "port=" << sibling_port << "\n"
            << "host=\"localhost\"\n"
            << "server_uri=\"/mortred_ai_server_v1/test/ready_sibling\"\n"
            << "server_exe=\"ready_sibling.out\"\n"
            << "fake_port=" << sibling_port << "\n"
            << "fake_mode=\"ready\"\n"
            << "fake_exit_after_ms=0\n"
            << "fake_exit_code=0\n";
    }
    fs::copy_file(root_ / "bin" / "fake_model_server.out", root_ / "bin" / "ready_sibling.out",
                  fs::copy_options::overwrite_existing, ec_);
    ASSERT_FALSE(ec_) << ec_.message();
    fs::permissions(root_ / "bin" / "ready_sibling.out",
                    fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                        fs::perms::others_read | fs::perms::others_exec);
    catalog_ = Catalog{};
    write_catalog();
    install_gateway_stub("ready");

    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    cfg.supervisor.start_concurrency = 2;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    sup->autostart_all(/*gateway_deadline_ms=*/3000, /*models_deadline_ms=*/2000);

    EXPECT_EQ(sup->status("ready_sibling").state, "running");
    const auto gate = sup->autostart_gate();
    EXPECT_TRUE(gate.models_failed);
    EXPECT_FALSE(gate.models_skipped);
    EXPECT_NE(gate.reason.find("deadline"), std::string::npos) << gate.reason;
    EXPECT_NE(gate.reason.find("fake_model_server"), std::string::npos) << gate.reason;
}

TEST_F(SupervisorTest, autostart_reports_start_failure_without_hanging) {
    // Missing gateway binary blocks the wave immediately (no model deadline wait).
    fs::remove(root_ / "bin" / "fake_model_server.out");
    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    catalog_ = Catalog{};
    write_catalog();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    const auto t0 = std::chrono::steady_clock::now();
    sup->autostart_all(/*gateway_deadline_ms=*/1000, /*models_deadline_ms=*/2000);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(ms, 3000) << "failed autostart must not block on model deadline";
    const auto gate = sup->autostart_gate();
    EXPECT_TRUE(gate.blocked);
    EXPECT_TRUE(gate.models_skipped);
    EXPECT_FALSE(gate.models_failed);
    EXPECT_NE(gate.reason.find("binary missing"), std::string::npos) << gate.reason;
    EXPECT_LE(sup->status("fake_model_server").pid, 0);
}

TEST_F(SupervisorTest, autostart_models_failed_on_spawn_failure) {
    // P1-1 AC2: gateway admitted, model start_server fails → models_failed, not skipped.
    install_gateway_stub("ready");
    // Point the catalog at a missing exe while keeping fake_model_server.out for the
    // gateway stub (absolute path baked into mortred-gateway.out). Id becomes the
    // missing exe stem (catalog rule for fakes without model=).
    {
        std::ofstream out(root_ / "conf" / "server" / "test" / "fake.toml");
        out << "[FAKE_SERVER]\n"
            << "port=" << port_ << "\n"
            << "host=\"localhost\"\n"
            << "server_uri=\"/mortred_ai_server_v1/test/fake\"\n"
            << "server_exe=\"missing_model.out\"\n"
            << "fake_port=" << port_ << "\n"
            << "fake_mode=\"ready\"\n"
            << "fake_exit_after_ms=0\n"
            << "fake_exit_code=0\n";
    }
    catalog_ = Catalog{};
    write_catalog();

    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    const auto t0 = std::chrono::steady_clock::now();
    sup->autostart_all(/*gateway_deadline_ms=*/3000, /*models_deadline_ms=*/2000);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(ms, 4000) << "spawn failure must not hang on models deadline";

    const auto gate = sup->autostart_gate();
    EXPECT_FALSE(gate.blocked);
    EXPECT_FALSE(gate.models_skipped);
    EXPECT_TRUE(gate.models_failed);
    EXPECT_NE(gate.reason.find("missing_model"), std::string::npos) << gate.reason;
    EXPECT_NE(gate.reason.find("failed"), std::string::npos) << gate.reason;
    EXPECT_LE(sup->status("missing_model").pid, 0);
}

TEST_F(SupervisorTest, autostart_skips_models_when_scrape_token_missing) {
    // P1-1 AC3: empty scrape stays fail-closed; models_failed must stay false.
    install_gateway_stub("ready");
    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    // empty metrics → permanent scrape failure
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    sup->autostart_all(/*gateway_deadline_ms=*/1000, /*models_deadline_ms=*/2000);

    const auto gw = sup->status(kGatewayId);
    EXPECT_EQ(gw.state, "failed");
    EXPECT_NE(gw.error.find("gateway scrape token rejected:"), std::string::npos) << gw.error;
    const auto gate = sup->autostart_gate();
    EXPECT_TRUE(gate.blocked);
    EXPECT_TRUE(gate.models_skipped);
    EXPECT_FALSE(gate.models_failed);
    EXPECT_NE(gate.reason.find("scrape"), std::string::npos) << gate.reason;
    EXPECT_LE(sup->status("fake_model_server").pid, 0);
    EXPECT_NE(sup->status("fake_model_server").state, "starting");
    EXPECT_NE(sup->status("fake_model_server").state, "running");
}

TEST_F(SupervisorTest, autostart_skips_models_when_gateway_deadline_missed) {
    install_gateway_stub("never-ready");
    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    const auto t0 = std::chrono::steady_clock::now();
    sup->autostart_all(/*gateway_deadline_ms=*/600, /*models_deadline_ms=*/5000);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(ms, 3000) << "must return after gateway deadline without starting models";

    const auto gate = sup->autostart_gate();
    EXPECT_TRUE(gate.blocked);
    EXPECT_TRUE(gate.models_skipped);
    EXPECT_FALSE(gate.models_failed);
    EXPECT_TRUE(gate.reason.find("deadline") != std::string::npos ||
                gate.reason.find("not ready") != std::string::npos)
        << gate.reason;
    const auto gw = sup->status(kGatewayId);
    EXPECT_TRUE(gw.error.find("deadline") != std::string::npos ||
                gw.error.find("not ready") != std::string::npos)
        << gw.error;
    EXPECT_LE(sup->status("fake_model_server").pid, 0);
}

TEST_F(SupervisorTest, empty_server_uri_fails_readiness_immediately) {
    // P1-2 AC1–AC3: catalog still rejects empty server_uri; unittest injects
    // empty uri after a successful load so the probe path is observable.
    ASSERT_FALSE(catalog_.entries().empty());
    const_cast<mortred::control::ServerEntry&>(catalog_.entries()[0]).uri.clear();
    ASSERT_TRUE(catalog_.entries()[0].uri.empty());

    install_gateway_stub("ready");
    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    // models_deadline_ms far longer than the 2s AC1 bound — must not rely on it.
    constexpr int kModelsDeadlineMs = 60000;
    const auto t0 = std::chrono::steady_clock::now();
    sup->autostart_all(/*gateway_deadline_ms=*/3000, /*models_deadline_ms=*/kModelsDeadlineMs);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(ms, 2000) << "empty server_uri must fail readiness within 2s, took " << ms
                        << "ms (models_deadline_ms=" << kModelsDeadlineMs << ")";

    const auto s = sup->status("fake_model_server");
    EXPECT_EQ(s.state, "failed") << "error=" << s.error;
    EXPECT_NE(s.state, "starting");
    EXPECT_NE(s.error.find("server_uri"), std::string::npos) << s.error;
    EXPECT_TRUE(s.error.find("readiness") != std::string::npos ||
                s.error.find("probe") != std::string::npos)
        << s.error;

    const auto gate = sup->autostart_gate();
    EXPECT_FALSE(gate.blocked);
    EXPECT_FALSE(gate.models_skipped);
    EXPECT_TRUE(gate.models_failed);
    EXPECT_NE(gate.reason.find("server_uri"), std::string::npos) << gate.reason;
}

TEST_F(SupervisorTest, autostart_healthy_models_not_failed) {
    // P1-1 AC4: gateway + models reach running within deadlines → models_failed false.
    write_fake_config("ready", 0);
    catalog_ = Catalog{};
    write_catalog();
    install_gateway_stub("ready");

    ControlConfig cfg = base_cfg();
    cfg.supervisor.autostart_default = true;
    auto sup = std::make_unique<ProcessSupervisor>(
        root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    sup->autostart_all(/*gateway_deadline_ms=*/3000, /*models_deadline_ms=*/5000);

    const auto gate = sup->autostart_gate();
    EXPECT_FALSE(gate.blocked);
    EXPECT_FALSE(gate.models_skipped);
    EXPECT_FALSE(gate.models_failed);
    EXPECT_TRUE(gate.reason.empty()) << gate.reason;
    expect_ready(sup.get());
    EXPECT_EQ(sup->status(kGatewayId).state, "running");
}

TEST_F(SupervisorTest, rapid_restarts_do_not_double_join_readers) {
    // P1-B2: spawn_locked used to join reader threads that handle_exit also
    // joins. stop→start can overlap those joins (UB / terminate). Ownership is
    // now handle_exit-only; this hammer exercises the wait path.
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));

    for (int round = 0; round < 20; ++round) {
        ASSERT_TRUE(sup->restart_server("fake_model_server", &err))
            << "round " << round << ": " << err;
        ASSERT_TRUE(wait_for([&sup]() {
            const auto s = sup->status("fake_model_server");
            return s.state == "running" && s.ready;
        }, 15000))
            << "round " << round << " did not recover";
    }
}

TEST_F(SupervisorTest, spawn_passes_model_flag_for_unified_exe) {
    fs::copy_file(root_ / "bin" / "fake_model_server.out",
                  root_ / "bin" / "mortred-model-server.out",
                  fs::copy_options::overwrite_existing, ec_);
    ASSERT_FALSE(ec_) << ec_.message();
    // SetUp already registered fake.toml on port_; this case must be the only
    // catalog entry so init does not fail on a duplicate port.
    fs::remove(root_ / "conf" / "server" / "test" / "fake.toml");

    const auto argv_file = root_ / "spawn_argv.txt";
    // The child env is an allowlist (r14), so a test-only variable set here
    // would never reach the child. Hand the dump path over through the fake's
    // own config instead.
    ::unsetenv("MORTRED_ARGV_FILE");

    {
        std::ofstream out(root_ / "conf" / "server" / "test" / "yolov8.toml");
        out << "[YOLOV8_DETECTION_SERVER]\n"
            << "model=\"YOLOV8\"\n"
            << "port=" << port_ << "\n"
            << "host=\"localhost\"\n"
            << "server_uri=\"/mortred_ai_server_v1/obj_detection/yolov8\"\n"
            << "server_exe=\"mortred-model-server.out\"\n"
            << "fake_port=" << port_ << "\n"
            << "fake_mode=\"ready\"\n"
            << "fake_argv_file=\"" << argv_file.string() << "\"\n"
            << "\n"
            << "[YOLOV8]\n"
            << "model_config_file_path=\"x.toml\"\n";
    }
    write_catalog();

    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("YOLOV8", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() {
        const auto s = sup->status("YOLOV8");
        return s.state == "running" && s.ready;
    }, 10000)) << "YOLOV8 child did not become ready";

    std::ifstream in(argv_file);
    ASSERT_TRUE(in.good());
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("--model\nYOLOV8\n"), std::string::npos) << content;
    EXPECT_NE(content.find("yolov8.toml"), std::string::npos) << content;

    ::unsetenv("MORTRED_ARGV_FILE");
}

TEST_F(SupervisorTest, shutdown_is_ordered_and_complete) {
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));

    sup->request_shutdown();
    sup->wait_shutdown();
    EXPECT_EQ(sup->status("fake_model_server").state, "stopped");
    EXPECT_EQ(sup->status(kGatewayId).state, "stopped");
}

// P1-3: stop delivers SIGINT to the whole group, so a grandchild that does not
// ignore SIGINT dies with its leader and the stop never needs the escalation.
TEST_F(SupervisorTest, stop_signals_the_whole_process_group) {
    write_fake_config("ready", 0, 0, /*spawn_grandchild=*/true,
                      /*grandchild_ignores_sigint=*/false);
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    pid_t leader = -1;
    pid_t grandchild = -1;
    start_fake_with_grandchild(sup.get(), /*grandchild_ignores_sigint=*/false, &leader,
                               &grandchild);

    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(sup->stop_server("fake_model_server", &err)) << err;
    const auto stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    // grace is 5000ms: finishing well inside it proves the group SIGINT was
    // delivered, because a surviving grandchild would force the escalation
    EXPECT_LT(stop_ms, 4000) << "stop needed the SIGKILL escalation for a SIGINT-able group";
    EXPECT_TRUE(process_group_gone(leader))
        << "process group " << leader << " survived stop (took " << stop_ms << "ms)";
    EXPECT_EQ(::kill(grandchild, 0), -1) << "grandchild " << grandchild << " survived stop";
    EXPECT_EQ(sup->status("fake_model_server").state, "stopped");
}

// P1-3 counter-example: the grandchild ignores SIGINT, so the leader exits on
// the group SIGINT while the group stays alive. The stop must spend its grace
// budget on the group and then SIGKILL the survivor - a single-pid kill of the
// (already reaped) leader would leave it running forever.
TEST_F(SupervisorTest, stop_reaps_grandchild_that_ignores_sigint) {
    write_fake_config("ready", 0, 0, /*spawn_grandchild=*/true,
                      /*grandchild_ignores_sigint=*/true);
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    pid_t leader = -1;
    pid_t grandchild = -1;
    start_fake_with_grandchild(sup.get(), /*grandchild_ignores_sigint=*/true, &leader,
                               &grandchild);

    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(sup->stop_server("fake_model_server", &err)) << err;
    const auto stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    // grace(5000) + 2000: the whole group must be gone inside one budget
    EXPECT_LE(stop_ms, 7000) << "stop exceeded grace+2s";
    EXPECT_TRUE(wait_for([&]() { return process_group_gone(leader); }, 2000))
        << "process group " << leader << " survived stop (took " << stop_ms << "ms)";
    EXPECT_EQ(::kill(grandchild, 0), -1) << "grandchild " << grandchild << " survived stop";
    EXPECT_EQ(sup->status("fake_model_server").state, "stopped");
}

TEST_F(SupervisorTest, gateway_missing_scrape_token_is_permanent_failure) {
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    EXPECT_FALSE(sup->start_server(kGatewayId, &err));
    EXPECT_NE(err.find("MORTRED_METRICS_TOKEN"), std::string::npos) << err;
    const auto s = sup->status(kGatewayId);
    EXPECT_EQ(s.state, "failed") << s.error;
    EXPECT_NE(s.state, "backoff");
}

TEST_F(SupervisorTest, gateway_colliding_scrape_token_is_permanent_failure) {
    auto sup = make_supervisor();
    sup->set_gateway_trust("shared-secret", "shared-secret", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    EXPECT_FALSE(sup->start_server(kGatewayId, &err));
    EXPECT_NE(err.find("matches an inference or management token"), std::string::npos) << err;
    EXPECT_EQ(sup->status(kGatewayId).state, "failed");
}

TEST_F(SupervisorTest, gateway_missing_infer_token_is_permanent_failure) {
    auto sup = make_supervisor();
    sup->set_gateway_trust("scrape-token", "", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    EXPECT_FALSE(sup->start_server(kGatewayId, &err));
    EXPECT_NE(err.find("MORTRED_GATEWAY_AUTH_TOKEN is unset"), std::string::npos) << err;
    const auto s = sup->status(kGatewayId);
    EXPECT_EQ(s.state, "failed") << s.error;
    EXPECT_NE(s.state, "backoff");
}

TEST_F(SupervisorTest, gateway_missing_admin_token_is_permanent_failure) {
    auto sup = make_supervisor();
    sup->set_gateway_trust("scrape-token", "infer-token", "");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    EXPECT_FALSE(sup->start_server(kGatewayId, &err));
    EXPECT_NE(err.find("MORTRED_API_TOKEN is unset"), std::string::npos) << err;
    const auto s = sup->status(kGatewayId);
    EXPECT_EQ(s.state, "failed") << s.error;
    EXPECT_NE(s.state, "backoff");
}

TEST_F(SupervisorTest, gateway_identical_infer_and_admin_is_permanent_failure) {
    auto sup = make_supervisor();
    sup->set_gateway_trust("scrape-token", "same-token", "same-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    EXPECT_FALSE(sup->start_server(kGatewayId, &err));
    EXPECT_NE(err.find("MORTRED_GATEWAY_AUTH_TOKEN and MORTRED_API_TOKEN are identical"),
              std::string::npos)
        << err;
    const auto s = sup->status(kGatewayId);
    EXPECT_EQ(s.state, "failed") << s.error;
    EXPECT_NE(s.state, "backoff");
}

// P1-2: the operator's shell / systemd EnvironmentFile may export management
// and scrape credentials into the supervisor's own environment. spawn_locked
// copies environ, so without the drop these keys ride into every managed child.
// The invariant: a model child receives no management credentials (metrics scrape token is injected), and
// the gateway child receives only the two its role needs, each exactly once.
TEST_F(SupervisorTest, managed_children_do_not_inherit_management_credentials) {
    // long-lived child: the environ file must still exist at assert time
    write_fake_config("ready", 60000);

    auto sup = make_supervisor();
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    // poison the supervisor's environment AFTER construction so the injected
    // internal token stays distinct from the inherited MORTRED_INTERNAL_TOKEN
    ::setenv("MORTRED_API_TOKEN", "poison-api", 1);
    ::setenv("MORTRED_METRICS_TOKEN", "poison-metrics", 1);
    ::setenv("MORTRED_GATEWAY_AUTH_TOKEN", "poison-gateway-auth", 1);
    ::setenv("MORTRED_INTERNAL_TOKEN", "poison-internal", 1);

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() {
        const auto s = sup->status("fake_model_server");
        return s.state == "running" && s.ready;
    }, 10000)) << "fake server did not become ready";

    const auto model_status = sup->status("fake_model_server");
    ASSERT_GT(model_status.pid, 0);
    const std::string model_env = read_child_environ(model_status.pid);
    ASSERT_FALSE(model_env.empty()) << "/proc/<pid>/environ unreadable for model child";

    // Allowlist: model must NOT inherit poison credentials. MORTRED_METRICS_TOKEN
    // is re-injected from gateway trust (scrape-token), never the poison value.
    for (const char* key : {"MORTRED_API_TOKEN", "MORTRED_GATEWAY_AUTH_TOKEN",
                            "MORTRED_INTERNAL_TOKEN"}) {
        EXPECT_EQ(count_env_key(model_env, key), 0)
            << "model child inherited " << key << " from the supervisor environment";
    }
    EXPECT_EQ(count_env_key(model_env, "MORTRED_METRICS_TOKEN"), 1)
        << "model child should receive injected scrape metrics token";
    EXPECT_NE(model_env.find("MORTRED_METRICS_TOKEN=scrape-token"), std::string::npos);
    // the model's own credential still arrives, and is not the poisoned value
    EXPECT_EQ(count_env_key(model_env, "MORTRED_AUTH_TOKEN"), 1);
    EXPECT_EQ(model_env.find("poison-"), std::string::npos)
        << "a poisoned credential value reached the model child";

    // gateway child: role-appropriate credentials are re-injected exactly once.
    // The stub answers /healthz through the same fake binary, so readiness (and
    // therefore state=="running") is reached the normal way.
    install_gateway_stub("ready");

    ASSERT_TRUE(sup->start_server(kGatewayId, &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() {
        const auto s = sup->status(kGatewayId);
        return s.state == "running" && s.pid > 0;
    }, 10000)) << "gateway child did not start: state=" << sup->status(kGatewayId).state
                << " pid=" << sup->status(kGatewayId).pid
                << " restarts=" << sup->status(kGatewayId).restart_count
                << " last_exit=" << sup->status(kGatewayId).last_exit_status
                << " error=" << sup->status(kGatewayId).error;

    const auto gw_status = sup->status(kGatewayId);
    ASSERT_GT(gw_status.pid, 0);
    const std::string gw_env = read_child_environ(gw_status.pid);
    ASSERT_FALSE(gw_env.empty()) << "/proc/<pid>/environ unreadable for gateway child";

    EXPECT_EQ(count_env_key(gw_env, "MORTRED_METRICS_TOKEN"), 1)
        << "gateway lost its scrape token";
    EXPECT_EQ(count_env_key(gw_env, "MORTRED_GATEWAY_AUTH_TOKEN"), 1)
        << "gateway lost its inference token";
    EXPECT_EQ(count_env_key(gw_env, "MORTRED_API_TOKEN"), 1)
        << "gateway lost its management token";
    EXPECT_EQ(count_env_key(gw_env, "MORTRED_INTERNAL_TOKEN"), 1)
        << "gateway lost its internal token";
    EXPECT_EQ(gw_env.find("poison-"), std::string::npos)
        << "a poisoned credential value reached the gateway child";

    ::unsetenv("MORTRED_API_TOKEN");
    ::unsetenv("MORTRED_METRICS_TOKEN");
    ::unsetenv("MORTRED_GATEWAY_AUTH_TOKEN");
    ::unsetenv("MORTRED_INTERNAL_TOKEN");
}

// Container boot sets MORTRED_EXPOSE=docker and MORTRED_PROFILE=cpu on the
// supervisor process, then the supervisor injects MORTRED_GATEWAY_HOST=0.0.0.0.
// Those two policy vars must reach the gateway child. They must not reach a
// model child: models are forced to 127.0.0.1 and a forwarded docker expose
// would widen their bind if that override ever failed.
TEST_F(SupervisorTest, gateway_child_receives_expose_and_profile) {
    write_fake_config("ready", 60000);
    ::setenv("MORTRED_EXPOSE", "docker", 1);
    ::setenv("MORTRED_PROFILE", "cpu", 1);

    auto sup = make_supervisor();
    sup->set_gateway_trust("scrape-token", "infer-token", "admin-token");
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for(
        [&sup]() {
            const auto s = sup->status("fake_model_server");
            return s.state == "running" && s.ready;
        },
        10000))
        << "fake server did not become ready";

    const auto model_status = sup->status("fake_model_server");
    ASSERT_GT(model_status.pid, 0);
    const std::string model_env = read_child_environ(model_status.pid);
    ASSERT_FALSE(model_env.empty());
    EXPECT_EQ(count_env_key(model_env, "MORTRED_EXPOSE"), 0);
    EXPECT_EQ(count_env_key(model_env, "MORTRED_PROFILE"), 0);

    install_gateway_stub("ready");
    ASSERT_TRUE(sup->start_server(kGatewayId, &err)) << err;
    ASSERT_TRUE(wait_for(
        [&sup]() {
            const auto s = sup->status(kGatewayId);
            return s.state == "running" && s.pid > 0;
        },
        10000))
        << "gateway child did not start: state=" << sup->status(kGatewayId).state << " error=" << sup->status(kGatewayId).error;

    const auto gw_status = sup->status(kGatewayId);
    ASSERT_GT(gw_status.pid, 0);
    const std::string gw_env = read_child_environ(gw_status.pid);
    ASSERT_FALSE(gw_env.empty()) << "/proc/<pid>/environ unreadable for gateway child";
    EXPECT_EQ(count_env_key(gw_env, "MORTRED_EXPOSE"), 1);
    EXPECT_NE(gw_env.find("MORTRED_EXPOSE=docker"), std::string::npos) << gw_env;
    EXPECT_EQ(count_env_key(gw_env, "MORTRED_PROFILE"), 1);
    EXPECT_NE(gw_env.find("MORTRED_PROFILE=cpu"), std::string::npos) << gw_env;

    ::unsetenv("MORTRED_EXPOSE");
    ::unsetenv("MORTRED_PROFILE");
}

// A child that refuses to start exits before its stderr is in the supervisor
// log. The reap path must still put the wait status on child->error so an
// autostart deadline line is not "error=" with nothing after it.
TEST_F(SupervisorTest, unexpected_nonzero_exit_is_recorded_on_child_error) {
    write_fake_config("exit-now", 0, 7);
    write_catalog();

    ControlConfig cfg = base_cfg();
    cfg.servers["fake_model_server"].enabled = true;
    cfg.servers["fake_model_server"].has_restart_policy = true;
    cfg.servers["fake_model_server"].restart_policy = "no";
    auto sup = std::make_unique<ProcessSupervisor>(root_.string(), cfg, (root_ / "conf" / "mortred.toml").string());
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for(
        [&sup]() {
            const auto s = sup->status("fake_model_server");
            return s.state == "stopped" && s.has_last_exit;
        },
        10000))
        << "child did not settle after exit";

    const auto s = sup->status("fake_model_server");
    EXPECT_EQ(s.last_exit_status, 7);
    EXPECT_LT(s.pid, 0);
    EXPECT_NE(s.error.find("exited 7"), std::string::npos) << s.error;
}

TEST_F(SupervisorTest, respawn_archives_ended_run_and_keeps_current_log_clean) {
    // P0-3a: respawn used to truncate the child log in place, destroying the
    // ended run's evidence. The second spawn must archive the first run's
    // file (run-stamped) and write a fresh current log.
    auto sup = make_supervisor();
    sup->set_catalog(catalog_);
    ASSERT_TRUE(sup->start_threads());

    std::string err;
    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));
    const int first_pid = sup->status("fake_model_server").pid;
    ASSERT_GT(first_pid, 0);
    ASSERT_TRUE(sup->stop_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() {
        return sup->status("fake_model_server").state == "stopped";
    }, 10000));

    ASSERT_TRUE(sup->start_server("fake_model_server", &err)) << err;
    ASSERT_TRUE(wait_for([&sup]() { return sup->status("fake_model_server").ready; }, 10000));
    const int second_pid = sup->status("fake_model_server").pid;
    ASSERT_GT(second_pid, 0);
    ASSERT_NE(second_pid, first_pid);

    // exactly one archive exists and it carries the first run's spawn line
    int archives = 0;
    bool first_run_archived = false;
    for (const auto& entry : fs::directory_iterator(root_ / "logs")) {
        const std::string name = entry.path().filename().string();
        if (name.find("fake_model_server.log.run") == std::string::npos) {
            continue;
        }
        ++archives;
        std::ifstream in(entry.path());
        ASSERT_TRUE(in.good());
        std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
        if (content.find("spawn pid " + std::to_string(first_pid)) != std::string::npos) {
            first_run_archived = true;
        }
        EXPECT_EQ(content.find("spawn pid " + std::to_string(second_pid)), std::string::npos)
            << "second-run line leaked into the archive";
    }
    EXPECT_EQ(archives, 1);
    EXPECT_TRUE(first_run_archived) << "first run's evidence was not archived";

    // current log holds only the second run
    auto log = sup->logs("fake_model_server");
    ASSERT_NE(log, nullptr);
    for (const auto& line : log->slice(0, 100)) {
        EXPECT_EQ(line.find("spawn pid " + std::to_string(first_pid)), std::string::npos)
            << "first-run line survived in the current log: " << line;
    }

    ASSERT_TRUE(sup->stop_server("fake_model_server", &err)) << err;
}

int main(int argc, char** argv) {
    // supervision signals must be blocked before any thread exists
    ProcessSupervisor::block_supervision_signals();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
