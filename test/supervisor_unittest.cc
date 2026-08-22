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

#include <unistd.h>
#include <signal.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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

namespace {

#ifndef MORTRED_FAKE_BIN_DEFAULT
#define MORTRED_FAKE_BIN_DEFAULT ""
#endif

int free_test_port() {
    return 38000 + (::getpid() % 20000);
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
        fs::remove_all(root_, ec_);
    }

    void write_fake_config(const std::string& mode, int exit_after_ms, int exit_code = 0) {
        std::ofstream out(root_ / "conf" / "server" / "test" / "fake.toml");
        out << "[FAKE_SERVER]\n"
            << "port=" << port_ << "\n"
            << "host=\"localhost\"\n"
            << "server_uri=\"/mortred_ai_server_v1/test/fake\"\n"
            << "server_exe=\"fake_model_server.out\"\n"
            << "fake_port=" << port_ << "\n"
            << "fake_mode=\"" << mode << "\"\n"
            << "fake_exit_after_ms=" << exit_after_ms << "\n"
            << "fake_exit_code=" << exit_code << "\n";
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
        auto* log = sup->logs("fake_model_server");
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

int main(int argc, char** argv) {
    // supervision signals must be blocked before any thread exists
    ProcessSupervisor::block_supervision_signals();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
