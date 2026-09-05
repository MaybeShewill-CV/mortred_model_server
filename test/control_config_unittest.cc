/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: control_config_unittest.cc
* Date: 26-8-22
************************************************/

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "control/control_config.h"

namespace fs = std::filesystem;
using mortred::control::ControlConfig;

namespace {

class ControlConfigTest : public ::testing::Test {
  protected:
    void SetUp() override {
        path_ = fs::temp_directory_path() / "mortred_control_config_test.toml";
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove(path_, ec);
    }
    void write(const std::string& content) {
        std::ofstream out(path_);
        out << content;
    }
    fs::path path_;
};

}  // namespace

TEST_F(ControlConfigTest, defaults_when_file_missing) {
    ControlConfig cfg;
    std::string err;
    // a missing file is an error (explicit path), defaults come from the struct
    EXPECT_FALSE(ControlConfig::load("/nonexistent/mortred.toml", &cfg, &err));
    EXPECT_FALSE(err.empty());
}

TEST_F(ControlConfigTest, parses_all_sections) {
    write(
        "[supervisor]\n"
        "api_port = 9000\n"
        "autostart_default = true\n"
        "start_concurrency = 4\n"
        "[gateway]\n"
        "port = 8081\n"
        "upstream_recv_timeout_ms = 60000\n"
        "[servers.fake_model_server]\n"
        "autostart = false\n"
        "restart_policy = \"always\"\n");
    ControlConfig cfg;
    std::string err;
    ASSERT_TRUE(ControlConfig::load(path_.string(), &cfg, &err)) << err;
    EXPECT_EQ(cfg.supervisor.api_port, 9000);
    EXPECT_TRUE(cfg.supervisor.autostart_default);
    EXPECT_EQ(cfg.supervisor.start_concurrency, 4);
    EXPECT_EQ(cfg.gateway.port, 8081);
    EXPECT_EQ(cfg.gateway.upstream_recv_timeout_ms, 60000);

    const auto policy = cfg.effective_policy("fake_model_server");
    EXPECT_FALSE(policy.autostart);
    EXPECT_EQ(policy.restart_policy, "always");

    const auto fallback = cfg.effective_policy("unknown_server");
    EXPECT_TRUE(fallback.autostart);  // falls back to autostart_default=true
    EXPECT_EQ(fallback.restart_policy, "on-failure");
}

TEST_F(ControlConfigTest, pack_limits_autostart_to_listed_ids) {
    write("[supervisor]\nautostart_default = true\n");
    ControlConfig cfg;
    std::string err;
    ASSERT_TRUE(ControlConfig::load(path_.string(), &cfg, &err)) << err;

    const fs::path pack = fs::temp_directory_path() / "mortred_pack_test.toml";
    {
        std::ofstream out(pack);
        out << "[pack.MOBILENETV2]\nworker_nums = 4\n";
    }
    ASSERT_TRUE(ControlConfig::apply_pack(pack.string(), {"MOBILENETV2", "YOLOV8"},
                                          fs::temp_directory_path().string(), &cfg, &err))
        << err;
    EXPECT_TRUE(cfg.supervisor.pack_active);
    const auto listed = cfg.effective_policy("MOBILENETV2");
    EXPECT_TRUE(listed.autostart);
    EXPECT_TRUE(listed.has_worker_nums);
    EXPECT_EQ(listed.worker_nums, 4);
    const auto other = cfg.effective_policy("YOLOV8");
    EXPECT_FALSE(other.autostart);
    EXPECT_FALSE(cfg.effective_policy("DBNET").autostart);

    std::error_code ec;
    fs::remove(pack, ec);
}

TEST_F(ControlConfigTest, pack_rejects_unknown_id) {
    write("[supervisor]\n");
    ControlConfig cfg;
    std::string err;
    ASSERT_TRUE(ControlConfig::load(path_.string(), &cfg, &err)) << err;
    const fs::path pack = fs::temp_directory_path() / "mortred_pack_bad.toml";
    {
        std::ofstream out(pack);
        out << "[pack.NOT_A_MODEL]\nworker_nums = 1\n";
    }
    EXPECT_FALSE(ControlConfig::apply_pack(pack.string(), {"MOBILENETV2"},
                                           fs::temp_directory_path().string(), &cfg, &err));
    EXPECT_NE(err.find("unknown catalog id"), std::string::npos);
    std::error_code ec;
    fs::remove(pack, ec);
}

TEST_F(ControlConfigTest, rejects_invalid_restart_policy) {
    write("[servers.x]\nrestart_policy = \"sometimes\"\n");
    ControlConfig cfg;
    std::string err;
    EXPECT_FALSE(ControlConfig::load(path_.string(), &cfg, &err));
    EXPECT_NE(err.find("restart_policy"), std::string::npos);
}

TEST_F(ControlConfigTest, rejects_out_of_range_port) {
    write("[gateway]\nport = 70000\n");
    ControlConfig cfg;
    std::string err;
    EXPECT_FALSE(ControlConfig::load(path_.string(), &cfg, &err));
    EXPECT_NE(err.find("'port'"), std::string::npos);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
