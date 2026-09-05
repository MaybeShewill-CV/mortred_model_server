/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: trt_spawn_gate_unittest.cc
* Date: 26-9-5
************************************************/

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "control/trt_spawn_gate.h"

namespace fs = std::filesystem;
using mortred::control::is_trt_gate_error;
using mortred::control::trt_engines_ready_for_spawn;
using mortred::control::trt_gate_error;

namespace {

class TrtSpawnGateTest : public ::testing::Test {
  protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / ("mortred_trt_gate_" + std::to_string(::getpid()));
        fs::remove_all(root_, ec_);
        fs::create_directories(root_ / "_bin", ec_);
        fs::create_directories(root_ / "conf" / "server", ec_);
        fs::create_directories(root_ / "conf" / "model", ec_);
        fs::create_directories(root_ / "weights", ec_);
        std::ofstream server(root_ / "conf" / "server" / "x.toml");
        server << "[X_SERVER]\nmodel=\"X\"\nport=1\nserver_uri=\"/x\"\n"
               << "server_exe=\"mortred-model-server.out\"\n"
               << "[X]\nmodel_config_file_path=\"../conf/model/x.toml\"\n";
    }
    void TearDown() override {
        fs::remove_all(root_, ec_);
    }
    void write_model(const std::string& body) {
        std::ofstream out(root_ / "conf" / "model" / "x.toml");
        out << body;
    }
    fs::path root_;
    std::error_code ec_;
};

}  // namespace

TEST_F(TrtSpawnGateTest, mnn_backend_is_not_gated) {
    write_model("[X.backend]\ntype=\"mnn\"\nmodel_file_path=\"../weights/x.mnn\"\n");
    std::string err;
    EXPECT_TRUE(trt_engines_ready_for_spawn(root_.string(), "_bin",
                                            (root_ / "conf" / "server" / "x.toml").string(),
                                            "", &err))
        << err;
}

TEST_F(TrtSpawnGateTest, missing_engine_fails_closed) {
    write_model("[X.backend]\ntype=\"tensorrt\"\nmodel_file_path=\"../weights/x.engine\"\n");
    std::string err;
    EXPECT_FALSE(trt_engines_ready_for_spawn(root_.string(), "_bin",
                                             (root_ / "conf" / "server" / "x.toml").string(),
                                             "", &err));
    EXPECT_TRUE(is_trt_gate_error(err)) << err;
    EXPECT_NE(err.find("prepare"), std::string::npos);
}

TEST_F(TrtSpawnGateTest, nonempty_engine_passes) {
    write_model("[X.backend]\ntype=\"tensorrt\"\nmodel_file_path=\"../weights/x.engine\"\n");
    std::ofstream engine(root_ / "weights" / "x.engine", std::ios::binary);
    engine << "payload";
    engine.close();
    std::string err;
    EXPECT_TRUE(trt_engines_ready_for_spawn(root_.string(), "_bin",
                                            (root_ / "conf" / "server" / "x.toml").string(),
                                            "", &err))
        << err;
}

TEST_F(TrtSpawnGateTest, empty_engine_fails) {
    write_model("[X.backend]\ntype=\"tensorrt\"\nmodel_file_path=\"../weights/x.engine\"\n");
    std::ofstream(root_ / "weights" / "x.engine", std::ios::binary).close();
    std::string err;
    EXPECT_FALSE(trt_engines_ready_for_spawn(root_.string(), "_bin",
                                             (root_ / "conf" / "server" / "x.toml").string(),
                                             "", &err));
    EXPECT_TRUE(is_trt_gate_error(err)) << err;
}

TEST_F(TrtSpawnGateTest, pack_override_model_config) {
    write_model("[X.backend]\ntype=\"mnn\"\nmodel_file_path=\"../weights/x.mnn\"\n");
    const auto override = root_ / "conf" / "model" / "trt.toml";
    std::ofstream out(override);
    out << "[X.backend]\ntype=\"tensorrt\"\nmodel_file_path=\"../weights/y.engine\"\n";
    out.close();
    std::string err;
    EXPECT_FALSE(trt_engines_ready_for_spawn(root_.string(), "_bin",
                                             (root_ / "conf" / "server" / "x.toml").string(),
                                             override.string(), &err));
    EXPECT_TRUE(is_trt_gate_error(err)) << err;
}

TEST(trt_spawn_gate, error_prefix) {
    EXPECT_TRUE(is_trt_gate_error(trt_gate_error("/tmp/x.engine")));
    EXPECT_FALSE(is_trt_gate_error("fork() failed"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
