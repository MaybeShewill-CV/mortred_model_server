/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: catalog_unittest.cc
* Date: 26-8-22
************************************************/

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "control/catalog.h"

namespace fs = std::filesystem;
using mortred::control::Catalog;

namespace {

class CatalogTest : public ::testing::Test {
  protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() /
                ("mortred_catalog_test_" + std::to_string(::getpid()));
        fs::remove_all(root_, ec_);
        fs::create_directories(root_ / "conf" / "server" / "object_detection", ec_);
    }
    void TearDown() override {
        fs::remove_all(root_, ec_);
    }

    void write_server(const std::string& name, const std::string& body) {
        const auto path = root_ / "conf" / "server" / "object_detection" / name;
        std::ofstream out(path);
        out << body;
    }

    fs::path root_;
    std::error_code ec_;
};

const char* kValidServer =
    "[FAKE_SERVER]\n"
    "port=39001\n"
    "host=\"localhost\"\n"
    "server_uri=\"/mortred_ai_server_v1/obj_detection/fake\"\n"
    "server_exe=\"fake_model_server.out\"\n";

}  // namespace

TEST_F(CatalogTest, parses_server_entry) {
    write_server("fake.toml", kValidServer);
    Catalog catalog;
    std::string err;
    ASSERT_TRUE(catalog.init(root_.string(), &err)) << err;
    ASSERT_EQ(catalog.entries().size(), 1u);
    const auto* e = catalog.find("fake_model_server");
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->port, 39001);
    EXPECT_EQ(e->uri, "/mortred_ai_server_v1/obj_detection/fake");
    EXPECT_EQ(e->exe, "fake_model_server.out");
    EXPECT_EQ(e->category, "object_detection");
    EXPECT_NE(catalog.find_by_uri("/mortred_ai_server_v1/obj_detection/fake"), nullptr);
}

TEST_F(CatalogTest, duplicate_port_is_fatal) {
    write_server("a.toml", kValidServer);
    write_server("b.toml",
                 "[OTHER_SERVER]\n"
                 "port=39001\n"
                 "server_uri=\"/other\"\n"
                 "server_exe=\"other.out\"\n");
    Catalog catalog;
    std::string err;
    EXPECT_FALSE(catalog.init(root_.string(), &err));
    EXPECT_NE(err.find("duplicate model server port"), std::string::npos) << err;
}

TEST_F(CatalogTest, duplicate_uri_is_fatal) {
    write_server("a.toml", kValidServer);
    write_server("b.toml",
                 "[OTHER_SERVER]\n"
                 "port=39002\n"
                 "server_uri=\"/mortred_ai_server_v1/obj_detection/fake\"\n"
                 "server_exe=\"other.out\"\n");
    Catalog catalog;
    std::string err;
    EXPECT_FALSE(catalog.init(root_.string(), &err));
    EXPECT_NE(err.find("duplicate server_uri"), std::string::npos) << err;
}

TEST_F(CatalogTest, missing_server_exe_is_skipped_not_fatal) {
    write_server("noexe.toml",
                 "[NOEXE_SERVER]\n"
                 "port=39003\n"
                 "server_uri=\"/noexe\"\n");
    write_server("fake.toml", kValidServer);
    Catalog catalog;
    std::string err;
    ASSERT_TRUE(catalog.init(root_.string(), &err)) << err;
    EXPECT_EQ(catalog.entries().size(), 1u);
}

TEST_F(CatalogTest, invalid_uri_is_fatal) {
    write_server("bad.toml",
                 "[BAD_SERVER]\n"
                 "port=39004\n"
                 "server_uri=\"no-leading-slash\"\n"
                 "server_exe=\"bad.out\"\n");
    Catalog catalog;
    std::string err;
    EXPECT_FALSE(catalog.init(root_.string(), &err));
}

TEST_F(CatalogTest, profile_cpu_filters_gpu_entries) {
    // same exe + port in two files: only one variant set is active per run,
    // so the duplicates never collide (filter runs before dedup checks)
    write_server("fake_gpu.toml", kValidServer);
    write_server("fake_cpu.toml",
                 "[FAKE_CPU_SERVER]\n"
                 "profile=\"cpu\"\n"
                 "port=39001\n"  // same port as the gpu variant: OK
                 "host=\"localhost\"\n"
                 "server_uri=\"/mortred_ai_server_v1/obj_detection/fake_cpu\"\n"
                 "server_exe=\"fake_cpu_model_server.out\"\n");
    std::string err;

    Catalog gpu;
    ASSERT_TRUE(gpu.init(root_.string(), &err, "gpu")) << err;
    EXPECT_EQ(gpu.entries().size(), 1u);
    EXPECT_EQ(gpu.entries()[0].profile, "gpu");  // absent field defaults to gpu
    EXPECT_EQ(gpu.find("fake_cpu_model_server"), nullptr);

    Catalog cpu;
    ASSERT_TRUE(cpu.init(root_.string(), &err, "cpu")) << err;
    EXPECT_EQ(cpu.entries().size(), 1u);
    ASSERT_NE(cpu.find("fake_cpu_model_server"), nullptr);
    EXPECT_EQ(cpu.find("fake_cpu_model_server")->profile, "cpu");
    EXPECT_EQ(cpu.find("fake_model_server"), nullptr);
}

TEST_F(CatalogTest, profile_any_appears_in_both) {
    write_server("any.toml",
                 "[ANY_SERVER]\n"
                 "profile=\"any\"\n"
                 "port=39002\n"
                 "server_uri=\"/mortred_ai_server_v1/obj_detection/any\"\n"
                 "server_exe=\"any_model_server.out\"\n");
    std::string err;
    Catalog gpu;
    ASSERT_TRUE(gpu.init(root_.string(), &err, "gpu")) << err;
    EXPECT_NE(gpu.find("any_model_server"), nullptr);
    Catalog cpu;
    ASSERT_TRUE(cpu.init(root_.string(), &err, "cpu")) << err;
    EXPECT_NE(cpu.find("any_model_server"), nullptr);
}

TEST_F(CatalogTest, profile_unknown_falls_back_to_gpu) {
    write_server("fake.toml", kValidServer);
    std::string err;
    Catalog weird;
    ASSERT_TRUE(weird.init(root_.string(), &err, "tpu")) << err;  // treated as gpu
    EXPECT_EQ(weird.entries().size(), 1u);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
