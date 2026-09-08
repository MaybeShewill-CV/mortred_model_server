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
#include "control/mini_toml.h"

namespace fs = std::filesystem;
using mortred::control::Catalog;
using mortred::control::mini_toml::Doc;
using mortred::control::mini_toml::Table;

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

TEST_F(CatalogTest, parses_model_identity_and_default_exe) {
    write_server("yolov8.toml",
                 "[YOLOV8_DETECTION_SERVER]\n"
                 "model=\"YOLOV8\"\n"
                 "port=39010\n"
                 "host=\"localhost\"\n"
                 "server_uri=\"/mortred_ai_server_v1/obj_detection/yolov8\"\n"
                 "\n"
                 "[YOLOV8]\n"
                 "model_config_file_path=\"../conf/model/object_detection/yolov8/yolov8_config.toml\"\n");
    Catalog catalog;
    std::string err;
    ASSERT_TRUE(catalog.init(root_.string(), &err)) << err;
    ASSERT_EQ(catalog.entries().size(), 1u);
    const auto* e = catalog.find("YOLOV8");
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->model, "YOLOV8");
    EXPECT_EQ(e->id, "YOLOV8");
    EXPECT_EQ(e->exe, "mortred-model-server.out");
    EXPECT_EQ(catalog.find("mortred-model-server"), nullptr);
}

TEST_F(CatalogTest, model_must_match_non_server_table) {
    write_server("bad.toml",
                 "[YOLOV8_DETECTION_SERVER]\n"
                 "model=\"MOBILENETV2\"\n"
                 "port=39011\n"
                 "host=\"localhost\"\n"
                 "server_uri=\"/bad\"\n"
                 "server_exe=\"mortred-model-server.out\"\n"
                 "\n"
                 "[YOLOV8]\n"
                 "model_config_file_path=\"x.toml\"\n");
    Catalog catalog;
    std::string err;
    EXPECT_FALSE(catalog.init(root_.string(), &err));
    EXPECT_NE(err.find("must equal the non-_SERVER table name"), std::string::npos) << err;
}

TEST_F(CatalogTest, profile_unknown_falls_back_to_gpu) {
    write_server("fake.toml", kValidServer);
    std::string err;
    Catalog weird;
    ASSERT_TRUE(weird.init(root_.string(), &err, "tpu")) << err;  // treated as gpu
    EXPECT_EQ(weird.entries().size(), 1u);
}

TEST_F(CatalogTest, unreadable_toml_is_fatal) {
    write_server("unreadable.toml", kValidServer);
    const auto path = root_ / "conf" / "server" / "object_detection" / "unreadable.toml";
    std::error_code pec;
    fs::permissions(path, fs::perms::none, pec);
    ASSERT_FALSE(pec) << pec.message();
    std::ifstream probe(path);
    if (probe.good()) {
        fs::permissions(path, fs::perms::owner_all, pec);
        GTEST_SKIP() << "platform still allows reading chmod 000 files";
    }
    Catalog catalog;
    std::string err;
    EXPECT_FALSE(catalog.init(root_.string(), &err));
    EXPECT_NE(err.find("failed to load TOML"), std::string::npos) << err;
    fs::permissions(path, fs::perms::owner_all, pec);
}

const Table* server_table_of(const Doc& doc) {
    for (const auto& [sec, table] : doc) {
        if (sec.size() > 7 && sec.compare(sec.size() - 7, 7, "_SERVER") == 0) {
            return &table;
        }
    }
    return nullptr;
}

void expect_real_conf_tree_boots(const std::string& profile) {
    const fs::path root = fs::current_path();
    const fs::path conf = root / "conf" / "server";
    ASSERT_TRUE(fs::is_directory(conf))
        << "conf/server not found under " << root
        << "; run from the repository root (ctest WORKING_DIRECTORY)";

    size_t eligible = 0;
    for (const auto& file : fs::recursive_directory_iterator(conf)) {
        if (!file.is_regular_file() || file.path().extension() != ".toml") {
            continue;
        }
        Doc doc;
        ASSERT_TRUE(mortred::control::mini_toml::load(file.path().string(), &doc))
            << file.path();
        const auto* kv = server_table_of(doc);
        ASSERT_NE(kv, nullptr) << file.path() << " has no *_SERVER section";
        const std::string entry_profile =
            kv->count("profile") != 0 ? kv->at("profile") : "gpu";
        if (entry_profile == "any" || entry_profile == profile) {
            ++eligible;
        }
    }

    Catalog catalog;
    std::string err;
    ASSERT_TRUE(catalog.init(root.string(), &err, profile)) << err;
    ASSERT_EQ(catalog.entries().size(), eligible)
        << profile << " catalog skipped or dropped files";
    ASSERT_FALSE(catalog.entries().empty()) << profile << " catalog is empty";
    for (const auto& e : catalog.entries()) {
        EXPECT_GT(e.port, 0) << e.id;
        ASSERT_FALSE(e.uri.empty()) << e.id;
        EXPECT_EQ(e.uri.front(), '/');
        EXPECT_FALSE(e.id.empty());
        EXPECT_TRUE(e.profile == profile || e.profile == "any")
            << e.id << " profile=" << e.profile;
        EXPECT_EQ(e.exe, mortred::control::kUnifiedServerExe) << e.id;
    }
}

TEST(CatalogRealTree, boots_gpu_profile) {
    ASSERT_NO_FATAL_FAILURE(expect_real_conf_tree_boots("gpu"));
    Catalog catalog;
    std::string err;
    ASSERT_TRUE(catalog.init(fs::current_path().string(), &err, "gpu")) << err;
    EXPECT_NE(catalog.find("YOLOV8"), nullptr);
    EXPECT_NE(catalog.find("MOBILENETV2"), nullptr);
}

TEST(CatalogRealTree, boots_cpu_profile) {
    ASSERT_NO_FATAL_FAILURE(expect_real_conf_tree_boots("cpu"));
    Catalog catalog;
    std::string err;
    ASSERT_TRUE(catalog.init(fs::current_path().string(), &err, "cpu")) << err;
    EXPECT_NE(catalog.find("MOBILENETV2"), nullptr);
    EXPECT_NE(catalog.find("RESNET"), nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
