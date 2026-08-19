/************************************************
 * Author: Codex
 * File: config_schema_test.cc
 *
 * Machine-enforced config schema coverage:
 * - every conf/server/*.toml [*_SERVER] section must pass the strict server
 *   schema (missing required keys / wrong types / probable typos fail);
 * - every conf/model/*.toml section must pass the common-MNN contract;
 * - negative cases pin the fail-fast behavior (typo suggestion, type error,
 *   missing key) and the forward-compat warning behavior.
 ************************************************/

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <toml/toml.hpp>

#include "server/server_config_schema.h"
#include "models/model_config_schema.h"

using jinq::server::validate_server_section;
using jinq::models::validate_model_config_section;

namespace {

std::vector<std::string> collect_files(const std::string& dir, const std::string& ext) {
    std::vector<std::string> out;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ext) {
            out.push_back(entry.path().string());
        }
    }
    return out;
}

toml::table parse_text(const std::string& text) {
    std::istringstream iss(text);
    auto parsed = toml::parse(iss);
    auto root = std::move(parsed).table();
    // 单元测试的文本只含一个 [X]/[M] 小节；校验器期望的是小节内表而非根表
    if (!root.empty()) {
        if (auto* section = root.begin()->second.as_table()) {
            return std::move(*section);
        }
    }
    return root;
}

bool is_server_section(const std::string& key) {
    const std::string suffix = "_SERVER";
    return key.size() > suffix.size() &&
           key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

TEST(config_schema, every_server_section_passes) {
    const auto files = collect_files("conf/server", ".toml");
    ASSERT_FALSE(files.empty());
    int checked = 0;
    for (const auto& path : files) {
        auto parsed = toml::parse_file(path);
        ASSERT_TRUE(parsed) << path;
        auto table = std::move(parsed).table();
        for (const auto& [key, value] : table) {
            // toml::v3::key 需物化为 std::string（str() 返回 string_view）
            const std::string key_name(key.str());
            if (!is_server_section(key_name)) {
                continue;
            }
            const auto* section = value.as_table();
            ASSERT_TRUE(section != nullptr) << path << " section " << key_name;
            std::string err;
            EXPECT_TRUE(validate_server_section(*section, &err))
                << path << " [" << key_name << "]: " << err;
            ++checked;
        }
    }
    EXPECT_GT(checked, 0);
}

TEST(config_schema, every_model_section_passes_contract) {
    const auto files = collect_files("conf/model", ".toml");
    ASSERT_FALSE(files.empty());
    for (const auto& path : files) {
        auto parsed = toml::parse_file(path);
        ASSERT_TRUE(parsed) << path;
        auto table = std::move(parsed).table();
        for (const auto& [key, value] : table) {
            const auto* section = value.as_table();
            if (section == nullptr) {
                continue;
            }
            std::string err;
            EXPECT_TRUE(validate_model_config_section(*section, &err))
                << path << " [" << key << "]: " << err;
        }
    }
}

TEST(config_schema, missing_required_key_fails) {
    auto table = parse_text("[X]\nhost=\"localhost\"\nport=9000\nserver_uri=\"/x\"\n");
    std::string err;
    EXPECT_FALSE(validate_server_section(table, &err));
    EXPECT_NE(err.find("worker_nums"), std::string::npos) << err;
}

TEST(config_schema, typo_key_fails_with_suggestion) {
    auto table = parse_text(
        "[X]\nhost=\"localhost\"\nport=9000\nserver_uri=\"/x\"\nworker_nums=4\nworker_numsx=4\n");
    std::string err;
    EXPECT_FALSE(validate_server_section(table, &err));
    EXPECT_NE(err.find("worker_nums"), std::string::npos) << err;
}

TEST(config_schema, wrong_type_fails) {
    auto table = parse_text(
        "[X]\nhost=\"localhost\"\nport=\"abc\"\nserver_uri=\"/x\"\nworker_nums=1\n");
    std::string err;
    EXPECT_FALSE(validate_server_section(table, &err));
    EXPECT_NE(err.find("port"), std::string::npos) << err;
}

TEST(config_schema, unrelated_unknown_key_warns_but_passes) {
    auto table = parse_text(
        "[X]\nhost=\"localhost\"\nport=9000\nserver_uri=\"/x\"\nworker_nums=1\n"
        "fake_delay_ms=100\n");
    std::string err;
    std::vector<std::string> warnings;
    EXPECT_TRUE(validate_server_section(table, &err, &warnings));
    EXPECT_EQ(warnings.size(), 1u);
}

TEST(config_schema, bad_compute_backend_fails) {
    auto table = parse_text("[M]\nmodel_file_path=\"x.mnn\"\ncompute_backend=\"tensorrt\"\n");
    std::string err;
    EXPECT_FALSE(validate_model_config_section(table, &err));
    EXPECT_NE(err.find("compute_backend"), std::string::npos) << err;
}

TEST(config_schema, empty_model_file_path_fails) {
    auto table = parse_text("[M]\nmodel_file_path=\"\"\n");
    std::string err;
    EXPECT_FALSE(validate_model_config_section(table, &err));
    EXPECT_NE(err.find("model_file_path"), std::string::npos) << err;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
