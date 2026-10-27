/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: model_cli_unittest.cc
* Date: 26-10-27
************************************************/

// The shared `--model` / `--list` / `--version` parser for the unified
// entrypoints. Family flags (--batch, --loops, diffusion knobs) must survive in
// `rest` untouched, which is why unknown flags are NOT rejected.

#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "apps/cli_flags.h"

using mortred::apps::ModelCli;
using mortred::apps::parse_model_cli;

namespace {

bool parse(std::initializer_list<const char*> raw, ModelCli* out, std::string* err) {
    std::vector<std::string> owned(raw.begin(), raw.end());
    std::vector<char*> argv;
    argv.reserve(owned.size());
    for (auto& arg : owned) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    return parse_model_cli(static_cast<int>(argv.size()), argv.data(), out, err);
}

} // namespace

TEST(model_cli, parses_model_and_config_path) {
    ModelCli cli;
    std::string err;
    ASSERT_TRUE(parse({"server", "--model", "yolov8s", "conf/x.toml"}, &cli, &err)) << err;
    EXPECT_EQ(cli.model, "yolov8s");
    EXPECT_EQ(cli.rest_argc(), 2);  // argv[0] + config path
}

TEST(model_cli, accepts_equals_form) {
    ModelCli cli;
    std::string err;
    ASSERT_TRUE(parse({"server", "--model=hrnet", "conf/x.toml"}, &cli, &err)) << err;
    EXPECT_EQ(cli.model, "hrnet");
}

TEST(model_cli, double_dash_hands_everything_to_the_driver) {
    ModelCli cli;
    std::string err;
    ASSERT_TRUE(parse({"server", "--model", "yolov8s", "conf/x.toml", "--", "--batch", "8"}, &cli, &err))
        << err;
    ASSERT_EQ(cli.rest_argc(), 4);
    EXPECT_STREQ(cli.rest[2], "--batch");
    EXPECT_STREQ(cli.rest[3], "8");
}

TEST(model_cli, rejects_conflicting_model_flags) {
    ModelCli cli;
    std::string err;
    EXPECT_FALSE(parse({"server", "--model", "a", "--model", "b"}, &cli, &err));
    EXPECT_NE(err.find("twice"), std::string::npos) << err;
}

TEST(model_cli, rejects_empty_model_value) {
    ModelCli cli;
    std::string err;
    EXPECT_FALSE(parse({"server", "--model="}, &cli, &err));
    EXPECT_NE(err.find("non-empty"), std::string::npos) << err;
}

TEST(model_cli, recognises_version_and_help) {
    ModelCli cli;
    std::string err;
    ASSERT_TRUE(parse({"server", "--version"}, &cli, &err)) << err;
    EXPECT_TRUE(cli.version);
    ASSERT_TRUE(parse({"server", "-h"}, &cli, &err)) << err;
    EXPECT_TRUE(cli.help);
}

TEST(model_cli, falls_back_to_the_environment) {
    ::setenv("MORTRED_MODEL", "dbnet", 1);
    ModelCli cli;
    std::string err;
    ASSERT_TRUE(parse({"server", "conf/x.toml"}, &cli, &err)) << err;
    EXPECT_EQ(cli.model, "dbnet");
    ::unsetenv("MORTRED_MODEL");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
