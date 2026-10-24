// Unit tests for the dependency-free mini TOML parser (control plane).

#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "control/mini_toml.h"

using mortred::control::mini_toml::Doc;
using mortred::control::mini_toml::parse;
using mortred::control::mini_toml::to_bool;
using mortred::control::mini_toml::to_int;

TEST(mini_toml, parses_sections_quotes_and_inline_comments) {
    const auto doc = parse(
        "[a]\n"
        "name = \"value\"   # trailing comment\n"
        "count = 7\n"
        "[b.sub]\n"
        "flag = true\n");
    ASSERT_EQ(doc.size(), 2u);
    ASSERT_TRUE(doc.count("a"));
    EXPECT_EQ(doc.at("a").at("name"), "value");
    EXPECT_EQ(doc.at("a").at("count"), "7");
    ASSERT_TRUE(doc.count("b.sub"));
    EXPECT_EQ(doc.at("b.sub").at("flag"), "true");
}

TEST(mini_toml, reports_dropped_lines_instead_of_staying_silent) {
    std::vector<std::string> problems;
    const auto doc = parse(
        "orphan = 1\n"
        "[keys.a]\n"
        "no_equal_sign\n"
        "hash = \"aa\"\n", &problems);
    ASSERT_EQ(problems.size(), 2u);
    EXPECT_NE(problems[0].find("line 1:"), std::string::npos) << problems[0];
    EXPECT_NE(problems[0].find("outside any [section]"), std::string::npos);
    EXPECT_NE(problems[1].find("line 3:"), std::string::npos) << problems[1];
    EXPECT_NE(problems[1].find("no_equal_sign"), std::string::npos);
    // the well-formed line still lands
    EXPECT_EQ(doc.at("keys.a").at("hash"), "aa");
}

TEST(mini_toml, records_section_and_key_lines) {
    mortred::control::mini_toml::KeyLines key_lines;
    mortred::control::mini_toml::SectionLines section_lines;
    const auto doc = parse(
        "[a]\n"
        "x = 1\n"
        "\n"
        "[b]\n"
        "y = 2\n", nullptr, &key_lines, &section_lines);
    EXPECT_EQ(doc.size(), 2u);
    EXPECT_EQ(section_lines.at("a"), 1);
    EXPECT_EQ(section_lines.at("b"), 4);
    EXPECT_EQ(key_lines.at("a").at("x"), 2);
    EXPECT_EQ(key_lines.at("b").at("y"), 5);
}

TEST(mini_toml, to_int_and_to_bool_fall_back) {
    EXPECT_EQ(to_int("42", 0), 42);
    EXPECT_EQ(to_int("not-a-number", 7), 7);
    EXPECT_EQ(to_int("", 3), 3);
    EXPECT_TRUE(to_bool("true", false));
    EXPECT_TRUE(to_bool("1", false));
    EXPECT_FALSE(to_bool("false", true));
    EXPECT_FALSE(to_bool("0", true));
    EXPECT_TRUE(to_bool("garbage", true));
}

TEST(mini_toml, load_missing_file_is_false_and_bad_lines_go_to_stderr) {
    Doc doc;
    EXPECT_FALSE(mortred::control::mini_toml::load("/tmp/mini_toml_no_such_file_zz.toml", &doc));
    const std::string path = "/tmp/mini_toml_load_test.toml";
    {
        std::ofstream out(path);
        out << "[k]\nv = \"x\"\nbroken_line\n";
    }
    EXPECT_TRUE(mortred::control::mini_toml::load(path, &doc));
    EXPECT_EQ(doc.at("k").at("v"), "x");
    std::remove(path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
