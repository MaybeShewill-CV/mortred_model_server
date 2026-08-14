/************************************************
 * Author: Codex
 * File: file_path_util_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/file_path_util.h"

using jinq::common::FilePathUtil;

TEST(file_path_util, exist_and_name) {
    std::string exist_file_path = "demo_data/model_test_input/ocr/railway_ticket.png";
    std::string not_exist_file_path = "demo_data/model_test_input/ocr/not_exist_test.png";

    std::string exist_dir_path = "demo_data/model_test_input/ocr";
    std::string not_exist_dir_path = "demo_data/model_test_input/ocr_not_exist";

    EXPECT_EQ(FilePathUtil::is_file_exist(exist_file_path), true);
    EXPECT_EQ(FilePathUtil::is_file_exist(not_exist_file_path), false);
    // a directory must not be reported as a regular file
    EXPECT_EQ(FilePathUtil::is_file_exist(exist_dir_path), false);
    EXPECT_EQ(FilePathUtil::is_dir_exist(exist_dir_path), true);
    EXPECT_EQ(FilePathUtil::is_dir_exist(not_exist_dir_path), false);

    EXPECT_STREQ(FilePathUtil::get_file_name(exist_file_path).c_str(), "railway_ticket.png");
}

TEST(file_path_util, concat_path) {
    EXPECT_STREQ(FilePathUtil::concat_path("demo_data/model_test_input/ocr", "railway_ticket.png").c_str(),
                 "demo_data/model_test_input/ocr/railway_ticket.png");
    // trailing slash on lhs must not produce a double separator
    EXPECT_STREQ(FilePathUtil::concat_path("demo_data/model_test_input/ocr/", "railway_ticket.png").c_str(),
                 "demo_data/model_test_input/ocr/railway_ticket.png");
    // empty rhs falls back to lhs
    EXPECT_STREQ(FilePathUtil::concat_path("demo_data/model_test_input/ocr", "").c_str(),
                 "demo_data/model_test_input/ocr");
    // empty lhs must not be undefined behavior (regression)
    EXPECT_STREQ(FilePathUtil::concat_path("", "railway_ticket.png").c_str(),
                 "railway_ticket.png");
}
