/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: file_path_util_unittest.cpp
* Date: 22-6-6
************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/file_path_util.h"

using jinq::common::FilePathUtil;

TEST(base64_unnittest, encode) {

    std::string exist_file_path = "demo_data/model_test_input/ocr/railway_ticket.png";
    std::string not_exist_file_path = "demo_data/model_test_input/ocr/not_exist_test.png";

    std::string exist_dir_path = "demo_data/model_test_input/ocr";
    std::string not_exist_dir_path = "demo_data/model_test_input/ocr_not_exist";

    std::string concat_a = "demo_data/model_test_input/ocr";
    std::string concat_b = "railway_ticket.png";
    std::string concat_result = "demo_data/model_test_input/ocr/railway_ticket.png";

    EXPECT_EQ(FilePathUtil::is_file_exist(exist_file_path), true);
    EXPECT_EQ(FilePathUtil::is_file_exist(not_exist_file_path), false);
    EXPECT_EQ(FilePathUtil::is_dir_exist(exist_dir_path), true);
    EXPECT_EQ(FilePathUtil::is_dir_exist(not_exist_dir_path), false);
    EXPECT_STREQ(FilePathUtil::get_file_name(exist_file_path).c_str(), "railway_ticket.png");
    EXPECT_STREQ(FilePathUtil::get_file_suffix(exist_file_path).c_str(), "png");
    EXPECT_STREQ(FilePathUtil::concat_path(concat_a, concat_b).c_str(), concat_result.c_str());
}
