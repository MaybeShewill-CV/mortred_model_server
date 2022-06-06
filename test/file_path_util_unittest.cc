/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: file_path_util_unittest.cpp
* Date: 22-6-6
************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/file_path_util.h"

using morted::common::FilePathUtil;

TEST(base64_unnittest, encode) {

    std::string exist_file_path = "./demo_data/model_test_input/image_ocr/db_text/test.jpg";
    std::string not_exist_file_path = "./demo_data/model_test_input/image_ocr/db_text/not_exist_test.jpg";

    std::string exist_dir_path = "./demo_data/model_test_input/image_ocr/db_text";
    std::string not_exist_dir_path = "./demo_data/model_test_input/image_ocr/db_text_not_exist";

    std::string concat_a = "./demo_data/model_test_input/image_ocr";
    std::string concat_b = "db_text/test.jpg";
    std::string concat_result = "./demo_data/model_test_input/image_ocr/db_text/test.jpg";

    EXPECT_EQ(FilePathUtil::is_file(exist_file_path), true);
    EXPECT_EQ(FilePathUtil::is_file(not_exist_file_path), false);
    EXPECT_EQ(FilePathUtil::is_dir(exist_dir_path), true);
    EXPECT_EQ(FilePathUtil::is_dir(not_exist_dir_path), false);
    EXPECT_STREQ(FilePathUtil::get_file_name(exist_file_path).c_str(), "test.jpg");
    EXPECT_STREQ(FilePathUtil::get_file_suffix(exist_file_path).c_str(), "jpg");
    EXPECT_STREQ(FilePathUtil::concat_path(concat_a, concat_b).c_str(), concat_result.c_str());
}