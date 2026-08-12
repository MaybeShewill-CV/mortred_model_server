/************************************************
 * Author: Codex
 * File: llm_datatype_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <gtest/gtest.h>

#include "models/llm/llm_datatype.hpp"

using jinq::models::llm::ChatMessage;
using jinq::models::llm::Dialog;

TEST(llm_datatype, empty_and_push_back) {
    Dialog dialog;
    EXPECT_TRUE(dialog.empty());

    dialog.push_back(ChatMessage{"user", "hello"});
    EXPECT_FALSE(dialog.empty());
    ASSERT_EQ(dialog.size(), 1u);
    EXPECT_EQ(dialog[0].role, "user");
    EXPECT_EQ(dialog[0].content, "hello");
}

TEST(llm_datatype, constructors_and_merge) {
    Dialog user_dialog("user", "hello");
    ASSERT_EQ(user_dialog.size(), 1u);

    Dialog assistant_dialog(ChatMessage{"assistant", "world"});
    user_dialog += assistant_dialog;
    ASSERT_EQ(user_dialog.size(), 2u);
    EXPECT_EQ(user_dialog[1].role, "assistant");

    Dialog system_dialog("system", "start");
    auto merged = system_dialog + user_dialog;
    ASSERT_EQ(merged.size(), 3u);
    EXPECT_EQ(merged[0].role, "system");
    EXPECT_EQ(merged[2].content, "world");
}

TEST(llm_datatype, clean_cache) {
    Dialog dialog("user", "x");
    dialog.push_back(ChatMessage{"assistant", "y"});
    EXPECT_EQ(dialog.size(), 2u);

    dialog.clean_cache();
    EXPECT_TRUE(dialog.empty());
    EXPECT_EQ(dialog.size(), 0u);
}
