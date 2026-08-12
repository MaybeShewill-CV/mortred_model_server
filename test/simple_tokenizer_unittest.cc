/************************************************
 * Author: Codex
 * File: simple_tokenizer_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/status_code.h"
#include "models/clip/simple_tokenizer.h"

using jinq::common::StatusCode;
using jinq::models::clip::SimpleTokenizer;

static toml::value build_tokenizer_cfg() {
    toml::value cfg;
    cfg["TOKENIZER"]["vocab_file_path"] = "test/testdata/bpe_simple_vocab_16e6.txt";
    return cfg;
}

TEST(simple_tokenizer, init_with_missing_vocab_fails) {
    SimpleTokenizer tokenizer;
    toml::value cfg;
    cfg["TOKENIZER"]["vocab_file_path"] = "no_such_vocab_file.txt";
    EXPECT_EQ(tokenizer.init(cfg), StatusCode::MODEL_INIT_FAILED);
    EXPECT_FALSE(tokenizer.is_successfully_initialized());
}

TEST(simple_tokenizer, tokenize_is_deterministic_and_nonempty) {
    SimpleTokenizer tokenizer;
    auto cfg = build_tokenizer_cfg();
    ASSERT_EQ(tokenizer.init(cfg), StatusCode::OK);
    ASSERT_TRUE(tokenizer.is_successfully_initialized());

    std::vector<int> tokens;
    auto status = tokenizer.tokenize("hello world", tokens);
    EXPECT_EQ(status, StatusCode::OK);
    EXPECT_FALSE(tokens.empty());

    std::vector<int> tokens_again;
    tokenizer.tokenize("hello world", tokens_again);
    ASSERT_EQ(tokens.size(), tokens_again.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], tokens_again[i]);
    }
}
