/************************************************
 * Author: Codex
 * File: status_code_unittest.cc
 * Date: 2026-08-11
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/status_code.h"

using jinq::common::StatusCode;
using jinq::common::error_code_to_str;

TEST(status_code, all_codes_have_readable_message) {
#define MORTRED_STATUS_CODE_ASSERT(name, value, desc) \
    EXPECT_NE(error_code_to_str(StatusCode::name), std::string("Unknown"));
    MORTRED_STATUS_CODE_LIST(MORTRED_STATUS_CODE_ASSERT)
#undef MORTRED_STATUS_CODE_ASSERT
}

TEST(status_code, sample_messages) {
    EXPECT_EQ(error_code_to_str(StatusCode::OK), std::string("OK"));
    EXPECT_EQ(error_code_to_str(StatusCode::MODEL_RUN_TIMEOUT),
              std::string("model run timeout"));
    EXPECT_EQ(error_code_to_str(StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED),
              std::string("convert onnx model to trt failed"));
    EXPECT_EQ(error_code_to_str(StatusCode::RAG_SEARCH_SEGMENT_CORPUS_FAILED),
              std::string("rag search segment corpus failed"));
    EXPECT_EQ(error_code_to_str(StatusCode::VLM_QWEN_PARSE_IMAGE_URL_FAILED),
              std::string("vlm qwen parse image url failed"));
}

TEST(status_code, ojbk_alias_maps_to_ok) {
    EXPECT_EQ(error_code_to_str(StatusCode::OJBK), std::string("OK"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
