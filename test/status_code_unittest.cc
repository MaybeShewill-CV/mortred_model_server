/************************************************
 * Author: Codex
 * File: status_code_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/status_code.h"

using jinq::common::status_code_to_str;
using jinq::common::StatusCode;
using jinq::common::to_underlying;

TEST(status_code, every_code_has_message) {
#define CHECK_MESSAGE(name, value, desc) EXPECT_EQ(status_code_to_str(StatusCode::name), std::string(desc));
    MORTRED_STATUS_CODE_LIST(CHECK_MESSAGE)
#undef CHECK_MESSAGE
}

TEST(status_code, wire_values_are_stable) {
    EXPECT_EQ(to_underlying(StatusCode::OK), 0);
    EXPECT_EQ(to_underlying(StatusCode::MODEL_INIT_FAILED), 1);
    EXPECT_EQ(to_underlying(StatusCode::MODEL_RUN_TIMEOUT), 4);
    EXPECT_EQ(to_underlying(StatusCode::MODEL_OUTPUT_CONTRACT_FAILED), 6);
    EXPECT_EQ(to_underlying(StatusCode::MODEL_NOT_IMPLEMENTED), 7);
    EXPECT_EQ(to_underlying(StatusCode::SERVER_INIT_FAILED), 11);
    EXPECT_EQ(to_underlying(StatusCode::JSON_DECODE_ERROR), 50);
    EXPECT_EQ(to_underlying(StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED), 92);
    EXPECT_EQ(to_underlying(StatusCode::INVALID_REQUEST_PARAMETER), 66);
    EXPECT_EQ(to_underlying(StatusCode::REQUEST_ITEM_LIMIT), 67);
    EXPECT_EQ(to_underlying(StatusCode::DEADLINE_EXCEEDED_PARTIAL), 68);
}

TEST(status_code, sample_messages) {
    EXPECT_EQ(status_code_to_str(StatusCode::OK), "OK");
    EXPECT_EQ(status_code_to_str(StatusCode::MODEL_RUN_TIMEOUT), "model run timeout");
    EXPECT_EQ(status_code_to_str(StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED), "convert onnx model to trt failed");
    EXPECT_EQ(status_code_to_str(StatusCode::INVALID_REQUEST_PARAMETER), "invalid request parameter");
    EXPECT_EQ(status_code_to_str(StatusCode::DEADLINE_EXCEEDED_PARTIAL), "deadline exceeded, partial results");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
