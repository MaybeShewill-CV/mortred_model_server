/************************************************
 * Author: Codex
 * File: status_code_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/response_envelope.h"
#include "common/status_code.h"

using mortred::common::status_code_to_str;
using mortred::common::StatusCode;
using mortred::common::to_underlying;

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
    EXPECT_EQ(to_underlying(StatusCode::MODEL_INPUT_SIZE_REJECTED), 8);
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

TEST(status_code, set_status_pairs_number_and_wire_string) {
    // status and status_str are one value on the wire; the helper exists so a
    // reply site cannot update one and forget the other.
    mortred::common::UnifiedResponse resp;
    mortred::common::set_status(&resp, StatusCode::MODEL_INPUT_SIZE_REJECTED);
    EXPECT_EQ(resp.status, to_underlying(StatusCode::MODEL_INPUT_SIZE_REJECTED));
    EXPECT_EQ(resp.status_str, status_code_to_str(StatusCode::MODEL_INPUT_SIZE_REJECTED));
    EXPECT_EQ(resp.status_str.find("256x256"), std::string::npos)
        << "the shared table must not hardcode one model's input size";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
