/************************************************
 * Author: Codex
 * File: http_contract_test.cc
 * Date: 2026-08-26
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/http_response.h"
#include "common/status_code.h"
#include "server/http_status.h"

using jinq::common::build_response_body;
using jinq::common::HttpResponse;
using jinq::common::StatusCode;
using jinq::server::http_status_of;

TEST(http_contract, success_maps_to_200) { EXPECT_EQ(http_status_of(StatusCode::OK), 200); }

TEST(http_contract, client_errors_map_to_4xx) {
    EXPECT_EQ(http_status_of(StatusCode::JSON_DECODE_ERROR), 400);
    EXPECT_EQ(http_status_of(StatusCode::MODEL_EMPTY_INPUT_IMAGE), 400);
    EXPECT_EQ(http_status_of(StatusCode::UNSUPPORTED_MEDIA_TYPE), 415);
    EXPECT_EQ(http_status_of(StatusCode::REQUEST_ENTITY_TOO_LARGE), 413);
    EXPECT_EQ(http_status_of(StatusCode::METHOD_NOT_ALLOWED), 405);
    EXPECT_EQ(http_status_of(StatusCode::NOT_FOUND), 404);
    EXPECT_EQ(http_status_of(StatusCode::NOT_READY), 503);
    EXPECT_EQ(http_status_of(StatusCode::UNAUTHORIZED), 401);
    EXPECT_EQ(http_status_of(StatusCode::RATE_LIMITED), 429);
}

TEST(http_contract, server_errors_map_to_5xx) {
    EXPECT_EQ(http_status_of(StatusCode::MODEL_INIT_FAILED), 500);
    EXPECT_EQ(http_status_of(StatusCode::MODEL_OUTPUT_CONTRACT_FAILED), 500);
    EXPECT_EQ(http_status_of(StatusCode::MODEL_RUN_TIMEOUT), 504);
}

TEST(http_contract, response_body_contains_envelope) {
    HttpResponse resp;
    resp.req_id = "abc";
    resp.code = 0;
    resp.msg = "success";
    auto body = build_response_body(resp);
    EXPECT_NE(body.find("\"req_id\":\"abc\""), std::string::npos);
    EXPECT_NE(body.find("\"code\":0"), std::string::npos);
    EXPECT_NE(body.find("\"msg\":\"success\""), std::string::npos);
    EXPECT_NE(body.find("\"data\":null"), std::string::npos);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
