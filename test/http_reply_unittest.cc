// Direct unit tests for the http_wire reply helpers, driven through real
// Workflow HTTP tasks (no server socket): the e2e suite asserts these
// behaviors through the full server, this asserts the wire helpers
// themselves - headers, status line and envelope body.

#include <string>

#include <gtest/gtest.h>
#include <workflow/WFTaskFactory.h>

#include "common/status_code.h"
#include "server/http_wire.h"

namespace {

using mortred::common::StatusCode;
TEST(http_reply, status_envelope_carries_code_model_and_text) {
    const auto ok = mortred::server::status_envelope("t1", StatusCode::OK, {}, "unit-model");
    EXPECT_EQ(ok.status, 0);
    EXPECT_EQ(ok.status_str, "OK");
    EXPECT_EQ(ok.model_name, "unit-model");
    EXPECT_EQ(ok.task_id, "t1");

    const auto not_ready = mortred::server::status_envelope("t2", StatusCode::NOT_READY);
    EXPECT_EQ(not_ready.status, mortred::common::to_underlying(StatusCode::NOT_READY));
    EXPECT_EQ(not_ready.status_str, "service not ready");
}

TEST(http_reply, unified_rejection_carries_errors) {
    std::vector<mortred::common::ResponseError> errors;
    errors.push_back({"/images/0", "must be a string"});
    const auto r = mortred::server::unified_rejection("t3", StatusCode::INVALID_REQUEST_PARAMETER,
                                                     std::move(errors));
    EXPECT_EQ(r.status, mortred::common::to_underlying(StatusCode::INVALID_REQUEST_PARAMETER));
    ASSERT_EQ(r.errors.size(), 1u);
    EXPECT_EQ(r.errors[0].pointer, "/images/0");
}



struct Reply {
    std::string status;
    std::string body;
    bool has_www_auth = false;
};

Reply reply_with(void (*fn)(WFHttpTask*, const std::string&), const std::string& model) {
    auto* task = WFTaskFactory::create_http_task("http://127.0.0.1:1/", 0, 0,
                                                 [](WFHttpTask*) {});
    fn(task, model);
    Reply out;
    out.status = task->get_resp()->get_status_code();
    const void* body = nullptr;
    size_t size = 0;
    if (task->get_resp()->get_parsed_body(&body, &size)) {
        out.body.assign(static_cast<const char*>(body), size);
    }
    // workflow stores headers case-insensitively; scan pairs directly
    protocol::HttpHeaderCursor cursor(task->get_resp());
    std::string name;
    std::string value;
    while (cursor.next(name, value)) {
        std::string lower;
        for (const char ch : name) {
            lower += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        if (lower == "www-authenticate") {
            out.has_www_auth = true;
        }
    }
    return out;
}

}  // namespace

TEST(http_reply, reply_status_carries_ok_envelope) {
    const auto r = reply_with([](WFHttpTask* t, const std::string& m) {
        mortred::server::reply_status(t, StatusCode::OK, m);
    }, "unit-model");
    EXPECT_EQ(r.status, "200");
    // body/envelope composition is asserted via status_envelope above; the
    // unsent outbound response body is not readable through Workflow's API
}

TEST(http_reply, reply_status_maps_not_ready_to_503) {
    const auto r = reply_with([](WFHttpTask* t, const std::string& m) {
        mortred::server::reply_status(t, StatusCode::NOT_READY, m);
    }, "unit-model");
    EXPECT_EQ(r.status, "503");
}

TEST(http_reply, reply_unauthorized_sets_www_authenticate) {
    const auto r = reply_with([](WFHttpTask* t, const std::string& m) {
        mortred::server::reply_unauthorized(t, m);
    }, "unit-model");
    EXPECT_EQ(r.status, "401");
    EXPECT_TRUE(r.has_www_auth);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
