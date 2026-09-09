/************************************************
 * Author: Codex
 * File: item_exec_unittest.cc
 * Date: 2026-09-09
 ************************************************/

#include <chrono>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <rapidjson/document.h>

#include "common/status_code.h"
#include "models/io/common_input.h"
#include "server/inference_task.h"
#include "server/item_exec.h"

using jinq::common::StatusCode;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::byte_source;
using jinq::server::InferenceResult;
using jinq::server::InferenceTask;
using jinq::server::aggregate_item_statuses;
using jinq::server::inference_result_to_unified;
using jinq::server::make_model_input;
using jinq::server::run_items;

namespace {

struct FakeOutput {
    int value = 0;
};

struct FakeModel {
    using input_type = base64_input;
    StatusCode run(const base64_input& in, FakeOutput& out) {
        if (in.input_image_content == "fail") {
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        out.value = static_cast<int>(in.input_image_content.size());
        return StatusCode::OK;
    }
};

using FakeWorker = std::unique_ptr<FakeModel>;

byte_source text_item(std::string payload) {
    byte_source item;
    item.origin = byte_source::origin_kind::base64_text;
    item.data = std::move(payload);
    return item;
}

}  // namespace

TEST(item_exec, make_model_input_copies_base64_payload) {
    auto input = make_model_input<base64_input>(text_item("abc"), nullptr);
    EXPECT_EQ(input.input_image_content, "abc");
}

TEST(item_exec, run_items_ok_and_aggregate) {
    FakeWorker worker = std::make_unique<FakeModel>();
    InferenceTask req;
    req.items.push_back(text_item("xy"));
    InferenceResult<FakeOutput> result;
    run_items(worker, req, &result);
    EXPECT_EQ(result.model_run_status, StatusCode::OK);
    ASSERT_EQ(result.item_outputs.size(), 1u);
    EXPECT_EQ(result.item_outputs[0].value, 2);
    EXPECT_FALSE(result.partial);
}

TEST(item_exec, mixed_timeout_is_partial) {
    InferenceResult<FakeOutput> result;
    result.item_status = {StatusCode::OK, StatusCode::MODEL_RUN_TIMEOUT};
    aggregate_item_statuses(&result);
    EXPECT_EQ(result.model_run_status, StatusCode::DEADLINE_EXCEEDED_PARTIAL);
    EXPECT_TRUE(result.partial);
}

TEST(item_exec, deadline_skips_remaining_items) {
    FakeWorker worker = std::make_unique<FakeModel>();
    InferenceTask req;
    req.items.push_back(text_item("a"));
    req.items.push_back(text_item("bb"));
    req.deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    InferenceResult<FakeOutput> result;
    run_items(worker, req, &result);
    EXPECT_EQ(result.model_run_status, StatusCode::MODEL_RUN_TIMEOUT);
    ASSERT_EQ(result.item_status.size(), 2u);
    EXPECT_EQ(result.item_status[0], StatusCode::MODEL_RUN_TIMEOUT);
    EXPECT_EQ(result.item_status[1], StatusCode::MODEL_RUN_TIMEOUT);
}

TEST(item_exec, unified_fill_only_ok_items) {
    InferenceResult<FakeOutput> result;
    result.model_run_status = StatusCode::OK;
    result.item_status = {StatusCode::OK};
    result.item_outputs.push_back(FakeOutput{3});
    auto unified = inference_result_to_unified(
        "t1", "TEST", result,
        [](rapidjson::Document::AllocatorType& allocator, rapidjson::Document& data,
           const FakeOutput& output, const jinq::server::OutputOptions&) {
            data.SetObject();
            data.AddMember("value", output.value, allocator);
        });
    EXPECT_EQ(unified.task_id, "t1");
    ASSERT_EQ(unified.results.size(), 1u);
    EXPECT_TRUE(unified.results[0].data.HasMember("value"));
    EXPECT_EQ(unified.results[0].data["value"].GetInt(), 3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
