/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: do_work_lifetime_unittest.cc
* Date: 26-9-7
************************************************/

// Regression: do_work must finish metrics/EWMA before enqueue, because the
// destructor treats "worker is home" as permission to destroy those members.
// Timeout-equivalent: start ~BaseAiServerImpl while do_work still holds the
// worker; the drain unblocks at enqueue and would UAF if metrics ran after.

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <rapidjson/document.h>

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/io/common_input.h"
#include "server/base_server_impl.h"
#include "server/inference_task.h"

using jinq::common::StatusCode;
using jinq::models::BaseAiModel;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::byte_source;
using jinq::server::BaseAiServerImpl;
using jinq::server::InferenceResult;
using jinq::server::InferenceTask;

namespace {

struct TestOutput {
    int value = 0;
};

class SlowModel : public BaseAiModel<base64_input, TestOutput> {
public:
    explicit SlowModel(int delay_ms) : _m_delay_ms(delay_ms) {}

    StatusCode init(const toml::table&) override {
        _m_initialized = true;
        return StatusCode::OK;
    }

    StatusCode run_impl(const base64_input&, TestOutput& out) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(_m_delay_ms));
        out.value = 1;
        return StatusCode::OK;
    }

    bool is_successfully_initialized() const override {
        return _m_initialized;
    }

private:
    int _m_delay_ms = 0;
    bool _m_initialized = false;
};

class LifetimeServer : public BaseAiServerImpl<std::unique_ptr<SlowModel>, TestOutput> {
public:
    StatusCode init(const toml::table&) override {
        return StatusCode::SERVER_INIT_FAILED;
    }

    StatusCode init_minimal(int delay_ms) {
        auto worker = std::make_unique<SlowModel>(delay_ms);
        toml::table empty;
        if (worker->init(empty) != StatusCode::OK) {
            return StatusCode::SERVER_INIT_FAILED;
        }
        _m_working_queue.enqueue(std::move(worker));
        _m_worker_nums = 1;
        _m_successfully_initialized = true;
        return StatusCode::OK;
    }

    void call_do_work(InferenceTask* req, InferenceResult* result) {
        do_work(req, result);
    }

    size_t queue_approx() const {
        return _m_working_queue.size_approx();
    }

protected:
    void fill_response_data(rapidjson::Document::AllocatorType& allocator,
                            rapidjson::Document& data,
                            const TestOutput& model_output,
                            const jinq::server::OutputOptions&) override {
        data.SetObject();
        data.AddMember("value", model_output.value, allocator);
    }
};

bool wait_for(const std::function<bool()>& pred, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

}  // namespace

TEST(do_work_lifetime, destructor_concurrent_with_in_flight_do_work) {
    auto impl = std::make_unique<LifetimeServer>();
    ASSERT_EQ(impl->init_minimal(200), StatusCode::OK);
    ASSERT_EQ(impl->queue_approx(), 1u);

    InferenceTask req;
    req.task_id = "lifetime";
    byte_source item;
    item.data = "x";
    req.items.push_back(std::move(item));
    InferenceResult<TestOutput> result;

    LifetimeServer* raw = impl.get();
    std::thread work([raw, &req, &result]() {
        raw->call_do_work(&req, &result);
    });

    ASSERT_TRUE(wait_for([&]() { return raw->queue_approx() == 0; }, 2000))
        << "do_work never checked out the worker";

    std::thread dtor([&impl]() {
        impl.reset();
    });

    work.join();
    dtor.join();

    EXPECT_EQ(result.model_run_status, StatusCode::OK);
    EXPECT_EQ(result.item_outputs.size(), 1u);
    EXPECT_EQ(result.item_outputs[0].value, 1);
    EXPECT_GT(result.worker_run_time_consuming, 0.0);
}
