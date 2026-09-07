/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_model_nvi_unittest.cc
* Date: 26-9-7
************************************************/

// NVI fence: run_impl throwing must become StatusCode, not terminate.

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "models/base_model.h"

using jinq::common::StatusCode;
using jinq::models::BaseAiModel;

namespace {

class ThrowModel : public BaseAiModel<int, int> {
public:
    StatusCode init(const toml::table&) override {
        _m_ok = true;
        return StatusCode::OK;
    }

    bool is_successfully_initialized() const override {
        return _m_ok;
    }

protected:
    StatusCode run_impl(const int&, int&) override {
        throw std::runtime_error("synthetic run_impl failure");
    }

private:
    bool _m_ok = false;
};

}  // namespace

TEST(base_model_nvi, run_maps_throw_to_session_failed) {
    ThrowModel model;
    ASSERT_EQ(model.init(toml::table{}), StatusCode::OK);
    int out = 0;
    EXPECT_EQ(model.run(1, out), StatusCode::MODEL_RUN_SESSION_FAILED);
}

TEST(base_model_nvi, run_batch_maps_throw_to_session_failed) {
    ThrowModel model;
    ASSERT_EQ(model.init(toml::table{}), StatusCode::OK);
    std::vector<int> in{1};
    std::vector<int> out;
    std::vector<StatusCode> item_status;
    EXPECT_EQ(model.run_batch(in, out, item_status), StatusCode::MODEL_RUN_SESSION_FAILED);
    ASSERT_EQ(item_status.size(), 1u);
    EXPECT_EQ(item_status[0], StatusCode::MODEL_RUN_SESSION_FAILED);
}
