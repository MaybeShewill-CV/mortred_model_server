/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: restart_policy_unittest.cc
* Date: 26-8-22
************************************************/

#include <gtest/gtest.h>

#include "control/restart_policy.h"

using mortred::control::RestartConstants;
using mortred::control::RestartEngine;
using mortred::control::RestartPolicyKind;
using mortred::control::SupervisedState;
using mortred::control::parse_restart_policy;

TEST(restart_policy, parse_valid_and_invalid) {
    RestartPolicyKind kind = RestartPolicyKind::kNo;
    ASSERT_TRUE(parse_restart_policy("no", &kind));
    EXPECT_EQ(kind, RestartPolicyKind::kNo);
    ASSERT_TRUE(parse_restart_policy("on-failure", &kind));
    EXPECT_EQ(kind, RestartPolicyKind::kOnFailure);
    ASSERT_TRUE(parse_restart_policy("always", &kind));
    EXPECT_EQ(kind, RestartPolicyKind::kAlways);
    EXPECT_FALSE(parse_restart_policy("sometimes", &kind));
}

TEST(restart_policy, on_failure_restarts_with_exponential_backoff) {
    RestartEngine e(RestartPolicyKind::kOnFailure);
    e.note_started(0);
    e.note_ready(100);

    auto d1 = e.note_exit(200, false, false);
    ASSERT_TRUE(d1.restart);
    EXPECT_EQ(d1.delay_ms, RestartConstants::kBackoffBaseMs);
    EXPECT_EQ(e.state(), SupervisedState::kBackoff);

    e.note_started(1000);
    e.note_ready(1100);
    auto d2 = e.note_exit(1200, false, false);
    ASSERT_TRUE(d2.restart);
    EXPECT_EQ(d2.delay_ms, RestartConstants::kBackoffBaseMs * 2);
    EXPECT_EQ(e.restart_count(), 2);
}

TEST(restart_policy, clean_exit_is_not_restarted_on_failure) {
    RestartEngine e(RestartPolicyKind::kOnFailure);
    e.note_started(0);
    e.note_ready(10);
    const auto d = e.note_exit(20, true, false);
    EXPECT_FALSE(d.restart);
    EXPECT_EQ(e.state(), SupervisedState::kStopped);
}

TEST(restart_policy, always_restarts_even_clean_exit) {
    RestartEngine e(RestartPolicyKind::kAlways);
    e.note_started(0);
    e.note_ready(10);
    const auto d = e.note_exit(20, true, false);
    EXPECT_TRUE(d.restart);
}

TEST(restart_policy, no_policy_never_restarts) {
    RestartEngine e(RestartPolicyKind::kNo);
    e.note_started(0);
    e.note_ready(10);
    const auto d = e.note_exit(20, false, false);
    EXPECT_FALSE(d.restart);
    EXPECT_EQ(e.state(), SupervisedState::kStopped);
}

TEST(restart_policy, expected_stop_resets_everything) {
    RestartEngine e(RestartPolicyKind::kOnFailure);
    e.note_started(0);
    e.note_ready(10);
    ASSERT_TRUE(e.note_exit(20, false, false).restart);
    e.note_started(600);
    e.note_ready(700);
    const auto d = e.note_exit(800, false, true);
    EXPECT_FALSE(d.restart);
    EXPECT_EQ(e.state(), SupervisedState::kStopped);
    EXPECT_EQ(e.restart_count(), 1);
}

TEST(restart_policy, stable_run_resets_backoff_and_window) {
    RestartEngine e(RestartPolicyKind::kOnFailure);
    e.note_started(0);
    e.note_ready(10);
    ASSERT_TRUE(e.note_exit(20, false, false).restart);  // delay 500

    e.note_started(1000);
    e.note_ready(1100);
    // ran ready for > 5 minutes: heal
    const auto d = e.note_exit(1100 + RestartConstants::kStableResetMs + 1, false, false);
    ASSERT_TRUE(d.restart);
    EXPECT_EQ(d.delay_ms, RestartConstants::kBackoffBaseMs);
}

TEST(restart_policy, crash_loop_gives_up_and_requires_manual_start) {
    RestartEngine e(RestartPolicyKind::kOnFailure);
    int64_t t = 0;
    for (int i = 0; i < RestartConstants::kMaxRestartsInWindow; ++i) {
        e.note_started(t);
        e.note_ready(t + 1);
        ASSERT_TRUE(e.note_exit(t + 2, false, false).restart) << "restart " << i;
        t += 100;  // all inside one 60s window
    }
    e.note_started(t);
    e.note_ready(t + 1);
    const auto d = e.note_exit(t + 2, false, false);
    EXPECT_FALSE(d.restart);
    EXPECT_TRUE(d.gave_up);
    EXPECT_EQ(e.state(), SupervisedState::kFailed);
}

TEST(restart_policy, permanent_failure_does_not_restart) {
    RestartEngine e(RestartPolicyKind::kOnFailure);
    e.note_started(0);
    e.note_permanent_failure();
    EXPECT_EQ(e.state(), SupervisedState::kFailed);
    const auto d = e.note_exit(10, false, false);
    EXPECT_FALSE(d.restart);
}

TEST(restart_policy, cancel_returns_to_stopped) {
    RestartEngine e(RestartPolicyKind::kOnFailure);
    e.note_started(0);
    e.note_ready(10);
    ASSERT_TRUE(e.note_exit(20, false, false).restart);
    e.note_cancel();
    EXPECT_EQ(e.state(), SupervisedState::kStopped);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
