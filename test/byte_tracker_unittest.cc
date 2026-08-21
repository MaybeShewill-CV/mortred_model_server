/************************************************
 * Author: Codex
 * File: byte_tracker_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <vector>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "common/status_code.h"
#include "models/mot/byte_tracker/byte_tracker.h"
#include "models/mot/byte_tracker/kalman_filter.h"
#include "models/mot/byte_tracker/lapjv.h"
#include "models/mot/byte_tracker/strack.h"

using jinq::common::StatusCode;
using jinq::models::mot::byte_tracker::ByteTracker;
using jinq::models::mot::byte_tracker::DETECTBOX;
using jinq::models::mot::byte_tracker::KalmanFilter;
using jinq::models::mot::byte_tracker::STrack;
using jinq::models::mot::byte_tracker::TrackState;
using jinq::models::mot::byte_tracker::cost_t;
using jinq::models::mot::byte_tracker::int_t;
using jinq::models::mot::byte_tracker::lapjv_internal;
using jinq::models::io_define::object_detection::bbox;

TEST(kalman_filter, initiate_shapes_and_values) {
    KalmanFilter kf;
    DETECTBOX measurement;
    measurement << 10, 20, 1.5f, 30;

    auto mc = kf.initiate(measurement);
    EXPECT_EQ(mc.first.rows(), 1);
    EXPECT_EQ(mc.first.cols(), 8);
    EXPECT_FLOAT_EQ(mc.first(0), 10);
    EXPECT_FLOAT_EQ(mc.first(1), 20);
    EXPECT_FLOAT_EQ(mc.first(2), 1.5f);
    EXPECT_FLOAT_EQ(mc.first(3), 30);
    EXPECT_FLOAT_EQ(mc.first(4), 0);
    EXPECT_FLOAT_EQ(mc.first(7), 0);

    EXPECT_EQ(mc.second.rows(), 8);
    EXPECT_EQ(mc.second.cols(), 8);
    EXPECT_GT(mc.second(0, 0), 0);
    EXPECT_GT(mc.second(4, 4), 0);
}

TEST(kalman_filter, predict_increases_covariance) {
    KalmanFilter kf;
    DETECTBOX measurement;
    measurement << 10, 20, 1.5f, 30;
    auto mc = kf.initiate(measurement);

    auto trace_before = mc.second.trace();
    kf.predict(mc.first, mc.second);
    EXPECT_GT(mc.second.trace(), trace_before);
    EXPECT_FLOAT_EQ(mc.first(0), 10);
    EXPECT_FLOAT_EQ(mc.first(1), 20);
}

TEST(kalman_filter, update_shrinks_covariance) {
    KalmanFilter kf;
    DETECTBOX measurement;
    measurement << 10, 20, 1.5f, 30;
    auto mc = kf.initiate(measurement);
    kf.predict(mc.first, mc.second);

    DETECTBOX new_measurement;
    new_measurement << 11, 21, 1.5f, 30;
    auto updated = kf.update(mc.first, mc.second, new_measurement);

    EXPECT_EQ(updated.first.rows(), 1);
    EXPECT_EQ(updated.first.cols(), 8);
    EXPECT_EQ(updated.second.rows(), 8);
    EXPECT_LT(updated.second.trace(), mc.second.trace());
}

TEST(lapjv, optimal_assignment_2x2) {
    const unsigned int n = 2;
    cost_t row0[2] = {1, 2};
    cost_t row1[2] = {2, 1};
    cost_t* cost[2] = {row0, row1};
    int_t x[2] = {0, 0};
    int_t y[2] = {0, 0};

    auto ret = lapjv_internal(n, cost, x, y);
    EXPECT_EQ(ret, 0);

    // the assignment must be a valid permutation with the optimal total cost 2
    EXPECT_TRUE((x[0] == 0 && x[1] == 1) || (x[0] == 1 && x[1] == 0));
    EXPECT_DOUBLE_EQ(cost[0][x[0]] + cost[1][x[1]], 2.0);
}

TEST(lapjv, optimal_assignment_3x3) {
    const unsigned int n = 3;
    cost_t row0[3] = {1, 100, 100};
    cost_t row1[3] = {100, 2, 100};
    cost_t row2[3] = {100, 100, 3};
    cost_t* cost[3] = {row0, row1, row2};
    int_t x[3] = {0, 0, 0};
    int_t y[3] = {0, 0, 0};

    auto ret = lapjv_internal(n, cost, x, y);
    EXPECT_EQ(ret, 0);

    EXPECT_EQ(cost[0][x[0]] + cost[1][x[1]] + cost[2][x[2]], 6.0);
    int seen[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        ASSERT_GE(x[i], 0);
        ASSERT_LT(x[i], 3);
        seen[x[i]]++;
    }
    EXPECT_EQ(seen[0], 1);
    EXPECT_EQ(seen[1], 1);
    EXPECT_EQ(seen[2], 1);
}

TEST(strack, coordinate_conversions) {
    std::vector<float> tlbr = {0, 0, 10, 20};
    auto tlwh = STrack::tlbr_to_tlwh(tlbr);
    EXPECT_FLOAT_EQ(tlwh[0], 0);
    EXPECT_FLOAT_EQ(tlwh[1], 0);
    EXPECT_FLOAT_EQ(tlwh[2], 10);
    EXPECT_FLOAT_EQ(tlwh[3], 20);

    std::vector<float> tlwh_in = {10, 20, 30, 40};
    auto xyah = STrack::tlwh_to_xyah(tlwh_in);
    EXPECT_FLOAT_EQ(xyah[0], 25);   // x + w/2
    EXPECT_FLOAT_EQ(xyah[1], 40);   // y + h/2
    EXPECT_FLOAT_EQ(xyah[2], 0.75f); // w/h
    EXPECT_FLOAT_EQ(xyah[3], 40);
}

TEST(strack, lifecycle_and_id) {
    KalmanFilter kf;
    STrack track({10, 20, 30, 40}, 0.9f);

    EXPECT_EQ(track.state, TrackState::New);
    EXPECT_FALSE(track.is_activated);

    track.activate(kf, 1);
    EXPECT_EQ(track.state, TrackState::Tracked);
    EXPECT_TRUE(track.is_activated);
    EXPECT_GT(track.track_id, 0);
    EXPECT_EQ(track.frame_id, 1);
    int first_id = track.track_id;

    track.mark_lost();
    EXPECT_EQ(track.state, TrackState::Lost);
    track.mark_removed();
    EXPECT_EQ(track.state, TrackState::Removed);

    STrack another({1, 1, 2, 2}, 0.5f);
    another.activate(kf, 1);
    EXPECT_GT(another.track_id, first_id);
}

TEST(strack, update_and_reactivate) {
    KalmanFilter kf;
    STrack track({10, 20, 30, 40}, 0.9f);
    track.activate(kf, 1);

    STrack new_track({12, 22, 30, 40}, 0.95f);
    track.update(new_track, 2);
    EXPECT_EQ(track.state, TrackState::Tracked);
    EXPECT_EQ(track.frame_id, 2);
    EXPECT_EQ(track.tracklet_len, 1);

    STrack lost({10, 20, 30, 40}, 0.8f);
    lost.activate(kf, 1);
    lost.mark_lost();
    STrack detection({11, 21, 30, 40}, 0.85f);
    lost.re_activate(detection, 5);
    EXPECT_EQ(lost.state, TrackState::Tracked);
    EXPECT_EQ(lost.frame_id, 5);
}

static toml::table build_byte_track_cfg() {
    return std::move(toml::parse(
        "[BYTE_TRACK]\n"
        "tracker_thresh = 0.5\n"
        "tracker_high_thresh = 0.6\n"
        "tracker_match_thresh = 0.8\n"
        "frame_rate = 30\n"
        "track_buffer = 2\n"
        "tracked_cls_ids = [0]\n"
        "tracked_cls_names = [\"car\"]\n"))
        .table();
}

static bbox make_object(float x, float y, float w, float h, float score, int cls_id) {
    bbox obj;
    obj.bbox = cv::Rect2f(x, y, w, h);
    obj.score = score;
    obj.class_id = cls_id;
    obj.category = "car";
    return obj;
}

TEST(byte_tracker, init_requires_bytetrack_section) {
    ByteTracker tracker;
    toml::table cfg;
    EXPECT_EQ(tracker.init(cfg), StatusCode::MODEL_INIT_FAILED);
    EXPECT_FALSE(tracker.is_successfully_initialized());
}

TEST(byte_tracker, stable_id_across_frames_and_lost) {
    ByteTracker tracker;
    auto cfg = build_byte_track_cfg();
    ASSERT_EQ(tracker.init(cfg), StatusCode::OK);
    ASSERT_TRUE(tracker.is_successfully_initialized());

    auto obj = make_object(10, 10, 50, 50, 0.9f, 0);

    auto frame1 = tracker.update({obj});
    ASSERT_EQ(frame1.size(), 1u);
    EXPECT_GT(frame1[0].track_id, 0);
    EXPECT_EQ(frame1[0].state, TrackState::Tracked);
    int track_id = frame1[0].track_id;

    auto frame2 = tracker.update({obj});
    ASSERT_EQ(frame2.size(), 1u);
    EXPECT_EQ(frame2[0].track_id, track_id);
    EXPECT_EQ(frame2[0].state, TrackState::Tracked);

    // object disappears: the track enters lost, output is empty
    auto frame3 = tracker.update({});
    EXPECT_TRUE(frame3.empty());

    // more empty frames must not crash
    auto frame4 = tracker.update({});
    EXPECT_TRUE(frame4.empty());
}

TEST(byte_tracker, untracked_class_is_ignored) {
    ByteTracker tracker;
    auto cfg = build_byte_track_cfg();
    ASSERT_EQ(tracker.init(cfg), StatusCode::OK);

    // class_id=1 is not in tracked_cls_ids, so it must be filtered out
    auto obj = make_object(10, 10, 50, 50, 0.9f, 1);
    auto frame = tracker.update({obj});
    EXPECT_TRUE(frame.empty());
}

TEST(byte_tracker, low_score_object_not_activated) {
    ByteTracker tracker;
    auto cfg = build_byte_track_cfg();
    ASSERT_EQ(tracker.init(cfg), StatusCode::OK);

    // score 0.3 < tracker_thresh 0.5: goes to low-score detection, not activated
    auto obj = make_object(10, 10, 50, 50, 0.3f, 0);
    auto frame = tracker.update({obj});
    EXPECT_TRUE(frame.empty());
}
