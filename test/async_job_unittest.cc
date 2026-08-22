/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: async_job_unittest.cc
* Date: 26-8-22
************************************************/

// Unit tests for async job state machine, LRU eviction and condition-variable
// signaling. The full lifecycle (submit/poll/result) is covered by e2e tests.

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace {

enum class JobState { PENDING = 0, RUNNING = 1, DONE = 2, FAILED = 3, TIMEOUT = 4 };

struct AsyncJob {
    std::string id;
    JobState state = JobState::PENDING;
    int64_t completed_at_ms = 0;
    std::mutex wait_mu;
    std::condition_variable wait_cv;
};

bool is_terminal(JobState s) {
    return static_cast<int>(s) >= static_cast<int>(JobState::DONE);
}

}  // namespace

TEST(async_job, terminal_states_ge_done) {
    EXPECT_TRUE(is_terminal(JobState::DONE));
    EXPECT_TRUE(is_terminal(JobState::FAILED));
    EXPECT_TRUE(is_terminal(JobState::TIMEOUT));
    EXPECT_FALSE(is_terminal(JobState::PENDING));
    EXPECT_FALSE(is_terminal(JobState::RUNNING));
}

TEST(async_job, running_jobs_survive_lru_eviction) {
    std::mutex mu;
    std::unordered_map<std::string, std::shared_ptr<AsyncJob>> jobs;
    std::deque<std::string> lru;

    auto running = std::make_shared<AsyncJob>();
    running->id = "running_job";
    running->state = JobState::RUNNING;
    jobs[running->id] = running;
    lru.push_back(running->id);

    for (int i = 0; i < 10; ++i) {
        auto done = std::make_shared<AsyncJob>();
        done->id = "done_" + std::to_string(i);
        done->state = JobState::DONE;
        jobs[done->id] = done;
        lru.push_back(done->id);
    }

    // simulate LRU eviction: only remove terminal-state jobs
    int completed = 0;
    for (auto& [id, job] : jobs) {
        if (is_terminal(job->state)) {
            ++completed;
        }
    }
    while (completed > 5 && !lru.empty()) {
        auto it = jobs.find(lru.front());
        if (it != jobs.end() && is_terminal(it->second->state)) {
            jobs.erase(it);
            --completed;
        }
        lru.pop_front();
    }

    // running job must still exist
    EXPECT_NE(jobs.find("running_job"), jobs.end());
    EXPECT_EQ(jobs.count("done_0"), 0u);  // oldest done evicted
}

TEST(async_job, wait_cv_wakes_on_terminal_state) {
    auto job = std::make_shared<AsyncJob>();
    job->id = "cv_test";
    job->state = JobState::RUNNING;

    std::thread changer([job]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::lock_guard<std::mutex> lock(job->wait_mu);
        job->state = JobState::DONE;
        job->wait_cv.notify_all();
    });

    std::unique_lock<std::mutex> lock(job->wait_mu);
    bool woken = job->wait_cv.wait_for(lock, std::chrono::seconds(2), [&job]() {
        return is_terminal(job->state);
    });
    changer.join();
    EXPECT_TRUE(woken);
    EXPECT_EQ(job->state, JobState::DONE);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
