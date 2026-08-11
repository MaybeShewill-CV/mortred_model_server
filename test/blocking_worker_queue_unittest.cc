/************************************************
 * Author: Codex
 * File: blocking_worker_queue_unittest.cc
 * Date: 2026-08-11
 ************************************************/

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "stl_container/blockingconcurrentqueue.h"

using Queue = moodycamel::BlockingConcurrentQueue<std::unique_ptr<int> >;

TEST(blocking_worker_queue, enqueue_then_wait_dequeue) {
    Queue q;
    q.enqueue(std::unique_ptr<int>(new int(42)));
    std::unique_ptr<int> item;
    q.wait_dequeue(item);
    ASSERT_NE(item, nullptr);
    EXPECT_EQ(*item, 42);
}

TEST(blocking_worker_queue, wait_blocks_until_wakeup) {
    Queue q;
    std::atomic<bool> done{false};
    std::thread waiter([&] {
        std::unique_ptr<int> item;
        q.wait_dequeue(item);
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(done.load()) << "空队列时 wait_dequeue 不应立即返回";
    q.enqueue(std::unique_ptr<int>(new int(7)));
    waiter.join();
    EXPECT_TRUE(done.load()) << "enqueue 后等待者应被唤醒";
}

TEST(blocking_worker_queue, timed_wait_timeout) {
    Queue q;
    std::unique_ptr<int> item;
    bool ok = q.wait_dequeue_timed(item, std::chrono::milliseconds(50));
    EXPECT_FALSE(ok);
}

TEST(blocking_worker_queue, fifo_order) {
    Queue q;
    for (int i = 1; i <= 3; ++i) {
        q.enqueue(std::unique_ptr<int>(new int(i)));
    }
    for (int i = 1; i <= 3; ++i) {
        std::unique_ptr<int> item;
        q.wait_dequeue(item);
        ASSERT_NE(item, nullptr);
        EXPECT_EQ(*item, i);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
