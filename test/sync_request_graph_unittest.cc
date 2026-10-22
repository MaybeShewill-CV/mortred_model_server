// Unit tests for the pure pieces of the sync request graph (the publish
// protocol itself is exercised through server_e2e_contract_test with a real
// Workflow series; this file covers what runs without one).

#include <set>
#include <string>

#include <gtest/gtest.h>

#include "server/sync_request_graph.h"

TEST(sync_request_graph, waiter_names_are_unique_and_embed_the_task_id) {
    const std::string task_id = "task-123";
    const auto a = mortred::server::make_sync_waiter_name(task_id);
    const auto b = mortred::server::make_sync_waiter_name(task_id);
    EXPECT_NE(a, b) << "two requests sharing a waiter name would count each other's completions";
    EXPECT_NE(a.find(task_id), std::string::npos);
    EXPECT_NE(a.find("mortred.syncwait."), std::string::npos);
}

TEST(sync_request_graph, waiter_names_are_unique_across_task_ids) {
    std::set<std::string> names;
    for (int i = 0; i < 100; ++i) {
        names.insert(mortred::server::make_sync_waiter_name("t" + std::to_string(i)));
    }
    EXPECT_EQ(names.size(), 100u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
