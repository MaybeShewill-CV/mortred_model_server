/************************************************
 * Round 6 P1-2: multi-tenant quota noisy-neighbor baseline
 ************************************************/
#include <memory>
#include <gtest/gtest.h>
#include "control/api_key_manager.h"

using mortred::control::ApiKey;
using mortred::control::ApiKeyManager;
using mortred::control::AuthReason;

static std::shared_ptr<ApiKey> make_key(const char* name, int qps, int inflight, int queue) {
    auto k = std::make_shared<ApiKey>();
    k->name = name;
    k->scope = "inference";
    k->enabled = true;
    k->rate_limit_qps = qps;
    k->max_inflight = inflight;
    k->max_queue = queue;
    k->worker_budget = inflight > 0 ? inflight : 0;
    return k;
}

TEST(TenantQuotaNoisyNeighbor, noisy_tenant_cannot_starve_other_inflight) {
    auto noisy = make_key("noisy", 0, 2, 4);
    auto quiet = make_key("quiet", 0, 2, 4);
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(noisy.get()));
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(noisy.get()));
    EXPECT_FALSE(ApiKeyManager::try_acquire_inflight(noisy.get()));
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(quiet.get()));
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(quiet.get()));
    EXPECT_FALSE(ApiKeyManager::try_acquire_inflight(quiet.get()));
    EXPECT_GE(noisy->reject_inflight.load(), 1u);
    ApiKeyManager::release_inflight(noisy.get());
    ApiKeyManager::release_inflight(noisy.get());
    ApiKeyManager::release_inflight(quiet.get());
    ApiKeyManager::release_inflight(quiet.get());
}

TEST(TenantQuotaNoisyNeighbor, queue_budget_independent_per_tenant) {
    auto a = make_key("a", 0, 0, 1);
    auto b = make_key("b", 0, 0, 1);
    EXPECT_TRUE(ApiKeyManager::try_acquire_queue(a.get()));
    EXPECT_FALSE(ApiKeyManager::try_acquire_queue(a.get()));
    EXPECT_TRUE(ApiKeyManager::try_acquire_queue(b.get()));
    EXPECT_GE(a->reject_queue.load(), 1u);
    ApiKeyManager::release_queue(a.get());
    ApiKeyManager::release_queue(b.get());
}

TEST(TenantQuotaNoisyNeighbor, gcra_qps_independent_per_tenant) {
    ApiKeyManager mgr;
    auto a = make_key("a", 1, 0, 0);
    auto b = make_key("b", 1, 0, 0);
    EXPECT_EQ(mgr.consume_rate_limit(a).reason, AuthReason::OK);
    EXPECT_EQ(mgr.consume_rate_limit(a).reason, AuthReason::RATE_LIMITED);
    EXPECT_EQ(mgr.consume_rate_limit(b).reason, AuthReason::OK);
    EXPECT_GE(a->reject_qps.load(), 1u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
