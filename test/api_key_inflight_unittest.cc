// Gateway wires try_acquire_inflight/release_inflight around upstream forward (R5 P1-3).
#include <gtest/gtest.h>
#include "control/api_key_manager.h"

using mortred::control::ApiKey;
using mortred::control::ApiKeyManager;

TEST(ApiKeyInflight, unlimited_when_zero) {
    ApiKey k;
    k.max_inflight = 0;
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
}

TEST(ApiKeyInflight, enforces_budget) {
    ApiKey k;
    k.max_inflight = 2;
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_FALSE(ApiKeyManager::try_acquire_inflight(&k));
    ApiKeyManager::release_inflight(&k);
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
}
