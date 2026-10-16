/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: rate_limiter.h
* Date: 26-10-16
************************************************/

// Rate-limit authority (Round 6 P1-1):
// Gateway is the sole public limiter. Product-path metering is GCRA via this
// RateLimiter abstraction. FixedWindowRateLimiter in src/server/rate_limiter.h
// remains for unit tests only and is not wired on the model-server product path.

#ifndef MORTRED_CONTROL_RATE_LIMIT_RATE_LIMITER_H
#define MORTRED_CONTROL_RATE_LIMIT_RATE_LIMITER_H

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "control/rate_limit/gcra.h"

namespace mortred {
namespace control {
namespace ratelimit {

struct RateLimitDecision {
    bool allowed = true;
    int32_t retry_after_ms = 0;
    int limit = 0;  // configured rate (qps) for RateLimit-Limit header
};

/*** Abstract admission meter. Implementations are GCRA-first. */
class RateLimiter {
  public:
    virtual ~RateLimiter() = default;
    virtual RateLimitDecision admit(const std::string& key, int64_t now_ms) = 0;
    virtual int limit_qps() const = 0;
    virtual const char* layer_tag() const = 0;
};

/*** Single-flow GCRA meter (one TAT cell). Suitable for per-API-key quotas. */
class GcraFlowLimiter final : public RateLimiter {
  public:
    GcraFlowLimiter(uint32_t rate_per_sec, uint32_t burst, const char* layer)
        : policy_{rate_per_sec, burst == 0 ? rate_per_sec : burst}, layer_(layer) {
        if (policy_.burst < 1 && policy_.rate_per_sec > 0) {
            policy_.burst = policy_.rate_per_sec;
        }
        tat_ms_.store(0, std::memory_order_relaxed);
    }

    RateLimitDecision admit(const std::string& /*key*/, int64_t now_ms) override {
        RateLimitDecision out;
        out.limit = static_cast<int>(policy_.rate_per_sec);
        if (!policy_.enabled()) {
            return out;
        }
        const GcraDecision d = gcra_admit(tat_ms_, policy_, now_ms);
        out.allowed = d.allowed;
        out.retry_after_ms = static_cast<int32_t>(d.retry_after_ms < 1 && !d.allowed ? 1 : d.retry_after_ms);
        return out;
    }

    int limit_qps() const override { return static_cast<int>(policy_.rate_per_sec); }
    const char* layer_tag() const override { return layer_; }

    GcraPolicy policy() const { return policy_; }

  private:
    GcraPolicy policy_;
    const char* layer_;
    std::atomic<int64_t> tat_ms_{0};
};

/*** Authority mode for leftover dual-path migration documentation.
 * Product path is always unified (gateway sole public limiter). The env
 * MORTRED_RATE_LIMIT_AUTHORITY=legacy is accepted but ignored for model peer
 * FixedWindow — that product path has been deleted. */
inline bool authority_is_unified() {
    const char* env = std::getenv("MORTRED_RATE_LIMIT_AUTHORITY");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::string(env) != "legacy";
}

}  // namespace ratelimit
}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_RATE_LIMIT_RATE_LIMITER_H
