/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: trust_tokens.h
* Date: 26-9-7
************************************************/

// Fail-closed scrape-token rules shared by the gateway and the supervisor.
// GET /metrics is never public: MORTRED_METRICS_TOKEN must be set and must
// not equal the inference or management bearer. Callers never mint a token.

#ifndef MORTRED_CONTROL_TRUST_TOKENS_H
#define MORTRED_CONTROL_TRUST_TOKENS_H

#include <string>

namespace mortred {
namespace control {

inline constexpr const char* kScrapeTokenMissing =
    "MORTRED_METRICS_TOKEN is unset";
inline constexpr const char* kScrapeTokenCollision =
    "MORTRED_METRICS_TOKEN matches an inference or management token";

struct ScrapeTokenCheck {
    bool ok = false;
    const char* reason = "";
};

inline ScrapeTokenCheck scrape_token_usable(const std::string& metrics,
                                            const std::string& infer,
                                            const std::string& admin) {
    if (metrics.empty()) {
        return {false, kScrapeTokenMissing};
    }
    if ((!infer.empty() && metrics == infer) || (!admin.empty() && metrics == admin)) {
        return {false, kScrapeTokenCollision};
    }
    return {true, ""};
}

inline constexpr const char* kInferTokenMissing =
    "MORTRED_GATEWAY_AUTH_TOKEN is unset";
inline constexpr const char* kAdminTokenMissing = "MORTRED_API_TOKEN is unset";
inline constexpr const char* kInferAdminIdentical =
    "MORTRED_GATEWAY_AUTH_TOKEN and MORTRED_API_TOKEN are identical";

// Gateway listen gate. Metrics rules stay first; a key file is not an input.
inline ScrapeTokenCheck tokens_usable_for_listen(const std::string& metrics,
                                                 const std::string& infer,
                                                 const std::string& admin) {
    const ScrapeTokenCheck scrape = scrape_token_usable(metrics, infer, admin);
    if (!scrape.ok) {
        return scrape;
    }
    if (infer.empty()) {
        return {false, kInferTokenMissing};
    }
    if (admin.empty()) {
        return {false, kAdminTokenMissing};
    }
    if (infer == admin) {
        return {false, kInferAdminIdentical};
    }
    return {true, ""};
}

inline std::string scrape_token_spawn_error(const ScrapeTokenCheck& check) {
    return std::string("gateway scrape token rejected: ") + check.reason;
}

inline bool is_scrape_token_error(const std::string& err) {
    return err.find("gateway scrape token rejected:") != std::string::npos;
}

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_TRUST_TOKENS_H
