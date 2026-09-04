/************************************************
 * Fail-closed weight gating for CI inference jobs.
 *
 * Local / tests-only builds keep GTEST_SKIP when weights are missing so a
 * laptop without the HF tree stays green. Jobs that claim to run inference
 * set MORTRED_CI_REQUIRE_WEIGHTS=1 (any value other than empty / "0" /
 * "false") and the same site becomes GTEST_FAIL.
 *
 * MORTRED_UPDATE_GOLDEN skips are unrelated: they must stay skips.
 ************************************************/

#ifndef MORTRED_TEST_CI_REQUIRE_WEIGHTS_H
#define MORTRED_TEST_CI_REQUIRE_WEIGHTS_H

#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

inline bool mortred_ci_require_weights() {
    const char *value = std::getenv("MORTRED_CI_REQUIRE_WEIGHTS");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

// Use only in a TEST body (or a helper that is the entire TEST body).
// GTEST_SKIP / GTEST_FAIL both return from the current function.
#define MORTRED_SKIP_OR_FAIL_WEIGHTS(msg)                                                                                              \
    do {                                                                                                                               \
        if (mortred_ci_require_weights()) {                                                                                            \
            GTEST_FAIL() << (msg);                                                                                                     \
        } else {                                                                                                                       \
            GTEST_SKIP() << (msg);                                                                                                     \
        }                                                                                                                              \
    } while (0)

#endif // MORTRED_TEST_CI_REQUIRE_WEIGHTS_H
