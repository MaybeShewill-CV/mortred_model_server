# SME-07 evidence — auth before IP rate limit (WSL)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-07-auth-before-rate-limit` @ `a68c716`
- **Host:** WSL (`DESKTOP-B8MJFUS`), `/mnt/g/Codex/mortred_model_server`
- **Intent:** A — authenticate before per-IP `rate_limit_qps`; exempt `/healthz` `/ready` `/openapi.json`

## Commands / results

```text
./_bin/server_e2e_contract_test --gtest_filter='*rate_limited*:*unauthenticated_stays_401*:*healthz_exempt*'
→ rate_limited_returns_429 PASSED
→ unauthenticated_stays_401_under_rate_limit PASSED
→ healthz_exempt_from_ip_rate_limit PASSED
→ ut_exit=0
```

## Outcome

**PASS.** Unauthenticated callers stay **401** under a tight QPS; authorized overflow still **429**; healthz remains **200** after IP budget exhausted.
