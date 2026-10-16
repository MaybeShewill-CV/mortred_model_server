# Design P1-3 v1 (user-locked)
## Goal
Model /metrics accepts MORTRED_METRICS_TOKEN (in addition to internal auth token).
## Fix
1. Supervisor injects MORTRED_METRICS_TOKEN into model children when available (still scrubbed then re-injected).
2. base_server_impl: /metrics authorized if bearer matches auth_token OR metrics scrape token from env.
