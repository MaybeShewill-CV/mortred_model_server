# Round 7 discovered

- P0-1: CI line passes version script as positional to check_consistency (argparse error / never runs as intended).
- P0-2: load() omits max_queue and worker_budget.
- P1-1: NvmlGpuProbe stub; sampler/supervisor still fork smi.
- P1-2: release_queue immediately after inflight acquire; worker_budget soft-only.
- P1-3: verify_gpu_gate_evidence not wired in CI.
- P1-4: FixedWindowRateLimiter member still on BaseServerImpl product path.
