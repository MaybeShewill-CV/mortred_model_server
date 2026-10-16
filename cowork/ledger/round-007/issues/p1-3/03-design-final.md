# 技术方案
- 问题: P1-3
- 版本: v1
## 目标
CI/release 运行 verify_gpu_gate_evidence（forks `--allow-missing-artifacts`）；版本检查保持独立 fail-closed（P0-1）。
## 边界
- 不改 packing/Docker；不要求 fork 有 GPU artifacts。
## 实现要点
1. ci.yml consistency step：verify example manifest with --allow-missing-artifacts。
2. release.yml：同样核验（allow-missing 或对真实 artifact 严格，取决于发布是否捆绑 evidence；默认 example+allow-missing 保证门禁存在）。
3. docs/ci/gpu-gate-evidence.md 写明 CI 已接入。
## 确认
| 角色 | 结论 | 方案版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意 | v1 | Round 7 队列锁定 |
