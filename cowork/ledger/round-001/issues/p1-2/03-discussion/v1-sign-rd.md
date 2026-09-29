# 第3步签字

- 问题: P1-2
- 方案版本: v1
- 结论: 不同意
- 条件: 卡在 `scripts/check_consistency.py` 规则 20（`check_example_packs_worker_nums`）：凡 `conf/packs/*.toml` 出现 `gpu_mem_mib` 即报错「machine-local; do not commit occupancy stamps」。按方案把 seed 的 `gpu_mem_mib` 写入 git pack 后，该门禁会红。方案改动面必须授权放宽该规则，且仅允许 `gpu_mem_source="seed"` 且没有 `gpu_name` / `gpu_memory_total_mib` 的起始样本；其它 `gpu_mem_mib` 仍禁止提交。
- 理由: occupancy 检查与 calibrate 拒写可以按方案落地，且不必削弱 demo self-test 与 doctor fail-closed。代表性 git pack 带 seed 与现有一致性门禁冲突，RD 不能自行改掉该门禁。
