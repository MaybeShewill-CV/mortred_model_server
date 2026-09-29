# 第3步签字

- 问题: P1-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 空 uri 时 `probe_path_of` 已返回 false、`probe_readiness` 早退，可在该路径立刻 `failed`+写 `error`（须含缺 `server_uri` 与未做 HTTP readiness），autostart 波内复用已有 `mark_models_failed_locked`；catalog 拒绝不变；doctor 无 `--live`，AC4 与脚本门闩一致。单测可在加载后构造空 uri child，WSL 上 `build/tests-only` 跑 `supervisor_unittest`/`catalog_unittest` 即可覆盖 AC1–AC3/AC5/AC6，一个 worktree 可完成。
