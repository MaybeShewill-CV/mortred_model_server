# 第3步签字

- 问题: P1-2
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 现有 `probe_readiness`/`probe_path_of` 已区分空 `uri`；在该路径立刻写 `failed` + `error`、并同步 `models_failed`/`reason` 落在 `supervisor.cpp` 即可。空 uri 可经 unittest 注入（不放宽 catalog）。AC1 可用计时在 ≤2s 且长 `models_deadline_ms` 下断言 `state=="failed"`；AC2/AC3 可对 `error`/`autostart.reason` 做字面 `server_uri` 断言。AC4 仅在 supervisor 可达时要求真实 `mortredctl doctor`，不挡 WSL 单测合入。
