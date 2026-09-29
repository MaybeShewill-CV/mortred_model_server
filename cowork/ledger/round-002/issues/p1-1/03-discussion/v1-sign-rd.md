# 第3步签字

- 问题: P1-1
- 方案版本: v1
- 结论: 不同意
- 条件: 自测命令表须补齐与 AC 一一对应的句子：（1）deadline 失败后 `GET /api/v1/status` 断言 `models_failed`（或方案等价名）为真、`reason` 含 deadline 与未就绪 id、`models_skipped` 为 false；（2）spawn 失败后同一组 status 断言，`reason` 含该 id 且 `models_skipped` 不为 true；（3）空 scrape 路径断言仍为 `blocked` 与 `models_skipped` 为 true、模型波终态为假、零 model spawn（可写明由 `supervisor_unittest` 对应用例覆盖）；（4）网关与模型均在 deadline 内 `running` 时模型波终态为假，且 `mortredctl doctor`（勿写不存在的 `--live`）不因本条非 0。失败路径 doctor 行保留退出码非 0 并含同一 `reason`。
- 理由: `AutostartGate` / `status.autostart` / `autostart_all` 模型波与 doctor 读 gate 的改动面可在一个 worktree 落地，现有 deadline/scrape 单测也可扩展；但当前自测表只有一条模糊 curl 与一条期望非 0 的 doctor，盖不住 AC1/AC2 分场景与 AC5 健康路径，且 `mortredctl doctor` 实际只认 `--strict`，`--live` 不是验收 CLI。
