# 第3步签字

- 问题: P1-1
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v2 仍落在既有 `status.autostart`/`AutostartGate`，明确不放宽空 scrape fail-closed，且验收与自测一律用真实 `mortredctl doctor`、禁止新增 `--live`；AC 与自测一一对应，可作为最后一版交付 RD/QA。
