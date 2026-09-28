# 第3步签字

- 问题: P0-2
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 现有 `ProcessSupervisor::autostart_all`、`SupervisorApp::init` scrape 早退、`GET /api/v1/status` 与 `mortredctl_doctor.sh` live 路径均可按方案改到「管理面存活 + gateway 永久失败/deadline 阻断可查 + 本次不起 model + doctor live 非 0」；AC3 相对 v1 只叠加 status 可读原因与 doctor live 非 0，不要求改推理契约或进程模型。P0-1 的 pack TRT/occupancy 默认 fail-closed 在 doctor 中先于 live，本条只在 supervisor 可达后叠加网关/autostart 诊断，不推翻该门禁。
