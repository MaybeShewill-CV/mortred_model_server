# 第3步签字

- 问题: P0-2
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: AC3 已明确要求 deadline 内网关未到 `running` 时，`GET /api/v1/status` 给出含 deadline/未就绪关键字的可读原因，且 supervisor 可达时 `mortredctl doctor`（live）退出码 ≠ 0 并输出同一原因；与本版其余边界一致，可落地交付。
