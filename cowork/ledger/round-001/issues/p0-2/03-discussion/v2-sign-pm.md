# 第3步签字

- 问题: P0-2
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: AC3 已黑盒写死：网关 deadline 内未就绪时 `GET /api/v1/status` 给出可读失败/阻断原因（含 deadline/未就绪关键字）；supervisor 可达时 `mortredctl doctor`（live）退出码 ≠ 0、输出含同一原因，并能看出本次未起 model；且写明不得只验「不起 model」。与 v1 条件对称，可签字。
