# 第3步签字

- 问题: P1-2
- 方案版本: v1
- 结论: 不同意
- 条件: AC1 须改为可在 WSL 直接判定的时限（例如：进入 readiness 探测路径后不超过 2 秒、且默认 models deadline 尚未触发时，`status(id).state == "failed"`，不得仍为 `starting`）。AC3 的 `reason` 须改为可字符串判定的关联方式（例如：`reason` 含 `server_uri`，或与该 child `error` 含同一失败要点），不得只写「能关联」。
- 理由: 使用者、场景与「非 gateway 空 uri 立刻终态、不靠 deadline、不放宽 catalog」与冻结 P1-2 一致，AC2/AC4–AC6 也可在 WSL 用退出码或字段判定；但 AC1「远短于」、AC3「能关联」仍无法黑盒裁定通过与否，故不能签字。
