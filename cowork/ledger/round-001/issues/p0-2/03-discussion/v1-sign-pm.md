# 第3步签字

- 问题: P0-2
- 方案版本: v1
- 结论: 不同意
- 条件: AC3（或新增对等 AC）须黑盒写死：网关 deadline 内未就绪时，`GET /api/v1/status` 给出可读失败/阻断原因；且 supervisor 可达时 `mortredctl doctor`（live）退出码 ≠ 0、输出含该原因，并能看出本次未起 model（与 AC4 对 token 案的要求对称）。不得只验「不起 model」。
- 理由: AC1/AC2/AC4 已可黑盒判定并覆盖「配错 token → status/API/doctor 给永久失败原因 + 本次不起 model」。AC5/AC6 可判定且不掺 P1。但冻结「改完后」同时要求 **网关未就绪** 时 status/API/`doctor --live` 给出明确原因与勿起 model 终态；现行 AC3 只落到阻断/不起 model，未把可读原因与 doctor live 非 0 绑进编号验收，故全文未盖住，不能签字。
