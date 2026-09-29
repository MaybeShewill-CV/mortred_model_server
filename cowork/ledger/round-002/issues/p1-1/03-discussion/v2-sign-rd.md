# 第3步签字

- 问题: P1-1
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v2 自测表已与 AC1–AC5 一一对应（AC1/AC2 分场景 status 断言、AC3 空 scrape 单测、AC4 健康路径 doctor 退出 0、AC5 失败路径 doctor 非 0 且含同一 reason）；验收 CLI 一律为 `mortredctl doctor`，未引入不存在的 `--live` 旗标。`AutostartGate` / `status.autostart` / doctor 读 gate 的改动面可在既有进程模型内落地。
