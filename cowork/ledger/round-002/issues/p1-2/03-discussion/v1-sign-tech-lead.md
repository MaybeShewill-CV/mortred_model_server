# 第3步签字

- 问题: P1-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 边界与改动面守住一 catalog 一进程、gateway `:8080`；明确不放宽 `catalog` 的 `server_uri` 必填（AC5）；失败终态走既有 child `error`/`state=failed`，autostart 波复用 `models_failed`，不新增 HTTP 或 status 字段名。
