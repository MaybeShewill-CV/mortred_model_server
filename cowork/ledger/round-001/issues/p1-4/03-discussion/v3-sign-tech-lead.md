# 第3步签字

- 问题: P1-4
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: 正文仍只改必读四份文档、明确不改运行时与 `src/`/`conf/`/`openapi`；推荐入口写死现有 `POST /v1/models/{id}/infer`（gateway `:8080`）；进程树仍是一 catalog 一 OS 进程；检测/OCR 按用户选定为顶层 JSON 数组，A4 自测已把 `loads_all` 顶层 `list` 收进检测/OCR 收集。
