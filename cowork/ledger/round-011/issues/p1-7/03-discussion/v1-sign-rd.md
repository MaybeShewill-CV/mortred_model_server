# 第3步签字

- 问题: P1-7
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 39 个 `server_uri` 已各自生成 path。共用 `Request_*` 的 id 仍是独立 path，示例可以写在该 path 的 `application/json.example` 上，不必改 `components.schemas`。请求 schema 必填 `images`、`additionalProperties` 为 false，`req_id` 可选，只有这两个键的占位能通过。200 响应是 `UnifiedResponse` 与带 data `$ref` 的 `allOf`；分类、检测、人脸、分割、OCR、抠图、增强、深度、特征点的现有 `$ref` 与方案字段一致。`msocrnet` 与三个 DINOv2 的 data `$ref` 已是只有 `nullable` 的 `EnvelopeData`，对象占位能过 schema，字段口径由 `--check` 另判。`x-contract-hash` 只哈希 `docs/contract_dump.json`，不动 dump 则保持 `b320c08dde6d040146cc649381361fae41acb57ea3ba9e531c842e6b0aece400`。`check_consistency.py` 只核对 path 覆盖和 header 内嵌；重新生成两份产物后仍一致。`GET /openapi.json` 原样下发 `openapi_doc.h`，不改进程。
