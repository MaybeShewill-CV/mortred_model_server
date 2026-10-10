# 第3步签字

- 问题: P1-7
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: v1 只覆盖冻结句 H1。39 个 path 与现有 `server_uri` 一致，含已有人读片段的 7 个；每个 path 各有一条内联请求 `example` 和一条 200 `example`，并按该 path 现有 schema 校验。`yolov8n`、`yolov8l`、`yolov8x` 的 `model.name` 分别是这三个 catalog id。`libface` 与 `centerface` 含 `landmarks`，`superpoint` 含 `score`、`location`、`descriptor`。`msocrnet` 与三个 DINOv2 的 data `$ref` 保持 `EnvelopeData`，字段口径由 `--check` 另判。`x-contract-hash` 仍由 dump 计算，dump 不动则保持原值。验收 1 至 11 都能一次黑盒判定。自测是 `python scripts/gen_openapi.py --check` 与 `python scripts/check_consistency.py`，期望退出码都是 0。不改推理、产品 toml、engine、进程，也不扩到 P0-2、P0-3、P1-8 至 P1-12、D-8 至 D-12。
