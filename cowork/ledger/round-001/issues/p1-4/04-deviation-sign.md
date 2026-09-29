# 偏离签字

- 问题: P1-4
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: `~~~http` 只改人读页里非 JSON 示例的 Markdown 围栏记号，避免 A4 `loads_all` 把原 ` ```http ` 的闭合 ` ``` ` 误当成新开围栏；推荐路径仍是 gateway `POST /v1/models/{id}/infer`，四类 `results[].data` 字段与顶层数组/对象形态未改，亦未动运行时或一 catalog 一 OS 进程。
