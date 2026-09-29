# 第3步签字

- 问题: P1-4
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 边界与「明确不改」把交付锁在四份必读文档；进程树仍是一 catalog 一 OS 进程、gateway `:8080` / supervisor `:8787`；推荐入口字面为现有 `POST /v1/models/{id}/infer`；HTTP/tensor 运行时与后端（MNN/ORT/TRT、toml `type=`、engine）不变。
