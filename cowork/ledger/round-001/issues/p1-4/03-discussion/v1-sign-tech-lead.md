# 第3步签字

- 问题: P1-4
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 只改必读四份文档，进程树仍是一 catalog 一 OS 进程、gateway `:8080`、supervisor `:8787`；推荐入口写死现有 `POST /v1/models/{id}/infer`，遗留 `{server_uri}` 仅兼容；`src/`、`conf/`、OpenAPI、MNN/ORT/TRT、CUDA 12 / TensorRT 10、`mortredctl prepare` 均明确不改。A1–A6 可在仓库内用 `rg`/`git diff` 黑盒判定，自测命令写了期望退出码。
