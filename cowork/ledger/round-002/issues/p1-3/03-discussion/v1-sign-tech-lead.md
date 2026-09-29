# 第3步签字

- 问题: P1-3
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 方案只加厚 `mortredctl prepare` 入口门禁，明确不改 TensorRT 转换语义与 CUDA 12 / TensorRT 10，一 catalog 一进程与 gateway `:8080` 不变，并禁止入口继续纯 `exec`。
