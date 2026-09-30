# 第3步签字

- 问题: P1-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `real_esrgan.h` 中 `postprocess` 已在 `protected`；`real_esrgan.inl` 约 93–95 行体积不符仍返回 `MODEL_EMPTY_OUTPUT`。空 outputs 与契约失败路径可按方案分流；FULL 下已有同类 `*_output_contract_unittest` 挂载先例，子类暴露 protected `postprocess` 无需改访问级别。方案在现有进程/后端内可实现。
