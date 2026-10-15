# 实现偏差

- 方案版本: v1
- 分支: cowork/r2-p2-19-gateway-dominant
- commit: 53ffb910936f72d98e0cb58b3b341691e6f653b8
- 技术负责人结论: 同意

## 偏差

仅构建/环境路径说明，产品行为与设计一致，无功能偏差。

1. 自测二进制路径：设计示例为 `./build/test/gateway_e2e_test`；本机 full-cpu 产物为 `/mnt/g/Codex/mortred_wt_r2_p2_19/_bin/gateway_e2e_test`（设计已写「实现时把二进制路径换成 worktree 构建产物」）。
2. 构建 profile：本机当前 WSL 无可用 `CMAKE_CUDA_COMPILER`，gpu profile 无法 configure；改用 `MORTRED_BUILD_PROFILE=cpu` 的 full 构建跑 gateway e2e（fake upstream，不需 TensorRT/CUDA）。设计验收明确不要求真实 GPU 模型。
3. worktree 无 gitignored 的 `3rd_party/`：对 baseline 做了目录符号链接 `3rd_party -> /mnt/g/Codex/mortred_model_server/3rd_party`，未拷贝、未改库内容。

## 未偏离

- 主导段门闩改为 `!stage_samples_.empty()`，未提高或去掉 `k_stage_sample_cap`。
- 未改 `record_stage_sample`、中位数/并列/严格大于规则、gauge 值 1。
- 改动面仅设计所列四文件（头文件无需改）。
