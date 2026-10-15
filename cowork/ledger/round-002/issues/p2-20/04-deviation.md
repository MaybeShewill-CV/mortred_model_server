# 实现偏差

- 方案版本: v1
- 分支: cowork/r2-p2-20-trt-proof
- commit: c15b1d723658b70079d0ab84e0d7e2e76344004c
- 技术负责人结论: 同意

## 偏差

无。产品行为与设计一致。

## 未偏离

- 默认 / `--product-trt no` 时 success 不打印 `TRT_PROOF=yes`。
- `--product-trt yes` 时仍可打印 `TRT_PROOF=yes`（供 P2-21）。
- 未读 gtest XML；未扩大 `MORTRED_GPU_SMOKE_FILTER`。
- 跳过路径仍为 `NOT_A_TRT_PROOF` + exit 2。
