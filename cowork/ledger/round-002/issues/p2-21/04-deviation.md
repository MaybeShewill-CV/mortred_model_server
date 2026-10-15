# 实现偏差

- 方案版本: v1
- 分支: cowork/r2-p2-21-trt-xml
- commit: 970b276e13fccdb78fad73865b694be9f95d999f
- 技术负责人结论: 同意

## 偏差

无。产品行为与设计一致。

## 未偏离

- `--product-trt yes` 无 XML 时不打印 `TRT_PROOF=yes`。
- XML 需含未 skip 的 `Yolov8TrtDetection` / `TrtProductGolden`。
- 未扩大 `MORTRED_GPU_SMOKE_FILTER`；工作流仍可 `--product-trt no`。
- 跳过路径仍为 `NOT_A_TRT_PROOF` + exit 2。
