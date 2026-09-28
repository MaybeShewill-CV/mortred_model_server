# 第3步签字

- 问题: P0-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 方案只收口 conf 真源与 ops（`product_static_bs1` / set_a / `onnx_sources` / 产品 toml、`check_consistency`、`mortredctl_init`/`doctor`/`prepare`），明确不改 `src/models`/`src/server`/`src/control`、一 id 一进程、`:8080`/`:8787`、MNN/ORT/TRT 与产品 `type`。现有 `product_static_bs1.py`（含 `--check`/`--write-manifest`）、`pack_trt`/`pack_occupancy`、`prepare_pack` 已走产品表；doctor 的 GATE_FAIL、init 的权重/文案只需把默认改为 fail-closed 并改下一跳，不依赖新后端或新进程模型。自测命令均为仓库内既有可执行入口。
