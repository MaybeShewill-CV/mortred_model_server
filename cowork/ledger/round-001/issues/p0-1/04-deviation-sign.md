# 偏离签字

- 问题: P0-1
- 方案版本: v2
- 结论: 不同意
- 理由: `python3 scripts/pack_occupancy.py --self-test` 必须回到方案自测表期望退出码 `0`；不得以 demo 无 stamp / P1-2 未改作为放过理由（不写 stamp 样本前提下把该自测改绿即可）。`check_consistency` 在已确认不再因产品 static_bs1 路径不在遗留 `trt_engines.json` 而红、剩余仅为既有 openapi/catalog 时，以及无 GPU/trtexec/onnx 导致 prepare 与 AC8 未执行，可接受为环境/既有债阻塞，但不构成把整包写成「偏离：无」。
