# 第3步签字

- 问题: D-11
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 第 108 行原文的 cpu 格是 `curated: mobilenetv2, resnet50`，Bench-only 是 `CLIP, LightGlue, FastSAM, SAM prompt, MsOcrNet`。第 253 行原文是 `models (cpu: the two `*_cpu` entries - mobilenetv2 / resnet50).`。两行都没有行尾空白。按方案各做一次原地替换：删掉 `, MsOcrNet`，两处 `resnet50` 改为 `resnet`，不增行不删行，行号不变。替换结果与验收 1、验收 2 的期望全文一致；Bench-only 按第一个句号切开后是 `CLIP, LightGlue, FastSAM, SAM prompt`，其中没有 `msocrnet`。现有文档一次就能改完，并被这两条命令看见。
