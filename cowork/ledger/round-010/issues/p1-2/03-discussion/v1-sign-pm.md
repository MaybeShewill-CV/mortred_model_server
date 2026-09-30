# 第3步签字

- 问题: P1-2
- 方案版本: v1
- 结论: 不同意
- 条件: 验收须覆盖冻结条目的可观察变化：对 CLIP、LightGlue、SAM、diffusion，在启动或契约层立刻点名缺段；对错会话绑定立刻点名错会话。不得把范围缩成仅 CLIP 缺 text_backend。
- 理由: v1 只验收 CLIP 缺 text_backend，明确不测另外三族和错会话。
