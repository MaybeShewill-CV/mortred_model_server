# 第3步签字

- 问题: P1-4
- 方案版本: v1
- 结论: 不同意
- 条件: 写清实测由谁提供、如何接到拒启文案，并给出 occupancy_gate 单测的可执行命令。现有 supervisor 只喂 stamp 和 nvidia-smi，未提供实测则跳过，列表要求的文案不会出现。
- 理由: 声明/实测比对放进 occupancy_ready_for_spawn 但没有生产侧接线。
