# 第3步签字

- 问题: P1-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: AC1–AC7 均可黑盒判定代表性 pack 带 seed occupancy，以及换机 calibrate 写回 doctor；无 GPU 时 AC7 记 blocked，不要求把瞬时 nvidia-smi 写进 git。
