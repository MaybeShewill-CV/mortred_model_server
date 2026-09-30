# 列表修订

- 轮次: 9
- 版本: v2
- 针对: v1（空 P0/P1）；分歧 C1 / C2

## P0
无

## P1
无

## 附录
- `scripts/check_legacy_yolov8s_engine_align.py` 可跑通，但 doctor、init、CI 都没有调用。
- lightglue / clip 已有错码与代表性契约测，仍无 dtype/NaN/Inf 全矩阵。golden 缺权重仍 SKIP，不升。

## 未采纳
| id | 来源表决 | 原因 |
| --- | --- | --- |
| C1 | 技术负责人保留 | 脚本未挂 doctor/CI 是防回归缺口，不是每换机必做的易错手工步骤。yolov8s 路径已关闭。 |
| C2 | 技术负责人保留 | 补全矩阵是覆盖面扩展，现场无新故障。golden SKIP 不升。 |
