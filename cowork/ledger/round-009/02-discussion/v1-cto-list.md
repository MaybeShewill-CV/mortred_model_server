# 列表修订

- 轮次: 9
- 版本: v1
- 针对: 起点综合分 8.4；第 8 轮四条已关闭后的复审新证据

## P0
无

## P1
无

## 附录
- `scripts/check_legacy_yolov8s_engine_align.py` 可跑通，但 doctor、init、CI 都没有调用。
- lightglue / clip 契约仍无 dtype/NaN/Inf 全矩阵。golden 缺权重仍 SKIP。

## 未采纳
| id | 来源表决 | 原因 |
| --- | --- | --- |
| cand-align-wire | ops 复审 | 路径对齐已关闭。未挂主路径不是每换机必做的手工步骤。 |
| cand-lg-clip-matrix | test 复审 | 错码与 YOLOv8 +Inf 已关闭。无新故障，不升 golden SKIP。 |
