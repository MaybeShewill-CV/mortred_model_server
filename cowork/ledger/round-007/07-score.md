# 项目评分

- 轮次: 7
- 自算原始分: 8.3
- 自算加权分: 8.3
- 上限规则: 无
- 相对上轮综合分: 不高于 8.3

## 模块
| id | 权重 | 分数 | 记录 |
| --- | --- | --- | --- |
| src-models | 15 | 8.0 | 本轮复审（AttentiveGAN / HRNet 错码） |
| src-server-apps | 15 | 8.1 | 未改，沿用上轮 |
| src-control | 10 | 8.7 | 未改，沿用上轮 |
| src-factory-common | 5 | 7.6 | 未改，沿用上轮 |
| conf | 10 | 8.0 | 未改，沿用上轮 |
| ops | 15 | 8.7 | 未改，沿用上轮 |
| test | 15 | 8.7 | 本轮复审 |
| docs | 10 | 8.7 | 未改，沿用上轮 |
| ci | 5 | 7.2 | 未改，沿用上轮 |

## P0
无

## P1
无

## 附录
AttentiveGAN / HRNet 错码已关闭。`lightglue.inl` 与 `openai_clip.inl` 仍有非空故障回 `MODEL_EMPTY_OUTPUT` 的路径，本轮未升。

## 未读模块
无
