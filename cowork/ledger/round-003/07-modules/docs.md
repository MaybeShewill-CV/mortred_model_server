# 模块评分

- id: docs
- 权重: 5
- 分数: 8.2
- 读过的路径: `docs/api-contract.md`、`docs/unsupported-boundaries.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`docs/tutorials_of_model_servers.md`、`docs/tutorials_of_model_servers.zh-cn.md`、`docs/how_to_add_new_server.md`、`docs/how_to_add_new_server.zh-cn.md`、`README.md`、`README.zh-cn.md`、`docs/openapi.json`、`docs/perf/coverage_set_http_rps.md`、`cowork/ledger/round-003/issues/p2-22/05-merge.md`、`cowork/ledger/round-003/issues/p2-23/05-merge.md`

## 已具备

- 第 2 轮已关闭的覆盖集教程、worker_nums/pack、how-to densenet、契约 model.name，当前文件仍保持。
- P2-22 已合入：两份教程非覆盖集三条 `--model` 为 `yolov5l` / `bisenetv2` / `attentive_gan`，章节标题同步。
- P2-23 已合入：两份 deployment §5.2 / §14.1 不再要求随包 `*_cpu` 产品，与 §10.7 自建 cpu pack、demo 为 TRT 种子一致。
- `docs/unsupported-boundaries.md` 与 §10.7 仍写明不期望默认 cpu 产品随包。

## 问题

无。本轮锁定的两条 P2 均已关闭。

## 范围外

- 已关闭重复描述不重开。
- `docs/ui-redesign/` 等可选长文未逐份重读；必读路径已覆盖，不触发未覆盖封顶。

## 依据

分数 8.2：第一小时、边界、OpenAPI、九行 HTTP 报告与教程/部署 cpu 预期在关闭 P2-22/P2-23 后可无歧义走完文档验收路径。高于本轮起点 7.8。未到 9.x：仍有可选文档与现场笔记对齐成本，且未做整包文档审计。

## 未覆盖

无
