# 模块评分

- id: ops
- 权重: 15
- 分数: 8.8
- 读过的路径: `scripts/mortredctl_init.sh`、`scripts/mortredctl_prepare.sh`、`scripts/mortredctl_doctor.sh`、`docker-compose.yml`、`deploy/` 目录、`scripts/ci_container_boot.sh`、`scripts/server/http_infer_rps.py`、`docs/perf/coverage_set_http_rps.md`

## 已具备

- `mortredctl prepare --id` / `--pack` 可为覆盖集生成 static batch=1 engine（OP1）。
- `mortredctl status` / supervisor 状态可区分每 id 与最近退出（OP2）；单 id 重启不拖垮 gateway 与其他 id（OP3）。
- `mortredctl doctor --strict` 在缺 engine、缺占用戳、占用门闩关闭时非 0（TP3）。
- 首小时路径含 `mortredctl next` → `doctor`（UX2 运维侧）。
- `docker-compose.yml`：gpu 与 cpu 两档均挂 `MORTRED_PACK=.../demo.toml`；注释写明 demo 为 TRT 种子，cpu profile 默认不自启模型，需按 deployment §10.7 自建 pack。
- `scripts/ci_container_boot.sh` 在无默认 cpu 产品后，启动时注入 CI overlay（cpu model toml + `profile="cpu"` server + pack）。

## 问题

无

## 范围外

- 已关闭的 OP/TP 重复描述不重新入列。
- 本次未在 WSL 重跑 prepare/doctor/压测；判断对的是仓库脚本与清单。

## 依据

分数 8.8：换机准备、doctor、单 id 运维与官方压测清单仍在。cpu 路径改为 pack 注入，与 `ea22867d` 一致，不构成新的运维缺口。

## 未覆盖

无
