# 模块评分

- id: src-server-apps
- 权重: 20
- 分数: 9.4
- 读过的路径: `src/apps/model_server_main.cpp`、`src/server/abstract_server.h`、`src/server/base_server_impl.h`、`src/server/async_endpoints.h`、`src/server/http_status.h`、`src/server/worker_pool.h`、`src/server/generic_cv_server.h`、`docs/perf/coverage_set_http_rps.md`、相关 e2e / contract 测试引用

## 已具备

- `--model` 大小写敏感，必须等于 catalog `model_section`；未知 id 打印 `unknown --model`（`model_server_main.cpp`）。
- 等待队列打满返回 429 与 `Retry-After`；未知字段 422；空图与错误 token 不进成功信封。
- `docs/perf/coverage_set_http_rps.md` 九行覆盖集均有 calibrate + `--raw` 压测；排队与推理直方图分开；batch 观测为 1；gateway 五段与 `server_time_ms` 同次可读。
- 并发按 rubric 取已部署 `worker_nums` 的两倍。

## 问题

无

## 范围外

- 已关闭的 ST/PF/TP 相关重复描述不重新入列。
- 响应里 `server_time_ms` 仍是排队等待与本次 run 之和（读两张直方图可拆开），不单列未满足行。
- 模型侧 IP 限流与 `/jobs` 队列满的 429 不写 `Retry-After`；覆盖集 toml 未打开这两条。

## 依据

分数 9.4。落在本模块的 PF1–PF5、ST1、ST3、TP1 对得上当前文件与九行报告。本次未重跑压测与 e2e，行为来自源码与已写报告；不把未跑写成失败。

## 未覆盖

无
