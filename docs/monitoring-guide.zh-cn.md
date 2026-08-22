# 监控指南

| [English](monitoring-guide.md) | [中文](monitoring-guide.zh-cn.md) |
|---|---|

Mortred Model Server 的三个组件均暴露 Prometheus 指标。本指南涵盖部署、指标参考、仪表盘与告警。

## 快速开始

```bash
# 1. 启动监控栈（Prometheus + Grafana）
docker compose -f deploy/docker-compose.monitoring.yml up -d

# 2. 打开 Grafana
# http://localhost:3000  (admin/admin)
#
# 注意：在 Linux 上，需要将 deploy/prometheus.yml 中的 "localhost"
# 替换为 "host.docker.internal"，容器才能访问宿主机服务。
```

## 指标来源

| 组件 | 端口 | 端点 | 关键指标 |
|---|---|---|---|
| 网关 | :8080 | `/metrics` | `mortred_http_requests_total`、`mortred_http_request_duration_ms` |
| 监督器 | :8787 | `/metrics` | `mortred_supervisor_state`、`mortred_supervisor_restarts_total` |
| 模型服务器 | :9001-9074 | `/metrics` | `mortred_inference_duration_ms`、`mortred_queue_depth`、`mortred_workers_busy`、`mortred_batch_size`、`mortred_async_jobs_total` |

## 指标参考

### 可用性

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `mortred_up` | gauge | model | 进程存活 (1) |
| `mortred_ready` | gauge | model | 有可用 worker (1) |
| `up` | gauge | job | Prometheus 抓取目标可达性 |

### 流量

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `mortred_http_requests_total` | counter | model, method, status | HTTP 请求总数 |
| `mortred_http_request_duration_ms` | histogram | model, method, status | HTTP 请求耗时 |
| `mortred_queue_rejected_total` | counter | model | 队列满拒绝的请求数 (429) |

### 推理

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `mortred_inference_duration_ms` | histogram | model | 模型推理耗时 |
| `mortred_queue_wait_duration_ms` | histogram | model | 等待 worker 耗时 |
| `mortred_inference_success_total` | counter | model | 成功推理数 |
| `mortred_inference_failure_total` | counter | model | 失败推理数 |
| `mortred_workers_busy` | gauge | model | 当前繁忙 worker |
| `mortred_workers_available` | gauge | model | 空闲 worker |
| `mortred_queue_depth` | gauge | model | 当前排队深度 |

### 批处理

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `mortred_batch_size` | histogram | model | 实际执行的批大小 |
| `mortred_batch_window_wait_ms` | histogram | model | 批收集窗口等待时间 |

### 异步任务

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `mortred_async_jobs_total` | counter | model, state | 异步任务按状态（submitted/running/done/failed/timeout） |
| `mortred_async_queue_depth` | gauge | model | 当前异步队列深度 |
| `mortred_async_job_duration_ms` | histogram | model | 异步任务执行耗时 |

### 监督器

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `mortred_supervisor_state` | gauge | server | 进程状态 (0=stopped, 1=starting, 2=running, 3=backoff, 4=failed) |
| `mortred_supervisor_ready` | gauge | server | 进程就绪 |
| `mortred_supervisor_restarts_total` | counter | server | 重启总数 |

## 告警

告警规则定义在 `deploy/alert-rules.yml`。关键告警：

| 告警 | 级别 | 触发条件 |
|---|---|---|
| GatewayDown | critical | 网关 30 秒不可达 |
| SupervisorDown | critical | 监督器 60 秒不可达 |
| ModelServerDown | warning | 模型服务器 60 秒不可达 |
| HighQueueDepth | warning | 队列深度 > 20 持续 60 秒 |
| OverloadRejections | warning | 429 速率 > 0.1/s 持续 2 分钟 |
| HighLatency | warning | p95 > 2000ms 持续 5 分钟 |
| HighErrorRate | critical | 5xx > 5% 持续 2 分钟 |
| AllWorkersBusy | critical | 0 可用 worker 持续 2 分钟 |
| AsyncQueueFull | warning | 异步深度 > 10 持续 2 分钟 |
| RestartStorm | critical | 重启速率 > 0.2/s 持续 2 分钟 |

## Grafana 仪表盘

导入 `deploy/grafana-dashboard.json`（Dashboard → Import → Upload JSON）。13 个面板：

1. 服务可用性（状态灯）
2. 推理速率（按模型 req/s）
3. 错误率（5xx %）
4. 过载拒绝（429/s）
5. 推理延迟 p50/p95/p99
6. 队列深度
7. Worker（繁忙/空闲）
8. 批大小分布
9. 批窗口等待
10. 异步任务（按状态）
11. 异步队列深度
12. 监督器进程状态
13. 进程重启率

## 手动验证

```bash
# 检查网关指标
curl -s http://localhost:8080/metrics | head -20

# 检查模型服务器指标
curl -s http://localhost:9002/metrics | grep mortred_inference_duration

# 检查监督器指标
curl -s -H "Authorization: Bearer $MORTRED_API_TOKEN" http://localhost:8787/api/v1/metrics

# 验证 Prometheus 抓取
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```
