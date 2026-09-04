# 监控指南

| [English](monitoring-guide.md) | [中文](monitoring-guide.zh-cn.md) |
|---|---|

本指南涵盖 Mortred Model Server 监控体系的完整生命周期：部署、配置、指标含义、仪表盘解读、告警响应与故障排查。适用于运维人员（SRE / DevOps）与服务管理员。

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                       监控数据流                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   /metrics    ┌──────────────┐               │
│  │  网关     │ ───────────→ │              │               │
│  │ (:8080)  │               │  Prometheus  │               │
│  └──────────┘               │  (:9090)     │               │
│                              │  抓取+存储    │               │
│  ┌──────────┐   /metrics    │  +告警评估    │──→ 告警通知    │
│  │  监督器   │ ───────────→ │              │   (可选)      │
│  │ (:8787)  │               └──────┬───────┘               │
│  └──────────┘                      │                        │
│                              ┌──────▼───────┐               │
│  ┌──────────┐   /metrics    │   Grafana    │               │
│  │ 模型服务器 │ ───────────→ │  (:3000)     │               │
│  │(仅环回)   │               │  可视化面板   │               │
│  └──────────┘               └──────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

默认刮取目标是网关 `:8080/metrics`（环回上未设 `MORTRED_METRICS_TOKEN` 时公开；
非环回网关必须设置 scrape token）。监督器 `:8787/api/v1/metrics`
需要管理 Bearer token。模型 `/metrics` 仅环回，且在监督器注入了
`MORTRED_AUTH_TOKEN` 时需要同一 Bearer。不要为了刮指标而映射模型端口。

## 快速开始

### 方式一：Docker Compose 一键部署（推荐）

```bash
export GRAFANA_ADMIN_PASSWORD="$(openssl rand -hex 16)"
docker compose -f deploy/docker-compose.monitoring.yml up -d
# Grafana: http://localhost:3000（用户 admin / $GRAFANA_ADMIN_PASSWORD）
# Prometheus: http://localhost:9090（仅环回）
# 注意：Linux 上需将 prometheus.yml 中网关 target 的 localhost
# 改为 host.docker.internal
```

### 方式二：裸机部署

```bash
sudo apt install prometheus grafana
sudo cp deploy/prometheus.yml /etc/prometheus/prometheus.yml
sudo cp deploy/alert-rules.yml /etc/prometheus/alert-rules.yml
sudo systemctl restart prometheus
```

### 方式三：Prometheus 单独运行

```bash
prometheus --config.file=deploy/prometheus.yml --storage.tsdb.path=/tmp/prom-data
```

## 指标来源

| 组件 | 端口 | 端点 | 说明 |
|---|---|---|---|
| 网关 | :8080 | `/metrics` | 推理入口（环回上未设 token 时公开；非环回必须 scrape token） |
| 监督器 | :8787 | `/api/v1/metrics` | 进程管理（需要 Bearer `MORTRED_API_TOKEN`） |
| 模型服务器 | 环回 :9001-9074 | `/metrics` | 与 Prometheus 同一网络命名空间；不要映射这些端口 |

### 添加新模型抓取

仅当 Prometheus 能打到模型的**环回**端口时（host 网络，或与 systemd 同机）。
永远不要为了刮指标而 `docker publish` 模型端口。

```yaml
  - job_name: mortred-model-yolov5
    static_configs:
      - targets: ['localhost:9053']
        labels: { component: model, model: yolov5 }
```

## 指标参考

### 可用性（3 个）— "服务是否活着"

| 指标 | 类型 | 说明 |
|---|---|---|
| `mortred_up` | gauge | 进程存活 (1=活) |
| `mortred_ready` | gauge | 有可用 worker (1=就绪) |
| `up` | gauge | Prometheus 可抓取 (1=可达) |

> **`mortred_up` vs `up`**：前者是进程自报告，后者是 Prometheus 外部探测。`up == 0` 是更可靠的宕机信号。

### 流量（3 个）— "有多少请求"

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `mortred_http_requests_total` | counter | model, method, status | HTTP 请求累计 |
| `mortred_http_request_duration_ms` | histogram | model, method, status | 请求耗时（毫秒） |
| `mortred_queue_rejected_total` | counter | model | 429 拒绝累计 |

```promql
# 每秒请求数
sum(rate(mortred_http_requests_total[5m]))

# 按模型的 p95 延迟
histogram_quantile(0.95, sum(rate(mortred_http_request_duration_ms_bucket[5m])) by (le, model))

# 5xx 错误率
100 * sum(rate(mortred_http_requests_total{status=~"5.."}[5m])) by (model)
  / sum(rate(mortred_http_requests_total[5m])) by (model)
```

### 推理（7 个）— "模型本身表现"

| 指标 | 类型 | 说明 |
|---|---|---|
| `mortred_inference_duration_ms` | histogram | 模型推理耗时（不含排队） |
| `mortred_queue_wait_duration_ms` | histogram | 等待 worker 耗时 |
| `mortred_inference_success_total` | counter | 成功推理累计 |
| `mortred_inference_failure_total` | counter | 失败推理累计 |
| `mortred_model_output_contract_failures_total` | counter | ???? dtype/shape/buffer ?????? |
| `mortred_workers_busy` | gauge | 繁忙 worker 数 |
| `mortred_workers_available` | gauge | 空闲 worker 数 |
| `mortred_queue_depth` | gauge | 排队深度 |

> **诊断**：p95 延迟高时，先看 queue_depth——深排队 → 容量不足（加 worker）；浅排队但慢 → 模型/GPU 问题。

### 批处理（2 个）— "批处理是否有效"

| 指标 | 类型 | 说明 |
|---|---|---|
| `mortred_batch_size` | histogram | 实际执行批大小 |
| `mortred_batch_window_wait_ms` | histogram | 批收集窗口等待 |

```promql
# 平均批大小（> 1.5 说明有效）
sum(mortred_batch_size_sum) by (model) / sum(mortred_batch_size_count) by (model)
```

### 异步任务（3 个）— "长任务状态"

| 指标 | 类型 | 说明 |
|---|---|---|
| `mortred_async_jobs_total` | counter | 按状态计数（submitted/running/done/failed/timeout） |
| `mortred_async_queue_depth` | gauge | 异步队列深度 |
| `mortred_async_job_duration_ms` | histogram | 异步任务耗时 |

### 监督器（3 个）— "进程管理"

| 指标 | 类型 | 说明 |
|---|---|---|
| `mortred_supervisor_state` | gauge | 进程状态码 |
| `mortred_supervisor_ready` | gauge | 进程就绪 (1) |
| `mortred_supervisor_restarts_total` | counter | 重启累计 |

**状态码**：

| 值 | 状态 | 含义 |
|---:|---|---|
| 0 | stopped | 已停止 |
| 1 | starting | 启动中 |
| 2 | running | 正常运行 |
| 3 | backoff | 重启退避中 |
| 4 | failed | 崩溃循环放弃 |

## 告警参考

### 告警概览（12 条 × 5 组）

#### 可用性（4 条）

| 告警 | 级别 | 触发 | 含义 | 响应 |
|---|---|---|---|---|
| GatewayDown | critical | 网关 30 秒不可达 | 推理流量全断 | 检查网关进程 |
| SupervisorDown | critical | 监督器 60 秒不可达 | 进程管理丢失 | 检查 supervisor |
| ModelServerDown | warning | 模型 60 秒不可达 | 单模型不可用 | 查看该模型日志 |
| ModelNotReady | warning | 就绪探针失败 2 分钟 | 进程在但不可用 | 检查模型加载 |

#### 性能（5 条）

| 告警 | 级别 | 触发 | 含义 | 响应 |
|---|---|---|---|---|
| HighQueueDepth | warning | 深度 > 20 持续 60 秒 | 排队过长 | 加 worker 或扩容 |
| OverloadRejections | warning | 429 > 0.1/s 持续 2 分钟 | 开始拒绝请求 | 增大 max_queue_depth |
| HighLatency | warning | p95 > 2000ms 持续 5 分钟 | 推理变慢 | 检查 GPU / 模型 |
| HighErrorRate | critical | 5xx > 5% 持续 2 分钟 | 大量推理失败 | 查看模型日志 |
| AllWorkersBusy | critical | 0 空闲 worker 持续 2 分钟 | 可能卡死 | 检查 stuck-worker |

#### 异步（2 条）

| 告警 | 级别 | 触发 | 响应 |
|---|---|---|---|
| AsyncQueueFull | warning | 异步深度 > 10 持续 2 分钟 | 检查异步 worker 数 |
| AsyncTimeouts | warning | 超时 > 0.05/s | 检查 async_timeout |

#### 批处理（1 条）

| 告警 | 级别 | 触发 | 响应 |
|---|---|---|---|
| BatchNotCoalescing | info | 平均批 < 1.5 持续 10 分钟 | 检查并发量和 delay |

#### 监督器（1 条）

| 告警 | 级别 | 触发 | 响应 |
|---|---|---|---|
| RestartStorm | critical | 重启 > 0.2/s 持续 2 分钟 | 查崩溃原因 |

### 级别与响应

| 级别 | 含义 | 通知 | 响应时间 |
|---|---|---|---|
| critical | 服务不可用 | 电话/短信 | < 5 分钟 |
| warning | 性能退化 | Slack/邮件 | < 30 分钟 |
| info | 优化建议 | 仪表盘 | 下次巡检 |

### 自定义告警

```yaml
- alert: MortredCustomGPUHigh
  expr: your_gpu_metric > 90
  for: 300s
  labels: { severity: warning }
  annotations:
    summary: "GPU 使用率过高"
```

```bash
curl -X POST http://localhost:9090/-/reload  # 热加载
```

## Grafana 仪表盘

### 导入

Dashboard → Import → 上传 `deploy/grafana-dashboard.json` → 选择 Prometheus 数据源 → Import

### 面板解读

#### 第一行：全局概览

| # | 面板 | 看什么 |
|---|---|---|
| 1 | 服务可用性 | 全绿 = 正常；红 = 有宕机 |
| 2 | 推理速率 | 流量趋势；突降 = 异常 |
| 3 | 错误率 | 正常 < 1%；突升 = 模型问题 |
| 4 | 过载拒绝 | 非零 = 过载开始 |

#### 第二行：性能

| # | 面板 | 看什么 |
|---|---|---|
| 5 | 延迟 p50/p95/p99 | p99 突刺 = 个别慢请求 |
| 6 | 队列深度 | 持续上升 = 容量不足 |
| 7 | Worker 状态 | busy = total → 满载 |

#### 第三行：批处理与异步

| # | 面板 | 看什么 |
|---|---|---|
| 8 | 批大小 | 平均 > 1.5 = 批有效 |
| 9 | 批窗口等待 | 过高 = delay 太大 |
| 10 | 异步任务 | done 上升正常；timeout 上升异常 |

#### 第四行：系统健康

| # | 面板 | 看什么 |
|---|---|---|
| 11 | 异步队列深度 | > 10 = 积压 |
| 12 | 进程状态 | 2=绿(运行)、4=红(失败) |
| 13 | 重启率 | 非零持续 = 崩溃循环 |

## 故障排查

### Prometheus 抓取失败

```bash
curl -s http://localhost:9090/api/v1/targets | \
  jq '.data.activeTargets[] | select(.health != "up") | {job: .labels.job, error: .lastError}'
```

常见原因：模型未运行 / 端口错误 / Docker 网络（用 host.docker.internal）

### 指标缺失

```bash
curl -s http://localhost:8080/metrics | head -5
```

常见原因：进程未运行 / 端口不对 / supervisor 需要 Bearer token /
Prometheus 在 Docker 里仍指向 `localhost` 而不是 `host.docker.internal`

### 告警不触发

```bash
curl -s http://localhost:9090/api/v1/rules | \
  jq '.data.groups[].rules[] | select(.type == "alerting") | {name, state}'

curl -s 'http://localhost:9090/api/v1/query?query=mortred_queue_depth' | jq '.data.result'
```

常见原因：表达式标签不匹配 / for 时长未达到 / 告警被静默

### Grafana 面板无数据

```bash
curl -s http://localhost:3000/api/datasources | jq '.[] | {name, type, url}'
```

确保 Prometheus URL 正确，在 Explore 中手动测试查询。

## 手动验证命令汇总

```bash
# 网关
curl -s http://localhost:8080/metrics | head -20
curl -s http://localhost:8080/metrics | grep mortred_http_requests_total

# 模型服务器（在跑模型的那台机器上打环回；不要 -p 这些端口）
curl -s http://localhost:9002/metrics | grep mortred_up
curl -s http://localhost:9002/metrics | grep mortred_queue_depth
curl -s http://localhost:9002/metrics | grep mortred_workers
curl -s http://localhost:9002/metrics | grep mortred_batch_size
curl -s http://localhost:9002/metrics | grep mortred_async

# 监督器
curl -s -H "Authorization: Bearer $MORTRED_API_TOKEN" \
  http://localhost:8787/api/v1/metrics

# Prometheus
curl -s http://localhost:9090/api/v1/targets | \
  jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

curl -s http://localhost:9090/api/v1/alerts | \
  jq '.data.alerts[] | {name: .labels.alertname, state}'

# 手动查询
curl -s 'http://localhost:9090/api/v1/query?query=mortred_up' | jq .
curl -s 'http://localhost:9090/api/v1/query?query=rate(mortred_http_requests_total[5m])' | jq .
```

## 最佳实践

### 抓取间隔与保留

```yaml
global:
  scrape_interval: 15s     # 高流量可降到 5s
  evaluation_interval: 15s
```

```bash
prometheus --storage.tsdb.retention.time=30d  # 保留 30 天
```

### 阈值调优

| 指标 | 初始值 | 调优建议 |
|---|---|---|
| queue_depth | > 20 | 观察 1 周按 P99 调 |
| p95 latency | > 2000ms | 按模型类型分组（分类 < 100ms，检测 < 500ms） |
| 5xx rate | > 5% | 高可靠可降到 1% |
| restart rate | > 0.2/s | 偶尔重启可放宽到 0.1/min |

### 覆盖度检查清单

- [ ] Gateway :8080 抓取正常
- [ ] Supervisor :8787 仅在 Prometheus 配了 Bearer 时抓取
- [ ] 模型 scrape job 只出现在 Prometheus 与模型共享环回的地方
- [ ] Grafana / Prometheus 端口在环回；Grafana 密码不是镜像默认值
- [ ] 12 条告警规则全部加载
- [ ] Grafana 13 面板有数据
- [ ] 配置了至少一个通知渠道
- [ ] 做过一次告警演练（停一个模型验证触发）
