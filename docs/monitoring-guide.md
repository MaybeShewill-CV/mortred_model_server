# Monitoring Guide

| [English](monitoring-guide.md) | [中文](monitoring-guide.zh-cn.md) |
|---|---|

Complete guide to the Mortred Model Server monitoring lifecycle: deployment, configuration, metric reference, dashboard interpretation, alert response, and troubleshooting. Intended for SRE / DevOps engineers and service administrators.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Data Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   /metrics    ┌──────────────┐               │
│  │ Gateway  │ ───────────→ │              │               │
│  │ (:8080)  │               │  Prometheus  │               │
│  └──────────┘               │  (:9090)     │               │
│                              │  Scrape+Store │               │
│  ┌──────────┐   /metrics    │  +Alert Eval  │──→ Alerting   │
│  │Supervisor│ ───────────→ │              │   (optional)  │
│  │ (:8787)  │               └──────┬───────┘               │
│  └──────────┘                      │                        │
│                              ┌──────▼───────┐               │
│  ┌──────────┐   /metrics    │   Grafana    │               │
│  │  Model   │ ───────────→ │  (:3000)     │               │
│  │ Servers  │               │  Dashboards  │               │
│  │(:9001-74)│               └──────────────┘               │
│  └──────────┘                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
docker compose -f deploy/docker-compose.monitoring.yml up -d
# Grafana: http://localhost:3000 (admin/admin)
# Note: on Linux, replace "localhost" with "host.docker.internal"
# in deploy/prometheus.yml targets.
```

### Option 2: Bare Metal

```bash
sudo apt install prometheus grafana
sudo cp deploy/prometheus.yml /etc/prometheus/prometheus.yml
sudo cp deploy/alert-rules.yml /etc/prometheus/alert-rules.yml
sudo systemctl restart prometheus
```

### Option 3: Standalone Prometheus

```bash
prometheus --config.file=deploy/prometheus.yml --storage.tsdb.path=/tmp/prom-data
```

## Metric Sources

| Component | Port | Endpoint | Description |
|---|---|---|---|
| Gateway | :8080 | `/metrics` | Inference traffic entry point |
| Supervisor | :8787 | `/api/v1/metrics` | Process management (requires token) |
| Model servers | :9001-9074 | `/metrics` | One per running model |

### Adding a New Model Scrape Target

```yaml
  - job_name: mortred-model-yolov5
    static_configs:
      - targets: ['localhost:9053']
        labels: { component: model, model: yolov5 }
```

## Metric Reference

### Availability (3) — "Is the service alive?"

| Metric | Type | Description |
|---|---|---|
| `mortred_up` | gauge | Process alive (1=up) |
| `mortred_ready` | gauge | Workers available (1=ready) |
| `up` | gauge | Prometheus can scrape (1=reachable) |

> **`mortred_up` vs `up`**: the former is self-reported by the process; the latter is probed externally by Prometheus. `up == 0` is the more reliable downtime signal.

### Traffic (3) — "How many requests?"

| Metric | Type | Labels | Description |
|---|---|---|---|
| `mortred_http_requests_total` | counter | model, method, status | Cumulative HTTP requests |
| `mortred_http_request_duration_ms` | histogram | model, method, status | Request latency (ms) |
| `mortred_queue_rejected_total` | counter | model | Cumulative 429 rejections |

```promql
# Requests per second
sum(rate(mortred_http_requests_total[5m]))

# Per-model p95 latency
histogram_quantile(0.95, sum(rate(mortred_http_request_duration_ms_bucket[5m])) by (le, model))

# 5xx error rate
100 * sum(rate(mortred_http_requests_total{status=~"5.."}[5m])) by (model)
  / sum(rate(mortred_http_requests_total[5m])) by (model)
```

### Inference (7) — "Model performance"

| Metric | Type | Description |
|---|---|---|
| `mortred_inference_duration_ms` | histogram | Model inference time (excl. queuing) |
| `mortred_queue_wait_duration_ms` | histogram | Worker wait time |
| `mortred_inference_success_total` | counter | Successful inferences |
| `mortred_inference_failure_total` | counter | Failed inferences |
| `mortred_workers_busy` | gauge | Busy workers |
| `mortred_workers_available` | gauge | Idle workers |
| `mortred_queue_depth` | gauge | Current queue depth |

> **Diagnosis**: when p95 latency is high, check queue_depth first — deep queue → capacity issue (add workers); shallow queue but slow → model/GPU issue.

### Batching (2) — "Is batching effective?"

| Metric | Type | Description |
|---|---|---|
| `mortred_batch_size` | histogram | Actual executed batch sizes |
| `mortred_batch_window_wait_ms` | histogram | Batch collection window wait |

```promql
# Average batch size (> 1.5 = effective)
sum(mortred_batch_size_sum) by (model) / sum(mortred_batch_size_count) by (model)
```

### Async Jobs (3) — "Long-running task status"

| Metric | Type | Description |
|---|---|---|
| `mortred_async_jobs_total` | counter | Jobs by state (submitted/running/done/failed/timeout) |
| `mortred_async_queue_depth` | gauge | Async queue depth |
| `mortred_async_job_duration_ms` | histogram | Async job execution time |

### Supervisor (3) — "Process management"

| Metric | Type | Description |
|---|---|---|
| `mortred_supervisor_state` | gauge | Process state code |
| `mortred_supervisor_ready` | gauge | Process ready (1) |
| `mortred_supervisor_restarts_total` | counter | Total restarts |

**State codes**:

| Value | State | Meaning |
|---:|---|---|
| 0 | stopped | Manually stopped or init failed |
| 1 | starting | Waiting for readiness probe |
| 2 | running | Normal operation |
| 3 | backoff | Waiting to restart |
| 4 | failed | Crash loop exhausted (needs manual restart) |

## Alert Reference

### Alert Overview (12 rules × 5 groups)

#### Availability (4)

| Alert | Severity | Trigger | Meaning | Response |
|---|---|---|---|---|
| GatewayDown | critical | Gateway unreachable 30s | All inference blocked | Check gateway process |
| SupervisorDown | critical | Supervisor unreachable 60s | Process management lost | Check supervisor |
| ModelServerDown | warning | Model unreachable 60s | Single model down | Check model logs |
| ModelNotReady | warning | Ready probe fails 2min | Process up but unusable | Check model loading |

#### Performance (5)

| Alert | Severity | Trigger | Meaning | Response |
|---|---|---|---|---|
| HighQueueDepth | warning | Depth > 20 for 60s | Long queue | Add workers or scale |
| OverloadRejections | warning | 429 > 0.1/s for 2min | Rejecting requests | Increase max_queue_depth |
| HighLatency | warning | p95 > 2000ms for 5min | Slowing down | Check GPU / model |
| HighErrorRate | critical | 5xx > 5% for 2min | Mass failures | Check model logs |
| AllWorkersBusy | critical | 0 idle workers 2min | Possible stuck | Check stuck-worker |

#### Async (2)

| Alert | Severity | Trigger | Response |
|---|---|---|---|
| AsyncQueueFull | warning | Async depth > 10 for 2min | Check async workers |
| AsyncTimeouts | warning | Timeouts > 0.05/s | Check async_timeout |

#### Batching (1)

| Alert | Severity | Trigger | Response |
|---|---|---|---|
| BatchNotCoalescing | info | Avg batch < 1.5 for 10min | Check concurrency and delay |

#### Supervisor (1)

| Alert | Severity | Trigger | Response |
|---|---|---|---|
| RestartStorm | critical | Restarts > 0.2/s for 2min | Investigate crash cause |

### Severity Levels

| Severity | Meaning | Notification | Response Time |
|---|---|---|---|
| critical | Service unavailable | Phone/SMS | < 5 minutes |
| warning | Performance degraded | Slack/Email | < 30 minutes |
| info | Optimization hint | Dashboard | Next review |

### Custom Alerts

```yaml
- alert: MortredCustomGPUHigh
  expr: your_gpu_metric > 90
  for: 300s
  labels: { severity: warning }
  annotations:
    summary: "GPU utilization too high"
```

```bash
curl -X POST http://localhost:9090/-/reload  # hot reload
```

## Grafana Dashboard

### Import

Dashboard → Import → Upload `deploy/grafana-dashboard.json` → Select Prometheus datasource → Import

### Panel Interpretation

#### Row 1: Global Overview

| # | Panel | What to Look For |
|---|---|---|
| 1 | Service Availability | All green = healthy; red = something down |
| 2 | Inference Rate | Traffic trend; sudden drop = anomaly |
| 3 | Error Rate | Normal < 1%; spike = model issue |
| 4 | Overload Rejections | Non-zero = overload starting |

#### Row 2: Performance

| # | Panel | What to Look For |
|---|---|---|
| 5 | Latency p50/p95/p99 | p99 spikes = individual slow requests |
| 6 | Queue Depth | Rising = capacity shortfall |
| 7 | Workers | busy == total → saturated |

#### Row 3: Batching & Async

| # | Panel | What to Look For |
|---|---|---|
| 8 | Batch Size | Average > 1.5 = effective |
| 9 | Batch Window Wait | High → delay too large |
| 10 | Async Jobs | done rising = normal; timeout rising = problem |

#### Row 4: System Health

| # | Panel | What to Look For |
|---|---|---|
| 11 | Async Queue Depth | > 10 = backlog |
| 12 | Process States | 2=green(running), 4=red(failed) |
| 13 | Restart Rate | Non-zero sustained = crash loop |

## Troubleshooting

### Prometheus Scrape Failures

```bash
curl -s http://localhost:9090/api/v1/targets | \
  jq '.data.activeTargets[] | select(.health != "up") | {job: .labels.job, error: .lastError}'
```

Common causes: model not running / wrong port / Docker networking (use host.docker.internal)

### Missing Metrics

```bash
curl -s http://localhost:8080/metrics | head -5
```

Common causes: process not running / wrong port / supervisor requires Bearer token

### Alerts Not Firing

```bash
curl -s http://localhost:9090/api/v1/rules | \
  jq '.data.groups[].rules[] | select(.type == "alerting") | {name, state}'

curl -s 'http://localhost:9090/api/v1/query?query=mortred_queue_depth' | jq '.data.result'
```

Common causes: expression labels don't match / `for` duration not reached / alert silenced

### Grafana Panels Empty

```bash
curl -s http://localhost:3000/api/datasources | jq '.[] | {name, type, url}'
```

Ensure Prometheus URL is correct; test manually in Explore.

## Manual Verification Commands

```bash
# Gateway
curl -s http://localhost:8080/metrics | head -20
curl -s http://localhost:8080/metrics | grep mortred_http_requests_total

# Model server
curl -s http://localhost:9002/metrics | grep mortred_up
curl -s http://localhost:9002/metrics | grep mortred_queue_depth
curl -s http://localhost:9002/metrics | grep mortred_workers
curl -s http://localhost:9002/metrics | grep mortred_batch_size
curl -s http://localhost:9002/metrics | grep mortred_async

# Supervisor
curl -s -H "Authorization: Bearer $MORTRED_API_TOKEN" \
  http://localhost:8787/api/v1/metrics

# Prometheus
curl -s http://localhost:9090/api/v1/targets | \
  jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

curl -s http://localhost:9090/api/v1/alerts | \
  jq '.data.alerts[] | {name: .labels.alertname, state}'

# Manual queries
curl -s 'http://localhost:9090/api/v1/query?query=mortred_up' | jq .
curl -s 'http://localhost:9090/api/v1/query?query=rate(mortred_http_requests_total[5m])' | jq .
```

## Best Practices

### Scrape Interval and Retention

```yaml
global:
  scrape_interval: 15s     # can reduce to 5s for high-traffic
  evaluation_interval: 15s
```

```bash
prometheus --storage.tsdb.retention.time=30d  # retain 30 days
```

### Threshold Tuning

| Metric | Initial | Tuning Advice |
|---|---|---|
| queue_depth | > 20 | Observe 1 week, then adjust to P99 |
| p95 latency | > 2000ms | Group by model type (classification < 100ms, detection < 500ms) |
| 5xx rate | > 5% | Reduce to 1% for high-reliability |
| restart rate | > 0.2/s | Relax to 0.1/min for occasional restarts |

### Coverage Checklist

- [ ] Gateway :8080 scraping OK
- [ ] Supervisor :8787 scraping OK
- [ ] Every running model has a scrape job
- [ ] All 12 alert rules loaded
- [ ] Grafana 13 panels showing data
- [ ] At least one notification channel configured
- [ ] One alert drill completed (stop a model, verify alert fires)
