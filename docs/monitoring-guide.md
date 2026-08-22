# Monitoring Guide

| [English](monitoring-guide.md) | [中文](monitoring-guide.zh-cn.md) |
|---|---|

Mortred Model Server exposes Prometheus metrics from three components. This guide covers deployment, metric reference, dashboard, and alerting.

## Quick Start

```bash
# 1. Start the monitoring stack (Prometheus + Grafana)
docker compose -f deploy/docker-compose.monitoring.yml up -d

# 2. Open Grafana
# http://localhost:3000  (admin/admin)
#
# Note: on Linux, replace "localhost" with "host.docker.internal" in
# deploy/prometheus.yml targets so the containers can reach the host.
```

## Metric Sources

| Component | Port | Endpoint | Key Metrics |
|---|---|---|---|
| Gateway | :8080 | `/metrics` | `mortred_http_requests_total`, `mortred_http_request_duration_ms` |
| Supervisor | :8787 | `/metrics` | `mortred_supervisor_state`, `mortred_supervisor_restarts_total` |
| Model servers | :9001-9074 | `/metrics` | `mortred_inference_duration_ms`, `mortred_queue_depth`, `mortred_workers_busy`, `mortred_batch_size`, `mortred_async_jobs_total` |

## Metric Reference

### Availability

| Metric | Type | Labels | Description |
|---|---|---|---|
| `mortred_up` | gauge | model | Process alive (1) |
| `mortred_ready` | gauge | model | Workers available (1) |
| `up` | gauge | job | Prometheus scrape target reachability |

### Traffic

| Metric | Type | Labels | Description |
|---|---|---|---|
| `mortred_http_requests_total` | counter | model, method, status | Total HTTP requests |
| `mortred_http_request_duration_ms` | histogram | model, method, status | HTTP request latency |
| `mortred_queue_rejected_total` | counter | model | Requests rejected by queue depth limit (429) |

### Inference

| Metric | Type | Labels | Description |
|---|---|---|---|
| `mortred_inference_duration_ms` | histogram | model | Model inference time |
| `mortred_queue_wait_duration_ms` | histogram | model | Time waiting for a worker |
| `mortred_inference_success_total` | counter | model | Successful inferences |
| `mortred_inference_failure_total` | counter | model | Failed inferences |
| `mortred_workers_busy` | gauge | model | Currently busy workers |
| `mortred_workers_available` | gauge | model | Idle workers |
| `mortred_queue_depth` | gauge | model | Current waiting queue depth |

### Batching

| Metric | Type | Labels | Description |
|---|---|---|---|
| `mortred_batch_size` | histogram | model | Executed batch sizes |
| `mortred_batch_window_wait_ms` | histogram | model | Batch collection window wait time |

### Async Jobs

| Metric | Type | Labels | Description |
|---|---|---|---|
| `mortred_async_jobs_total` | counter | model, state | Async jobs by state (submitted/running/done/failed/timeout) |
| `mortred_async_queue_depth` | gauge | model | Current async job queue depth |
| `mortred_async_job_duration_ms` | histogram | model | Async job execution time |

### Supervisor

| Metric | Type | Labels | Description |
|---|---|---|---|
| `mortred_supervisor_state` | gauge | server | Process state (0=stopped, 1=starting, 2=running, 3=backoff, 4=failed) |
| `mortred_supervisor_ready` | gauge | server | Process readiness |
| `mortred_supervisor_restarts_total` | counter | server | Total restarts |

## Alerting

Alert rules are defined in `deploy/alert-rules.yml`. Key alerts:

| Alert | Severity | Trigger |
|---|---|---|
| GatewayDown | critical | Gateway unreachable 30s |
| SupervisorDown | critical | Supervisor unreachable 60s |
| ModelServerDown | warning | Model server unreachable 60s |
| HighQueueDepth | warning | Queue depth > 20 for 60s |
| OverloadRejections | warning | 429 rate > 0.1/s for 2min |
| HighLatency | warning | p95 > 2000ms for 5min |
| HighErrorRate | critical | 5xx > 5% for 2min |
| AllWorkersBusy | critical | 0 available workers for 2min |
| AsyncQueueFull | warning | Async depth > 10 for 2min |
| RestartStorm | critical | Restarts > 0.2/s for 2min |

## Grafana Dashboard

Import `deploy/grafana-dashboard.json` (Dashboard → Import → Upload JSON). 13 panels:

1. Service Availability (stat)
2. Inference Rate (req/s by model)
3. Error Rate (5xx %)
4. Overload Rejections (429/s)
5. Inference Latency p50/p95/p99
6. Queue Depth
7. Workers (busy/available)
8. Batch Size Distribution
9. Batch Window Wait
10. Async Jobs by State
11. Async Queue Depth
12. Supervisor Process States
13. Process Restart Rate

## Manual Verification

```bash
# Check gateway metrics
curl -s http://localhost:8080/metrics | head -20

# Check model server metrics
curl -s http://localhost:9002/metrics | grep mortred_inference_duration

# Check supervisor metrics
curl -s -H "Authorization: Bearer $MORTRED_API_TOKEN" http://localhost:8787/api/v1/metrics

# Verify Prometheus is scraping
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```
