# 交付队列

- 轮次: 9
- 版本: v1
- 性质: 长期根本修复（非 band-aid）；顺序锁定；无 packing/Docker
- 加权分上限: P0 上限 7.9
- 审计基线: `47b96321`（Round 8 已关闭；Round 9 由用户直接锁定 4 P0）
- 来源: 用户指令 Round 9 cv-cowork — four P0s

## P0

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | `/ready` 以有空闲 worker 当就绪信号，满载时返回 503，过载会把健康实例摘出 LB | 驻留集全部 worker 都在推理时探活（LB / k8s / 外部探针） | `base_server_impl.h`: ready = init && available_approx()>0；set_ready → mortred_ready 变 0 | 压满 worker 期间 GET /ready 仍 200；饱和只用独立 gauge；mortred_ready 不因饱和翻转 | `src/server/base_server_impl.h`；prometheus_metrics set_ready |
| P0-2 | 请求体大小限制在解析之后才生效，且 `/jobs` 完全没有限制 | 超限 body（含 chunked、空 Content-Length） | serve_process 先解析后判 declared_body_exceeds；空 CL 返回 false；/jobs 无检查 | 超限在解析前 413；/jobs 同样；空 CL 与 chunked 也受限 | request_admission.h；async_endpoints.h |
| P0-3 | 整报文无超时，滴流连接占满入站 | 慢速客户端滴流至 max_connections | receive_timeout 沿用库默认 -1 | 半截请求在 conf 时限内关闭；滴流不阻止正常请求 | abstract_server.h；WFServer SERVER_PARAMS_DEFAULT |
| P0-4 | supervisor 日志 I/O 停顿时永久挂死 | 日志盘停摆，或子进程写满 stdout 且忽略 SIGINT | LogBuffer 每次 append flush；handle_exit 在 reaper join 读线程；wait_shutdown 无超时 | 其余子进程仍回收；stop 在 grace 后返回；SIGTERM 后 30s 内退出 | log_buffer.h；supervisor.cpp |

## P1

（无）

## P2

（无）

## 附录

| 类别 | 陈述 |
| --- | --- |
| 用户禁止 | packing / max_batch_size；Docker USER / 本机 Docker 测试 |
| 顺延说明 | 同内容曾记入 round-010 候选；用户本轮指令明确为 Round 9 |

## 确认

| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意 | v1 | 用户指令 2026-10-16 Round 9 锁定（Steps 2-5） |
| cv-pm | 同意 | v1 | 用户直接锁定 |
| cv-tech-lead | 同意 | v1 | 用户直接锁定 |
| cv-cto | 同意 | v1 | 用户直接锁定 |

## 用户决定

- 日期: 2026-10-16
- 选用版本: v1
- 决定原文: Round 9 four P0s; Steps 2-5; --no-ff merges
- 未选用: 无
