# 分支清理记录（2026-10-19）

- 基线: `sub-agent-co-work @ cd39f537`
- 决策: 用户确认"已合入的都删、两个未合入的也都删"
- 结果: 本地分支 79 → **3**（`main`、`sub-agent-co-work`、`v_llm`），无 worktree 残留

## 1. 已合入的 71 个 `cowork/*`（`git branch -d` 安全删除）

删除前以 `git branch --merged HEAD` 逐个确认祖先关系，`-d` 不会丢弃任何提交。

```
cowork/r1-p0-1-libface
cowork/r1-p1-1-gw-metrics
cowork/r1-p1-2-doc-ids
cowork/r1-p2-1-batch-metrics
cowork/r1-p2-10-zh-next
cowork/r1-p2-2-rate-auth
cowork/r1-p2-3-fork-trace
cowork/r1-p2-4-resident-pair
cowork/r1-p2-5-token-listen
cowork/r1-p2-6-stop-order
cowork/r1-p2-7-size-reject
cowork/r1-p2-9-golden-drift
cowork/r10-p1-1-clip-boxes
cowork/r10-p1-2-cred-scrub
cowork/r10-p1-3-process-group
cowork/r10-p1-4-tensor-contract
cowork/r10-p1-5-timeout-clamp
cowork/r10-p1-6-method-label
cowork/r10-p1-7-stale-tests
cowork/r11-p0-2-doctor-id
cowork/r11-p0-3-yolov7-contract
cowork/r11-p1-10-readme-gpu
cowork/r11-p1-11-mask-overlap
cowork/r11-p1-12-trt-product
cowork/r11-p1-7-openapi-examples
cowork/r11-p1-8-cpu-golden
cowork/r11-p1-9-h3-bench
cowork/r12-d10-gpu-na
cowork/r12-d11-deploy-names
cowork/r12-d12-fork-exec
cowork/r12-d9-auth-quota
cowork/r12-p1-12-trt-product
cowork/r2-p2-12-tutorial-model
cowork/r3-p2-22-tutorials
cowork/r3-p2-23-deployment
cowork/r4-p0-1-tls
cowork/r4-p0-2-rate-limit
cowork/r4-p0-3-occupancy-bypass
cowork/r4-p0-4-gpu-params
cowork/r4-p1-1-rate-semantics
cowork/r4-p1-2-smi-reaper
cowork/r4-p1-3-openapi-slim
cowork/r4-p1-4-inflight-fix
cowork/r4-p1-4-tenant-inflight
cowork/r4-p1-5-semver
cowork/r5-p1-1-rate-headers
cowork/r5-p1-2-gpu-sampler
cowork/r5-p1-3-inflight
cowork/r5-p1-4-version
cowork/r5-p1-5-occ-tests
cowork/r6-p1-1-rate-authority
cowork/r6-p1-2-tenant-quota
cowork/r6-p1-3-gpu-probe
cowork/r6-p1-4-ci-evidence
cowork/r7-p0-1-ci-version
cowork/r7-p0-2-apikey-load
cowork/r7-p1-1-nvml-probe
cowork/r7-p1-2-queue-residency
cowork/r7-p1-3-evidence-ci
cowork/r7-p1-4-drop-fixedwindow
cowork/r8-p0-1-openapi-loader
cowork/r8-p1-1-worker-budget
cowork/r8-p1-2-toml-load
cowork/r8-p1-3-gateway-e2e
cowork/r8-p1-4-nvml-lifetime
cowork/r8-p1-5-ci-evidence
cowork/r9-p0-1-ready
cowork/r9-p0-2-body-limit
cowork/r9-p0-3-receive-timeout
cowork/r9-p0-4-supervisor-log
cowork/r9-p1-worker-budget-cas
```

## 2. 两个未合入的分支（`-D` 强制删除，经用户确认）

| 分支 | 提交 | 提交说明 | 内容 | 现状 |
| --- | --- | --- | --- | --- |
| `cowork/r1-p2-11-http-rps` | `6837bf4f` | Keep the coverage reload path at twice worker_nums. | `scripts/coverage_http_report.py` +358/−76：新增 `--reload`（部署已有行而不重新 calibrate，再跑官方 `--raw`，并发=2×`worker_nums`）。HEAD 已实现"并发=2×worker_nums"（`:335/:562/:1093`）但**没有** `--reload` | 分支已删；提交对象仍存在但**不可达**（`git gc` 会清理） |
| `cowork/r12-d8-gpu-user` | `4a732063` | D-8: run the default GPU image as uid 10001. | `Dockerfile` +10：`mortred-gpu` 阶段加 `USER 10001`（nologin 账户 + 归属 10001 的 logs 目录），把默认镜像从 root 降权。HEAD 的 `Dockerfile` 无 10001 | 同上 |

删除理由：
- `r1-p2-11-http-rps`：其唯一独有功能是 `--reload`，当前覆盖集报告流程已不再依赖它；用户决定「都删」。
- `r12-d8-gpu-user`：Docker USER / 本地 Docker 测试此前已被用户明确排除在本轮范围外。

**恢复方式**（仅在对象尚未被 gc 清理时有效）：

```bash
git branch cowork/r1-p2-11-http-rps 6837bf4f
git branch cowork/r12-d8-gpu-user   4a732063
```

删除后实测：两个 SHA 的 `git cat-file -t` 仍返回 `commit`，但 `git branch --contains` 为空（悬空）。`git gc --prune=now` 之后将不可恢复。

## 3. 保留的分支

| 分支 | 说明 |
| --- | --- |
| `sub-agent-co-work` | 当前工作分支，`cd39f537`；比 `origin/sub-agent-co-work` **ahead 568**，尚未推送 |
| `main` | 跟踪 `origin/main`（`8450262c`） |
| `v_llm` | 跟踪 `origin/v_llm`（`9167bdce`） |
