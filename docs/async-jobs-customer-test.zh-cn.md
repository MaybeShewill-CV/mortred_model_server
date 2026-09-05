# 长任务 `/jobs` 客户验收手册

| [English](async-jobs-customer-test.md) | [中文](async-jobs-customer-test.zh-cn.md) |
|---|---|

本文是异步推理（`POST /jobs` → 轮询 / 长等 → 取结果）的 **验收步骤**。用它证明
长任务符合公开契约，而不是把 `/jobs` 当成阻塞的 `/infer`。

契约：[api-contract.zh-cn.md](api-contract.zh-cn.md)「异步任务」。
实现细节：[async-job-table.zh-cn.md](async-job-table.zh-cn.md)。

---

## 1. 何谓「符合预期」

| 步骤 | 预期 |
|---|---|
| 提交 | `POST …/jobs` 在 **毫秒级** 返回 HTTP **202**，此时模型往往还有数秒到数分钟的计算 |
| 响应体 | JSON `{"job_id","state":"pending","poll_url","result_url"}`，并带 `Location` |
| 提交后立刻 | 轮询为 `pending` 或 `running`，**不是** `done`；`GET …/result` 为 **409** |
| 完成 | 轮询最终变为 `done`（或 `failed` / `timeout`）；只有 `done` 时 `GET …/result` 才是 **200** 统一信封 |
| 长等 | `GET …/wait` 在 job **进入终态**，或 wait 预算耗尽时返回（耗尽时 state 仍可能是 `pending`/`running`） |
| 队列满 | `pending`+`running` 达到 `async_max_queue` 时，再提交返回 **429** |

`202` 表示 **已准入**，不是 **已算完**。如果 `POST /jobs` 的耗时和同一 payload
的 `POST …/infer` 差不多，则 **没有** 满足本契约。

---

## 2. 前置条件

1. 控制面已起来（网关 `:8080`，supervisor `:8787`）。模型端口只绑环回；**全部测试走网关**，除非运维明确允许例外。
2. 一个 **已开启异步** 且 **ready** 的模型。仓库里默认 `async_enabled=true` 的有：

   | Catalog id | 典型耗时 |
   |---|---|
   | `DDPM` | 扩散采样（数秒～数分钟） |
   | `DDIM` | 扩散采样 |
   | `CLS_COND_DDIM` | 类别条件扩散 |
   | `LDM` | 潜空间扩散 |
   | `SAM_AMG` | SAM 自动出 mask |

   分类 / 检测（`MOBILENETV2`、`YOLOV8` 等）默认关闭异步。
   `POST /v1/models/MOBILENETV2/jobs` 必须是 **404**。
3. 请求体与成功的 `/infer` 相同（`images` 为 base64 数组）。`/infer` 都过不了的 body，
   `/jobs` 同样会在准入前拒绝（多为 422）——那不是 202 时延问题。
4. 工具：`curl`、`python3`（或 `jq`）。用 `time_total` 看墙钟。

---

## 3. 环境变量

Token 换成运维发放的值。网关推理用 `MORTRED_GATEWAY_AUTH_TOKEN`（或
`conf/api_keys.toml` 里的 key）。Supervisor 目录用 `MORTRED_API_TOKEN`。

```bash
export GW="http://127.0.0.1:8080"
export SUP="http://127.0.0.1:8787"
export TOKEN="__GATEWAY_BEARER__"
export MGMT="__SUPERVISOR_BEARER__"
export MODEL="DDPM"   # 或 DDIM / LDM / SAM_AMG / …

# 与成功的 POST /v1/models/$MODEL/infer 相同的 payload
IMG_B64="$(base64 < /path/to/valid-input.png | tr -d '\n')"   # macOS: base64 -i file
export BODY=$(python3 -c "import json,os; print(json.dumps({'images':[os.environ['IMG_B64']],'req_id':'jobs-test-1'}))")
```

Linux 可用 `base64 -w0 file`。Docker compose 把网关发在 `127.0.0.1:8080`
（见 `docker-compose.yml`）。

确认进程：

```bash
curl -sS -o /dev/null -w "%{http_code}\n" "$GW/healthz"
# 期望 200

curl -sS -H "Authorization: Bearer $MGMT" "$SUP/api/v1/catalog" | python3 -m json.tool
# 应能看到 $MODEL

curl -sS -H "Authorization: Bearer $MGMT" "$SUP/api/v1/status" | python3 -m json.tool
# 该 id 应为 ready / running，不能是 start_failed
```

---

## 4. 不要和超时打架

| 旋钮 | 默认 | 作用 |
|---|---|---|
| `GET …/wait?timeout=N` | N 的单位是 **毫秒**；默认 30000；上限 300000 | **这一次 HTTP 请求** 最多挂多久 |
| curl `--max-time` | 无 | 必须 **大于** 你传入的 wait 预算 |
| 网关 `upstream_recv_timeout_ms` | `conf/mortred.toml` 里 180000（180 秒） | 若 N 大于它，网关会把挂起的 wait 打成 **502** |
| 模型 `peer_resp_timeout` | 扩散 / SAM AMG 的 toml 为 600 秒 | Workflow 对端 I/O 超时 |
| 模型 `async_timeout` | 那些 toml 为 600000 ms | job 预算（等 worker + 推理）。终态 `timeout` 不是 `done` |
| 模型 `async_max_queue` | 那些 toml 为 8 | 准入深度（`pending` + `running`） |

规则：**wait 的 `timeout`（毫秒）< curl `--max-time`（秒）× 1000**，且
**wait 超时 < `upstream_recv_timeout_ms`**。例如 `timeout=120000` 需要
`--max-time 180`，且网关 recv 超时 ≥ 120 秒。

---

## 5. 测试用例

第一次请按顺序做。之后任一用例都可单独回归。

### T1 — 对照：阻塞 infer（记录真实耗时）

这 **不是** 异步 API。它告诉你这个模型到底要跑多久。

```bash
curl -sS -o /tmp/infer.json -w "infer http=%{http_code} time=%{time_total}s\n" \
  --max-time 600 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/infer"
```

**通过：** HTTP 200（payload 不对时也可能是文档里的其它模型错误——先把 payload
修好再继续）。记下 `time_total` 为 `T_infer`。扩散模型常常是 **数秒以上**。

**失败：** 401（token 错）、404（catalog id 错）、503（模型没起来）。先修环境，不要做 T2。

### T2 — 提交立刻 202（主契约）

```bash
curl -sS -D /tmp/jobs.hdr -o /tmp/jobs.json -w "jobs http=%{http_code} time=%{time_total}s\n" \
  --max-time 30 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
cat /tmp/jobs.hdr
python3 -m json.tool < /tmp/jobs.json
```

**通过（全部满足）：**

1. HTTP **202**。
2. `time_total` **远小于 `T_infer`**（通常 < 0.5 s，且一定远小于采样时间）。
   若 `T_infer` 是 20 s 而 jobs 花了 19 s，判 **失败**。
3. Body 含 `job_id`（`job_` 前缀）、`"state":"pending"`、`poll_url`、`result_url`。
4. `Location` 是 `/v1/models/<id>/jobs/<job_id>`（网关改写），**不是** 模型端口上的 `/jobs/<job_id>`。
5. `poll_url` / `result_url` 同样在 `/v1/models/<id>/jobs/…` 下。

保存 id：

```bash
export JOB=$(python3 -c "import json; print(json.load(open('/tmp/jobs.json'))['job_id'])")
export POLL=$(python3 -c "import json; print(json.load(open('/tmp/jobs.json'))['poll_url'])")
export RESULT=$(python3 -c "import json; print(json.load(open('/tmp/jobs.json'))['result_url'])")
echo "JOB=$JOB POLL=$POLL RESULT=$RESULT"
```

### T3 — 提交后立刻：未完成，result 为 409

T2 之后 **不要 sleep**，立刻执行：

```bash
curl -sS -w "\npoll http=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  "$GW$POLL"
echo
curl -sS -w "\nresult http=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  "$GW$RESULT"
```

**通过：**

- 轮询 HTTP 200，`"state"` 为 `"pending"` 或 `"running"`，在 `T_infer` 仍是数秒时
  **绝不是** `"done"`。
- result HTTP **409**，body 含 `"error"`，并带当前状态
  （`job not finished (state: pending)` 或 `running`）。

**失败：** 刚拿到 202 立刻 result 200 且带完整信封（旧缺陷：202 要等 `run_items`
跑完才刷出）。

### T4 — 轮询直到终态

```bash
python3 - <<'PY'
import json, os, time, urllib.request
gw, token, poll = os.environ["GW"], os.environ["TOKEN"], os.environ["POLL"]
deadline = time.time() + 600
while time.time() < deadline:
    req = urllib.request.Request(gw + poll, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read().decode())
    print(time.strftime("%H:%M:%S"), body.get("state"), "elapsed_ms=", body.get("elapsed_ms"))
    if body.get("state") in ("done", "failed", "timeout"):
        break
    time.sleep(0.5)
else:
    raise SystemExit("job did not reach a terminal state in 600s")
open("/tmp/jobs-last-poll.json","w").write(json.dumps(body, indent=2))
PY
```

**通过：** state 变为 `done`（正常路径）或 `failed` / `timeout`（也是合法终态；
带着这个认知做 T5/T6）。运行期间 `elapsed_ms` 会增加。

**失败：** 一直 `pending`；404（重启会丢掉内存账本）；401。

### T5 — 仅在 `done` 时取结果

```bash
curl -sS -w "\nresult http=%{http_code} time=%{time_total}s\n" \
  --max-time 30 \
  -H "Authorization: Bearer $TOKEN" \
  "$GW$RESULT" | tee /tmp/jobs-result.json
```

**若轮询是 `done`：** HTTP 200，统一信封（`status`、`results` …）。再 GET 一次：
**仍然 200**（TTL 内可重复读）。

**若轮询是 `failed` 或 `timeout`：** HTTP **409**（只有 `done` 才给 result）。
错误信息在 poll/wait 的 `"error"` 里。

### T6 — wait 等到终态（长轮询）

再提交一个 **新** job，让 wait 真正挂住：

```bash
curl -sS -o /tmp/jobs2.json -w "submit http=%{http_code} time=%{time_total}s\n" \
  --max-time 30 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
export JOB2=$(python3 -c "import json; print(json.load(open('/tmp/jobs2.json'))['job_id'])")

curl -sS -o /tmp/wait.json -w "wait http=%{http_code} time=%{time_total}s\n" \
  --max-time 180 \
  -H "Authorization: Bearer $TOKEN" \
  "$GW/v1/models/$MODEL/jobs/$JOB2/wait?timeout=120000"
python3 -m json.tool < /tmp/wait.json
```

**通过：**

- 提交的 `time_total` 仍远小于 `T_infer`。
- wait HTTP 200，`"state":"done"`（或 `failed`/`timeout`），且 wait 的
  `time_total` 大约是剩余运行时间，**不是** 满满 120 s，也 **不是** 在 job
  未结束时接近 0 s。
- wait **不会** 把 `"running"` 当成「已完成」。若只看到 `running`，说明 timeout
  太短（见 T7）或任务还在跑。

### T7 — wait 预算耗尽（仍非终态）

新 job，wait 预算远小于 `T_infer`：

```bash
curl -sS -o /tmp/jobs3.json -w "submit http=%{http_code} time=%{time_total}s\n" \
  --max-time 30 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
export JOB3=$(python3 -c "import json; print(json.load(open('/tmp/jobs3.json'))['job_id'])")

curl -sS -o /tmp/wait-short.json -w "wait http=%{http_code} time=%{time_total}s\n" \
  --max-time 10 \
  -H "Authorization: Bearer $TOKEN" \
  "$GW/v1/models/$MODEL/jobs/$JOB3/wait?timeout=200"
python3 -m json.tool < /tmp/wait-short.json
```

**通过：** 约 **200 ms** 内 HTTP 200（不是 `T_infer`）。`"state"` 为 `pending`
或 `running`，不是 `done`。这 **不是** job 失败；再 wait 或改为 poll 即可。

`timeout` 单位是 **毫秒**。`timeout=30` 是 30 毫秒，不是 30 秒。

### T8 — 两个客户端同时 wait 同一 job

```bash
curl -sS -o /tmp/jobs4.json \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
export JOB4=$(python3 -c "import json; print(json.load(open('/tmp/jobs4.json'))['job_id'])")

curl -sS -o /tmp/w-a.json -w "A %{http_code} %{time_total}s\n" \
  --max-time 180 -H "Authorization: Bearer $TOKEN" \
  "$GW/v1/models/$MODEL/jobs/$JOB4/wait?timeout=120000" &
curl -sS -o /tmp/w-b.json -w "B %{http_code} %{time_total}s\n" \
  --max-time 180 -H "Authorization: Bearer $TOKEN" \
  "$GW/v1/models/$MODEL/jobs/$JOB4/wait?timeout=120000" &
wait
python3 -m json.tool < /tmp/w-a.json
python3 -m json.tool < /tmp/w-b.json
```

**通过：** 两次都是 HTTP 200，终态 `state` 相同。

### T9 — 队列满时准入 429

`async_max_queue` 计的是 **pending + running**（git 里这些 toml 默认 8）。
在任务还没跑完时，连续提交超过该上限：

```bash
python3 - <<'PY'
import json, os, urllib.request
gw, token, model, body = os.environ["GW"], os.environ["TOKEN"], os.environ["MODEL"], os.environ["BODY"].encode()
url = f"{gw}/v1/models/{model}/jobs"
codes = []
for i in range(12):
    req = urllib.request.Request(url, data=body, method="POST",
        headers={"Authorization": "Bearer " + token,
                 "Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            codes.append(r.status)
    except urllib.error.HTTPError as e:
        codes.append(e.code)
print(codes)
print("202", codes.count(202), "429", codes.count(429))
if codes.count(429) < 1:
    raise SystemExit("expected at least one 429; queue did not fill (jobs too fast or max_queue huge)")
PY
```

**通过：** 同时出现 202 与 429；429 的 body 含 `"error"`，文案带 `async queue full`。
**串行** 循环提交（不必只靠两个并行线程）在前面的 job 仍在跑时 **必须** 能打出 429。

若全部是 202：要么模型快于你 POST 的速度，要么 `async_max_queue` 大于 12 ——
加大循环次数，或核对 server toml。

### T10 — 鉴权：缺 / 错 Bearer 为 401

```bash
curl -sS -o /tmp/unauth.json -w "http=%{http_code}\n" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
# 期望 401

curl -sS -o /tmp/bad.json -w "http=%{http_code}\n" \
  -H "Authorization: Bearer definitely-wrong" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
# 期望 401
```

`/healthz` 无 token 仍是 200。`/jobs` 不会。

### T11 — 未知 job 为 404

```bash
curl -sS -w "\nhttp=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  "$GW/v1/models/$MODEL/jobs/job_does_not_exist"
```

**通过：** HTTP 404，`"error"` 含 `job not found`。对该 id 的 `…/wait`、
`…/result` 同样 404。

### T12 — 未开异步为 404（反例）

选一个 **正在运行且未开异步** 的 catalog id（例如 CPU pack 里的 `MOBILENETV2`）：

```bash
curl -sS -w "\nhttp=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/MOBILENETV2/jobs"
```

**通过：** HTTP **404**（上游 `async_enabled=false`；网关原样透传）。不要当成网关路由故障。

### T13 — 错误方法为 405

```bash
curl -sS -w "\nhttp=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  -X POST \
  "$GW/v1/models/$MODEL/jobs/$JOB"
```

**通过：** HTTP 405。

### T14 — 重启丢掉进行中的 job

```bash
# 先提交，再重启该模型（管理 token）
curl -sS -o /tmp/jobs-rst.json \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
export JOBR=$(python3 -c "import json; print(json.load(open('/tmp/jobs-rst.json'))['job_id'])")

curl -sS -H "Authorization: Bearer $MGMT" -X POST \
  "$SUP/api/v1/servers/$MODEL/restart"

# 等 ready 之后：
curl -sS -w "\nhttp=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  "$GW/v1/models/$MODEL/jobs/$JOBR"
```

**通过：** 重启后 404。账本在内存里，这是预期，不是数据面 bug。客户端必须重提。

### T15 — 网关路径 vs 模型端口（仅运维）

模型端口不应对外。若你在宿主机上且运维允许环回探测：

| 网关 | 模型进程上游 |
|---|---|
| `POST /v1/models/{id}/jobs` | `POST /jobs` |
| `GET /v1/models/{id}/jobs/{job}` | `GET /jobs/{job}` |
| `GET /v1/models/{id}/jobs/{job}/wait?timeout=N` | `GET /jobs/{job}/wait?timeout=N` |
| `GET /v1/models/{id}/jobs/{job}/result` | `GET /jobs/{job}/result` |

`:8080` 上光秃秃的 `/jobs` **不是** 公开路由；请走 catalog 前缀（遗留 `{server_uri}` 只用于同步 infer）。

### T16 — 坏信封是 422，不是卡住的 202

```bash
curl -sS -w "\nhttp=%{http_code} time=%{time_total}s\n" \
  --max-time 10 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"not":"the envelope"}' \
  "$GW/v1/models/$MODEL/jobs"
```

**通过：** 毫秒级 HTTP 422（或 400）。没有 `job_id`。非法输入不得占用 `async_max_queue`。

---

## 6. 指标（可选）

**模型进程** 的 `GET /metrics` 需要进程自己的 auth token（supervisor 注入的
`MORTRED_AUTH_TOKEN`，不是网关推理 token）。运维可看：

- `mortred_async_jobs_total{state="submitted|running|done|failed|timeout"}`
- `mortred_async_job_duration_ms`

T2+T4 成功路径会依次增加 `submitted`、`running`、`done`。
网关 `GET /metrics` 是另一把 token（`MORTRED_METRICS_TOKEN`）。

---

## 7. 签字清单

把下面复制进测试报告。每一行都是 **通过/失败**。

- [ ] T1：`/infer` 可用；已记录 `T_infer`
- [ ] T2：`POST /jobs` 为 202；墙钟 << `T_infer`；`Location` / URL 已改写
- [ ] T3：立刻 poll 不是 `done`；`/result` 为 409
- [ ] T4：poll 到达终态
- [ ] T5：仅 `done` 时 `/result` 为 200；可重复读
- [ ] T6：`/wait` 在终态返回，且不会坐满整个预算
- [ ] T7：短 `timeout=200` 约 200 ms 返回 `pending`/`running`
- [ ] T8：两个 waiter 都完成
- [ ] T9：任务在飞时至少一次 429
- [ ] T10：缺/错 token → 401
- [ ] T11：未知 id → 404
- [ ] T12：未开异步的模型 → 404
- [ ] T13：对 job id 发 POST → 405
- [ ] T14：重启后旧 id → 404
- [ ] T16：坏 JSON 信封被快速拒绝（没有假 202）

若 T1 通过但 T2、T3 失败，说明长任务路径仍把 runner 堵在 HTTP series 上
（本版本修复的缺陷）。把 `/tmp/jobs.hdr`、`/tmp/jobs.json`、curl 的
`time_total` 和 `T_infer` 交给运维。

---

## 8. 服务端旋钮（运维）

`conf/server/**/*_server_config.toml`：

```toml
async_enabled=true          # 默认 false；为 false 时 /jobs 是 404
async_timeout=600000        # job 预算（毫秒）；0 = 不限制
async_max_queue=8           # pending + running 准入上限
async_job_ttl=300000        # 终态 job 保留给 poll/result 的毫秒数
async_max_completed=100     # 终态 job 的 LRU 上限
```

改这些需要重启模型进程（`POST /api/v1/servers/{id}/restart`）。
不要在没看 GPU 显存的情况下盲目加大 `async_max_queue`：已准入的 job 仍和同步
`/infer` 共用同一 worker 池。
