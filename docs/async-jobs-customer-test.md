# Long-task `/jobs` customer test guide

| [English](async-jobs-customer-test.md) | [中文](async-jobs-customer-test.zh-cn.md) |
|---|---|

This is the **acceptance procedure** for asynchronous inference (`POST /jobs` →
poll / wait → result). Use it to prove a long task behaves as the public
contract describes, not as a blocking `/infer`.

Wire contract: [api-contract.md](api-contract.md) § Async jobs.
Internals: [async-job-table.md](async-job-table.md).

---

## 1. What “correct” looks like

| Step | Expected |
|---|---|
| Submit | `POST …/jobs` returns **HTTP 202 in milliseconds**, while the model still has seconds/minutes of work left |
| Body | JSON `{"job_id","state":"pending","poll_url","result_url"}` plus `Location` |
| Immediately after submit | `GET` poll is `pending` or `running`, **not** `done`; `GET …/result` is **409** |
| Completion | Poll eventually shows `done` (or `failed` / `timeout`); only then `GET …/result` is **200** with the unified envelope |
| Wait | `GET …/wait` returns when the job is **terminal**, or when the wait budget expires (state may still be `pending`/`running`) |
| Full queue | Extra submits return **429** while `pending`+`running` count is at `async_max_queue` |

`202` means **admitted**, not **finished**. If `POST /jobs` takes about as long
as `POST …/infer` on the same payload, the deployment is **not** meeting this
contract.

---

## 2. Prerequisites

1. A running Mortred control plane (gateway `:8080`, supervisor `:8787`). Model
   ports are loopback-only; **do all tests through the gateway** unless the
   operator has given you an explicit exception.
2. An **async-enabled** model that is **ready**. Git defaults with
   `async_enabled=true`:

   | Catalog id | Typical work |
   |---|---|
   | `DDPM` | diffusion sampling (seconds–minutes) |
   | `DDIM` | diffusion sampling |
   | `CLS_COND_DDIM` | class-conditional diffusion |
   | `LDM` | latent diffusion |
   | `SAM_AMG` | SAM automatic mask generation |

   Classification / detection servers (`MOBILENETV2`, `YOLOV8`, …) keep
   `async_enabled` off. `POST /v1/models/MOBILENETV2/jobs` must be **404**.
3. The same JSON envelope you already use for `/infer` (`images` is a base64
   array). If `/infer` rejects the body, `/jobs` will reject it too (usually
   422) **before** admission — that is not a 202-timing failure.
4. Tools: `curl`, `python3` (or `jq`). `date`/`time` for wall-clock checks.

---

## 3. Environment

Replace tokens with the values the operator issued. Gateway inference uses
`MORTRED_GATEWAY_AUTH_TOKEN` (or a key from `conf/api_keys.toml`). Supervisor
catalog uses `MORTRED_API_TOKEN`.

```bash
export GW="http://127.0.0.1:8080"
export SUP="http://127.0.0.1:8787"
export TOKEN="__GATEWAY_BEARER__"
export MGMT="__SUPERVISOR_BEARER__"
export MODEL="DDPM"   # or DDIM / LDM / SAM_AMG / …

# Same payload as a successful POST /v1/models/$MODEL/infer
IMG_B64="$(base64 < /path/to/valid-input.png | tr -d '\n')"   # macOS: base64 -i file
export BODY=$(python3 -c "import json,os; print(json.dumps({'images':[os.environ['IMG_B64']],'req_id':'jobs-test-1'}))")
```

On Linux `base64 -w0 file` avoids the `tr`. Docker compose publishes gateway as
`127.0.0.1:8080` (see `docker-compose.yml`).

Confirm the process is up:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" "$GW/healthz"
# expect 200

curl -sS -H "Authorization: Bearer $MGMT" "$SUP/api/v1/catalog" | python3 -m json.tool
# $MODEL should be present

curl -sS -H "Authorization: Bearer $MGMT" "$SUP/api/v1/status" | python3 -m json.tool
# that id should be ready / running, not start_failed
```

---

## 4. Timeouts you must not fight

| Knob | Default | Role |
|---|---|---|
| `GET …/wait?timeout=N` | N is **milliseconds**; default 30000; cap 300000 | How long **this HTTP request** may hang |
| curl `--max-time` | none | Must be **greater** than the wait budget you pass |
| gateway `upstream_recv_timeout_ms` | 180000 (180 s) in `conf/mortred.toml` | Gateway aborts a hung upstream wait with **502** if N is larger than this |
| model `peer_resp_timeout` | 600 s on diffusion / SAM AMG server tomls | Workflow peer I/O timeout |
| model `async_timeout` | 600000 ms on those tomls | Job budget (worker wait + run). Terminal `timeout` is not `done` |
| model `async_max_queue` | 8 on those tomls | Admission depth (`pending` + `running`) |

Rule: **wait `timeout` (ms) < curl `--max-time` (s) × 1000**, and **wait
timeout < `upstream_recv_timeout_ms`**. Example: `timeout=120000` needs
`--max-time 180` and a gateway recv timeout ≥ 120 s.

---

## 5. Test cases

Run them in order the first time. After that, any single case is enough to
re-check a regression.

### T1 — Control: blocking infer (baseline duration)

This is **not** the async API. It tells you how long the model actually takes.

```bash
curl -sS -o /tmp/infer.json -w "infer http=%{http_code} time=%{time_total}s\n" \
  --max-time 600 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/infer"
```

**Pass:** HTTP 200 (or another documented model error if the payload is wrong —
fix the payload before continuing). Record `time_total` as `T_infer`. For
diffusion this is often **several seconds or more**.

**Fail:** 401 (wrong token), 404 (bad catalog id), 503 (model not running). Fix
the environment; do not proceed to T2.

### T2 — Submit returns 202 immediately (the main contract)

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

**Pass (all of these):**

1. HTTP **202**.
2. `time_total` is **much smaller than `T_infer`** (typically < 0.5 s, always
   well under the sampling time). If `T_infer` is 20 s and jobs took 19 s, **fail**.
3. Body has `job_id` (prefix `job_`), `"state":"pending"`, `poll_url`, `result_url`.
4. `Location` header is `/v1/models/<id>/jobs/<job_id>` (gateway rewrite), **not**
   the bare `/jobs/<job_id>` model-port path.
5. `poll_url` / `result_url` are also under `/v1/models/<id>/jobs/…`.

Save the id:

```bash
export JOB=$(python3 -c "import json; print(json.load(open('/tmp/jobs.json'))['job_id'])")
export POLL=$(python3 -c "import json; print(json.load(open('/tmp/jobs.json'))['poll_url'])")
export RESULT=$(python3 -c "import json; print(json.load(open('/tmp/jobs.json'))['result_url'])")
echo "JOB=$JOB POLL=$POLL RESULT=$RESULT"
```

### T3 — Right after submit: not done, result is 409

Run **immediately** after T2, without sleeping:

```bash
curl -sS -w "\npoll http=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  "$GW$POLL"
echo
curl -sS -w "\nresult http=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  "$GW$RESULT"
```

**Pass:**

- Poll HTTP 200, `"state"` is `"pending"` or `"running"`, **never** `"done"`
  while `T_infer` is still several seconds.
- Result HTTP **409**, body contains `"error"` and mentions the current state
  (`job not finished (state: pending)` or `running`).

**Fail:** result 200 with a full envelope right after a 202 that was supposed to
be admission-only (the old bug: 202 was only flushed after `run_items`).

### T4 — Poll until terminal

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

**Pass:** state becomes `done` (happy path) or `failed` / `timeout` (still a
valid terminal; continue to T5/T6 with that in mind). `elapsed_ms` increases
while the job is running.

**Fail:** stuck on `pending` forever; 404 (job lost — restart wipes the
in-memory ledger); 401.

### T5 — Fetch result only when `done`

```bash
curl -sS -w "\nresult http=%{http_code} time=%{time_total}s\n" \
  --max-time 30 \
  -H "Authorization: Bearer $TOKEN" \
  "$GW$RESULT" | tee /tmp/jobs-result.json
```

**Pass if poll was `done`:** HTTP 200, unified envelope (`status`, `results`,
…). Repeat the same GET: **still 200** (result is repeatable until TTL).

**Pass if poll was `failed` or `timeout`:** HTTP **409** (result is only for
`done`). The poll/wait body already carries `"error"`.

### T6 — Wait until terminal (long-poll)

Submit a **new** job so wait actually hangs:

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

**Pass:**

- Submit `time_total` still << `T_infer`.
- Wait HTTP 200, `"state":"done"` (or `failed`/`timeout`), and wait
  `time_total` is about the remaining run time, **not** the full 120 s budget
  and **not** ~0 s unless the job was already terminal.
- Wait does **not** return `"running"` as a successful “completion”. If you
  only see `running`, you used a too-small timeout (T7) or the job is still
  going.

### T7 — Wait budget expiry (still non-terminal)

New job, wait budget much shorter than `T_infer`:

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

**Pass:** HTTP 200 in **about 200 ms** (not `T_infer`). `"state"` is `pending`
or `running`, not `done`. This is **not** a job failure; call wait again or
poll.

`timeout` is **milliseconds**. `timeout=30` is 30 ms, not 30 seconds.

### T8 — Two clients wait on the same job

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

**Pass:** both HTTP 200, both terminal `state` equal.

### T9 — Admission 429 when the queue is full

`async_max_queue` is **pending + running** (default 8 on the git tomls). Fire
more submits than that **without** waiting for completion:

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

**Pass:** a mix of 202 and 429; 429 body has `"error"` mentioning `async queue full`.
Serial submits (a `for` loop, not only two parallel threads) **must** be able
to hit 429 while earlier jobs are still running.

If every submit is 202, either the model finishes faster than you can POST, or
`async_max_queue` is larger than 12 — raise the loop count or check the server
toml.

### T10 — Auth: missing / wrong Bearer is 401

```bash
curl -sS -o /tmp/unauth.json -w "http=%{http_code}\n" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
# expect 401

curl -sS -o /tmp/bad.json -w "http=%{http_code}\n" \
  -H "Authorization: Bearer definitely-wrong" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
# expect 401
```

`/healthz` stays 200 without a token. `/jobs` does not.

### T11 — Unknown job is 404

```bash
curl -sS -w "\nhttp=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  "$GW/v1/models/$MODEL/jobs/job_does_not_exist"
```

**Pass:** HTTP 404, `"error"` contains `job not found`. Same for
`…/wait` and `…/result` on that id.

### T12 — Async disabled is 404 (negative control)

Pick a **non-async** catalog id that is running (e.g. `MOBILENETV2` on a CPU
pack):

```bash
curl -sS -w "\nhttp=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/MOBILENETV2/jobs"
```

**Pass:** HTTP **404** (upstream has `async_enabled=false`; gateway passes 404
through). Do not treat this as a gateway routing bug.

### T13 — Wrong method is 405

```bash
curl -sS -w "\nhttp=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  -X POST \
  "$GW/v1/models/$MODEL/jobs/$JOB"
```

**Pass:** HTTP 405.

### T14 — Restart drops in-flight jobs

```bash
# submit, then restart that model (management token)
curl -sS -o /tmp/jobs-rst.json \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "$BODY" \
  "$GW/v1/models/$MODEL/jobs"
export JOBR=$(python3 -c "import json; print(json.load(open('/tmp/jobs-rst.json'))['job_id'])")

curl -sS -H "Authorization: Bearer $MGMT" -X POST \
  "$SUP/api/v1/servers/$MODEL/restart"

# wait until ready again, then:
curl -sS -w "\nhttp=%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  "$GW/v1/models/$MODEL/jobs/$JOBR"
```

**Pass:** 404 after restart. The ledger is in-memory; this is expected, not a
data-plane bug. Clients must resubmit.

### T15 — Gateway vs model-port paths (operators only)

Model ports must not be exposed. If you are on the host and the operator
allows loopback checks:

| Gateway | Upstream on the model process |
|---|---|
| `POST /v1/models/{id}/jobs` | `POST /jobs` |
| `GET /v1/models/{id}/jobs/{job}` | `GET /jobs/{job}` |
| `GET /v1/models/{id}/jobs/{job}/wait?timeout=N` | `GET /jobs/{job}/wait?timeout=N` |
| `GET /v1/models/{id}/jobs/{job}/result` | `GET /jobs/{job}/result` |

Bare `/jobs` on `:8080` is **not** a public route; always use the catalog
prefix (or the legacy `{server_uri}` only for sync infer).

### T16 — Malformed envelope is 422, not a hung 202

```bash
curl -sS -w "\nhttp=%{http_code} time=%{time_total}s\n" \
  --max-time 10 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"not":"the envelope"}' \
  "$GW/v1/models/$MODEL/jobs"
```

**Pass:** HTTP 422 (or 400) in milliseconds. No `job_id`. Invalid input never
occupies `async_max_queue`.

---

## 6. Metrics (optional)

On the **model** process, `GET /metrics` needs the process auth token
(`MORTRED_AUTH_TOKEN`, injected by the supervisor — not the gateway inference
token). Operators can scrape:

- `mortred_async_jobs_total{state="submitted|running|done|failed|timeout"}`
- `mortred_async_job_duration_ms`

A successful T2+T4 path increments `submitted` then `running` then `done`.
Gateway `GET /metrics` is a different token (`MORTRED_METRICS_TOKEN`).

---

## 7. Sign-off checklist

Copy this into the test report. Every line is a **pass/fail**.

- [ ] T1: `/infer` works; `T_infer` recorded
- [ ] T2: `POST /jobs` is 202; wall clock << `T_infer`; `Location` / URLs rewritten
- [ ] T3: immediate poll is not `done`; `/result` is 409
- [ ] T4: poll reaches a terminal state
- [ ] T5: `/result` is 200 iff state is `done`; repeatable
- [ ] T6: `/wait` returns terminal without sitting for the full budget
- [ ] T7: short `timeout=200` returns in ~200 ms with `pending`/`running`
- [ ] T8: two waiters both complete
- [ ] T9: at least one 429 while jobs are in flight
- [ ] T10: missing/wrong token → 401
- [ ] T11: unknown id → 404
- [ ] T12: non-async model → 404
- [ ] T13: POST on a job id → 405
- [ ] T14: restart → 404 for the old id
- [ ] T16: bad JSON envelope rejected quickly (no fake 202)

If T2 and T3 fail but T1 works, the long-task path is still blocking the HTTP
series (the defect this release fixes). Collect `/tmp/jobs.hdr`,
`/tmp/jobs.json`, curl `time_total`, and `T_infer` for the operator.

---

## 8. Server knobs (operators)

In `conf/server/**/*_server_config.toml`:

```toml
async_enabled=true          # default false; /jobs is 404 when false
async_timeout=600000        # ms job budget; 0 = unlimited
async_max_queue=8           # pending + running admission cap
async_job_ttl=300000        # ms to keep a terminal job for poll/result
async_max_completed=100     # LRU cap on terminal jobs
```

Changing these requires a model-process restart (`POST /api/v1/servers/{id}/restart`).
Do not raise `async_max_queue` without checking GPU memory: each admitted job
still shares the same worker pool as sync `/infer`.
