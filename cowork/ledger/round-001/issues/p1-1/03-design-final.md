# 技术方案

- 问题: P1-1
- 版本: v1
- 对应列表: v3

## 边界

本方案只解决这一件事：不创建 `/tmp/mortred_stage_timing`。对一个已经能成功推理的覆盖集 id 发一次请求，再读一次 gateway `/metrics`，能看到 `gw_in`、`route`、`auth`、`fwd`、`upstream`。同一次还能从该次推理响应里读到 `server_time_ms`。这次 gateway 阶段 p50 大于该 `server_time_ms` 的一半时，这一次 `/metrics` 正文写明时间主要花在哪一段。

外部样本固定为 `yolov8s`。gateway 只改这一处阶段读数，九个覆盖集 id 共用这一个 gateway，不必按 id 各做一遍。P0-1 已合入基线，本方案不改 `libface`。P2-11 的 calibrate 与 `--raw` HTTP RPS 报告不在本方案内。

「同一次」指：这一次成功推理的响应，加上紧接着的那一次 gateway `GET /metrics`。`server_time_ms` 仍在推理响应正文里。写明主要段落的是这一次 `/metrics` 正文。

TensorRT 保持 static shape、batch=1。每个 worker 仍各自持有一个 execution context。

## 做法

### 推荐

gateway 在成功推理的转发回调里，自行记下五段耗时，并在 `GET /metrics` 里以直方图打出。这段记录不调用 `stage_timing::enabled()`，也不看 `/tmp/mortred_stage_timing`。flag 文件继续只决定要不要把原有的 `stage_trace` 打到 stderr。

五段的时钟与现有 mark 对齐，单位是毫秒，用 `steady_clock` 的小数，不截成整数毫秒。只在上游 HTTP 状态为 200、且正文顶层 `server_time_ms` 是有限数字且大于等于 0 时，五段连同这个数字一次写入。写入点在现有 `upstream` mark 处，位于把正文拷回客户端之前。

- `t_enter`：`process()` 入口。
- `gw_in`：`t_enter` 到现有 `gw_in` mark。这一段含 CORS 和进入推理路径前的三次短路判断，不含 workflow 在 `process()` 之前读 socket 的时间。
- `route`：现有 `gw_in` mark 到现有 `route` mark。这一段含现有的来源限流判断和路由查找。
- `auth`：现有 `route` mark 到现有 `auth` mark。
- `fwd`：现有 `auth` mark 到现有 `fwd` mark。
- `upstream`：现有 `fwd` mark 到现有 `upstream` mark。这一段是等待模型进程的时间，不含随后把正文写回客户端的时间。

`server_time_ms` 用 gateway 已链接的 rapidjson 解析上游正文的副本，只读顶层数字。解析不改写转发给客户端的字节。

gateway 阶段 p50 是五段各自 p50 之和。count 为 1 时，一段的 p50 就是该段 `_sum`。五段 p50 之和大于响应里这个 `server_time_ms` 的一半时，同一次正文多一行 gauge，`stage` 为五段中 p50 最大的名字。并列时按 `gw_in`、`route`、`auth`、`fwd`、`upstream` 取靠后的名字。大于是严格大于。

代价：每次成功推理在 gateway 回调里多一次 JSON 解析；`/metrics` 正文变长；监控指南和 API 契约摘要要补上这两条序列；fake 上游要能返回带 `server_time_ms` 的信封。gateway 不加载 engine，不增加显存。

### 否决

把 `stage_timing::enabled()` 改成恒为真，用 stderr 的 `stage_trace` 当读数。模型进程上的 decode、预处理和 TensorRT 阶段也会每条请求打日志。锁定的读数位置是 gateway `/metrics`，而且验收要求不创建 flag 文件。

把五段塞进现有的 `mortred_http_request_duration_ms`，给它加 `stage` 标签。这条直方图现在的标签是 `method` 和 `status`，计时窗口在 `forward_to_model` 内部，并用整数毫秒、桶从 5 开始。`gw_in`、`route`、`auth` 盖不住，已有的整段 HTTP 耗时序列也会被拆开。

## 改动面

- `src/control/gateway/gateway_app.cpp`：成功的推理转发在回调里写入五段和配对的 `server_time_ms`；`GET /metrics` 在现有正文之后追加本方案的直方图和条件 gauge。监听、鉴权、路由和转发字节保持原路径。
- `src/control/gateway/gateway_app.h`：仅当转发函数要带上这五个时间点时改签名。监听和鉴权接口不改。
- `test/fake_model_server.cc`：新增 `--mode envelope`，并且必须同时带 `--server-time-ms`。该模式只替换非 `/jobs` 的 POST 正文，GET `/ready` 仍返回 `ok`。正文是带顶层数字 `server_time_ms` 的统一信封，HTTP 200。缺 `--server-time-ms` 时该模式以非 0 退出。默认模式的正文保持 `{"code":0,"msg":"success","data":{"fake":true}}`。
- `test/gateway_e2e_test.cc`：按下述测试义务增加用例。
- `docs/monitoring-guide.md`、`docs/monitoring-guide.zh-cn.md`：增加 gateway 专有的这两条序列，以及 count 为 1 时的 p50 算法、比较式、并列规则，并写明不创建 `/tmp/mortred_stage_timing` 也能读到。
- `docs/api-contract.md`、`docs/api-contract.zh-cn.md`：在 gateway 拓扑说明里写明，一次成功推理之后 gateway `/metrics` 带有这五个 stage；`server_time_ms` 仍在推理响应正文；`GET /metrics` 的 scrape 鉴权不变。

## 明确不改

- `src/common/stage_timing.h`。`enabled()` 仍只探测 `/tmp/mortred_stage_timing`。`to_log_line` 仍不打印 `gw_in`。flag 存在时，stderr 上 `route`、`auth`、`fwd`、`upstream` 的差值与现在相同。
- `src/server/prometheus_metrics.h` 和模型进程的 `/metrics`。模型进程不出现 `mortred_gateway_stage_duration_ms` 或 `mortred_gateway_stage_dominant`。
- `mortred_http_request_duration_ms` 的名字、标签、整数毫秒和现有计时窗口。
- 推理请求和响应的 JSON 字段。gateway 不增删、不改写 `server_time_ms`。`docs/openapi.json` 不改。
- `conf/` 下的 toml、pack、`worker_nums`、产品 `type`、engine、`docs/benchmark-h3/`。
- 来源限流的调用位置和 `src/control/rate_limit/gcra.h`。限流默认仍关闭。
- 模型进程、TensorRT execution context、worker 池、`src/server/item_exec.h`。
- `src/control/cli/cli_app.cpp` 里为性能埋点创建 flag 文件的路径。本方案的验收命令不走那条路径。
- `deploy/prometheus.yml` 和告警规则。新序列随 gateway 正文被现有抓取带上。
- `/jobs` 的转发和 URL 改写。`/jobs` 不写这五段。

## 契约

`GET /metrics` 仍要求 `MORTRED_METRICS_TOKEN`，含环回。状态 200，`Content-Type` 仍是 `text/plain; version=0.0.4; charset=utf-8`。

一次配对成功之后，正文追加直方图 `mortred_gateway_stage_duration_ms`。标签只有 `model="gateway"` 和 `stage`。`stage` 只取 `gw_in`、`route`、`auth`、`fwd`、`upstream`，并按这个顺序输出。不按 catalog id 拆系列。桶上界为 `0.05`、`0.1`、`0.25`、`0.5`、`1`、`2`、`5`、`10`、`25`、`50`、`100`、`250`、`500`、`1000`、`2000`、`5000`、`10000` 和 `+Inf`。

count 为 1 时，该段 p50 等于该段 `_sum`。gateway 阶段 p50 等于五段 p50 之和。比较用的模型时间是同一次推理响应里的那一个 `server_time_ms`。

条件成立时，直方图之后恰好再出现这三行，`STAGE` 为按并列规则选出的名字：

```text
# HELP mortred_gateway_stage_dominant Time is mainly spent in this gateway stage
# TYPE mortred_gateway_stage_dominant gauge
mortred_gateway_stage_dominant{model="gateway",stage="STAGE"} 1
```

gauge 的值固定为 1，表示段落名字，不表示毫秒。条件不成立时，正文里不出现 `mortred_gateway_stage_dominant` 这串字符，包括 HELP 和 TYPE。

进程内保留最近 4096 条配对记录，供 count 大于 1 时计算中位数：升序后奇数取正中，偶数取正中两个的算术平均。直方图的 `_count` 和 `_sum` 仍是进程启动以来的累计值，不随淘汰减小。累计条数小于等于 4096 时，主导段按全部原始样本的中位数计算。累计条数大于 4096 时，不再输出主导段。本条外部验收只使用五段 `_count` 都是 1 的读数，此时中位数就是 `_sum`。

401、404、405、429、上游传输失败、上游非 200、顶层 `server_time_ms` 缺失或不是有限非负数、以及 `/jobs`，都不写入五段。`OPTIONS`、`/healthz`、`/metrics` 也不写入。

## 性能与资源

gateway 是常驻进程。本方案不在 gateway 里初始化 CUDA，不加载 engine，不增加显存。模型进程的 worker 数和每个 worker 自己的 execution context 保持现状。自测拉起的 `yolov8s` 使用现有 server toml 的 `worker_nums=1`，不设置 `MORTRED_WORKER_NUMS` 或 `MORTRED_DECODE_AUTO`。

五段直方图的桶数是固定的。原始样本最多 4096 条，每条是五个耗时加一个 `server_time_ms`。锁只包住写入和渲染，不包住等待上游的时间。

`upstream` 含等待模型进程的时间，通常是五段里最大的一段。主导段因此经常是 `upstream`。这是这段时钟的结果。本方案不从 `upstream` 里扣掉 `server_time_ms`，也不把模型进程里的 decode、预处理、h2d、exec、d2h 做进 gateway 读数。

现有的 `mortred_http_request_duration_ms` 仍是转发窗口上的整数毫秒，不作为本条的 gateway 阶段 p50。

## 验收标准

1. 验收开始前 `/tmp/mortred_stage_timing` 不存在。完成下述推理和一次 gateway `/metrics` 之后它仍然不存在。gateway 进程不创建、不写入、不删除这个路径。
2. 对已能成功推理的 `yolov8s` 发一次 `POST /v1/models/yolov8s/infer`。HTTP 状态为 200，响应 JSON 的 `status` 为 0，顶层 `server_time_ms` 是大于等于 0 的有限数字。这次响应正文仍带这个字段。
3. 紧接着用 metrics token 对这个 gateway 发一次 `GET /metrics`，HTTP 状态为 200。正文里 `gw_in`、`route`、`auth`、`fwd`、`upstream` 各有一条 `mortred_gateway_stage_duration_ms_count{model="gateway",stage="该名字"}`，值都是 1，并各有对应的 `_sum`。
4. 同一次 `/metrics` 正文仍含 `mortred_http_request_duration_ms`。
5. 设 G 为这五个 `_sum` 之和，S 为标准 2 的 `server_time_ms`。G 大于 `0.5 * S` 时，正文有标准契约里的那一行 gauge，`stage` 等于五段中 `_sum` 最大的名字，并列时取 `gw_in`、`route`、`auth`、`fwd`、`upstream` 中靠后的名字。G 小于或等于 `0.5 * S` 时，正文不出现 `mortred_gateway_stage_dominant`。
6. 用该模型进程的 Bearer 抓一次模型端口 `/metrics`。正文不出现 `mortred_gateway_stage_duration_ms`，也不出现 `mortred_gateway_stage_dominant`。
7. `docs/monitoring-guide.md` 与 `docs/monitoring-guide.zh-cn.md` 写明上述直方图、五个 stage、count 为 1 时 p50 取 `_sum`、比较式和并列规则，并写明不创建 `/tmp/mortred_stage_timing` 也能读到。中文指南含「时间主要花在」。`docs/api-contract.md` 与 `docs/api-contract.zh-cn.md` 写明阶段序列在 gateway `/metrics`，`server_time_ms` 仍在推理响应正文，scrape 鉴权不变。

## 测试义务

QA 先跑 `gateway_e2e_test` 里下面五个用例，再按验收标准在 WSL 上对 `yolov8s` 做一次外部验收。外部验收只打 `yolov8s` 这一次，不对其余八个覆盖集 id 重复，也不跑 calibrate 或 `http_infer_rps.py`。

- `StageMetricsVisibleWithoutFlag`：不创建 flag 文件。fake 使用 `--mode envelope --server-time-ms 100000`。推理响应里的 `server_time_ms` 为 100000。一次 `/metrics` 里五段 count 都是 1，正文没有 `mortred_gateway_stage_dominant`。结束后 flag 文件仍不存在。
- `StageMetricsDominantNamesTheLargestStage`：`--server-time-ms 0.125`。响应里的数为 0.125。按验收标准 5 用五个 `_sum` 计算获胜段，gauge 的 `stage` 与之相同，HELP 行与契约一致。
- `StageMetricsSkippedOnUnknownRoute`：对未注册路径 POST 一次之后，`/metrics` 不出现 `mortred_gateway_stage_duration_ms`。
- `StageMetricsSkippedOnJobPost`：对 fake 的 `POST /jobs` 返回 202 之后，五段 count 仍为 0。
- `StageMetricsRecordedWhenFlagFileExists`：测试自己创建 flag 文件后再发一次 envelope 推理，五段 count 仍为 1。用例退出前删除该文件。产品代码不创建它。

已有的 `metrics_endpoint_renders` 仍通过。默认 fake 模式的旧用例仍通过。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面每条的期望退出码都是 0。

第一条：

```bash
cd /mnt/g/Codex/mortred_model_server
cmake --build build/full --target gateway_e2e_test
scripts/run_tests.sh build/full -R '^gateway_e2e_test$' --output-on-failure
```

第二条覆盖验收标准 1 至 6。开始时 `/tmp/mortred_stage_timing` 必须不存在，`127.0.0.1:9056/ready` 与 `127.0.0.1:18080/healthz` 必须连不上。模型工作目录是仓库的 `_bin`，可执行文件是 `_bin/mortred-model-server.out`。gateway 使用 `_bin/mortred-gateway.out`，端口 18080。三个对外 token 与模型 token 彼此不同。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_RATE_LIMIT`。不执行 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。

```bash
cd /mnt/g/Codex/mortred_model_server
test ! -e /tmp/mortred_stage_timing
curl -sf --max-time 1 http://127.0.0.1:9056/ready >/dev/null && exit 1
curl -sf --max-time 1 http://127.0.0.1:18080/healthz >/dev/null && exit 1
model_pid=""
gw_pid=""
cleanup() {
  if [ -n "$gw_pid" ]; then kill "$gw_pid" 2>/dev/null || true; wait "$gw_pid" 2>/dev/null || true; fi
  if [ -n "$model_pid" ]; then kill "$model_pid" 2>/dev/null || true; wait "$model_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
cd /mnt/g/Codex/mortred_model_server/_bin
env -u MORTRED_WORKER_NUMS -u MORTRED_DECODE_AUTO -u MORTRED_MODEL_CONFIG_FILE \
  MORTRED_AUTH_TOKEN=p1-1-internal \
  ./mortred-model-server.out --model yolov8s \
  /mnt/g/Codex/mortred_model_server/conf/server/object_detection/yolov8/yolov8s_server_config.toml \
  >/tmp/p1-1-model.log 2>&1 &
model_pid=$!
ready=0
for _ in $(seq 1 90); do
  if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:9056/ready; then ready=1; break; fi
  sleep 1
done
test "$ready" -eq 1
cd /mnt/g/Codex/mortred_model_server
env -u MORTRED_RATE_LIMIT \
  MORTRED_GATEWAY_HOST=127.0.0.1 MORTRED_GATEWAY_PORT=18080 \
  MORTRED_GATEWAY_AUTH_TOKEN=p1-1-infer MORTRED_API_TOKEN=p1-1-admin \
  MORTRED_METRICS_TOKEN=p1-1-metrics MORTRED_INTERNAL_TOKEN=p1-1-internal \
  ./_bin/mortred-gateway.out >/tmp/p1-1-gateway.log 2>&1 &
gw_pid=$!
gw_ready=0
for _ in $(seq 1 30); do
  if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:18080/healthz; then gw_ready=1; break; fi
  sleep 1
done
test "$gw_ready" -eq 1
curl -sS -D /tmp/p1-1.hdr -o /tmp/p1-1.json \
  -H "Authorization: Bearer p1-1-infer" -H "Content-Type: image/jpeg" \
  --data-binary @/mnt/g/Codex/mortred_model_server/demo_data/model_test_input/object_detection/bus.jpg \
  http://127.0.0.1:18080/v1/models/yolov8s/infer
curl -sS -H "Authorization: Bearer p1-1-metrics" \
  http://127.0.0.1:18080/metrics -o /tmp/p1-1.metrics
curl -sS -H "Authorization: Bearer p1-1-internal" \
  http://127.0.0.1:9056/metrics -o /tmp/p1-1.model-metrics
test ! -e /tmp/mortred_stage_timing
python3 - <<'PY'
import json
from pathlib import Path
hdr = Path("/tmp/p1-1.hdr").read_text(encoding="utf-8", errors="replace")
code = int(hdr.splitlines()[0].split()[1])
body = json.loads(Path("/tmp/p1-1.json").read_text(encoding="utf-8"))
metrics = Path("/tmp/p1-1.metrics").read_text(encoding="utf-8")
model_metrics = Path("/tmp/p1-1.model-metrics").read_text(encoding="utf-8")
assert code == 200
assert body["status"] == 0
s = body["server_time_ms"]
assert isinstance(s, (int, float)) and s >= 0
names = ["gw_in", "route", "auth", "fwd", "upstream"]
sums = {}
for name in names:
    count_key = f'mortred_gateway_stage_duration_ms_count{{model="gateway",stage="{name}"}} 1'
    assert count_key in metrics, name
    prefix = f'mortred_gateway_stage_duration_ms_sum{{model="gateway",stage="{name}"}} '
    line = next(x for x in metrics.splitlines() if x.startswith(prefix))
    sums[name] = float(line[len(prefix):])
assert "mortred_http_request_duration_ms" in metrics
g = sum(sums.values())
marker = "mortred_gateway_stage_dominant"
if g > 0.5 * s:
    winner = max(names, key=lambda n: (sums[n], names.index(n)))
    assert metrics.count(marker) == 3
    assert f'{marker}{{model="gateway",stage="{winner}"}} 1' in metrics
    assert "# HELP mortred_gateway_stage_dominant Time is mainly spent in this gateway stage" in metrics
else:
    assert marker not in metrics
assert "mortred_gateway_stage_duration_ms" not in model_metrics
assert marker not in model_metrics
print("p1-1-ok", g, s)
PY
```

第三条覆盖验收标准 7：

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
from pathlib import Path
root = Path("/mnt/g/Codex/mortred_model_server")
guides = [
    root / "docs/monitoring-guide.md",
    root / "docs/monitoring-guide.zh-cn.md",
]
contracts = [
    root / "docs/api-contract.md",
    root / "docs/api-contract.zh-cn.md",
]
need = ["mortred_gateway_stage_duration_ms", "mortred_gateway_stage_dominant",
        "gw_in", "route", "auth", "fwd", "upstream", "/tmp/mortred_stage_timing"]
for path in guides:
    text = path.read_text(encoding="utf-8")
    for item in need:
        assert item in text, (path, item)
    assert "_sum" in text
assert "时间主要花在" in (root / "docs/monitoring-guide.zh-cn.md").read_text(encoding="utf-8")
for path in contracts:
    text = path.read_text(encoding="utf-8")
    assert "mortred_gateway_stage_duration_ms" in text
    assert "server_time_ms" in text
    assert "MORTRED_METRICS_TOKEN" in text
print("docs-ok")
PY
```

## RD 不得自行决定

- 五个 stage 名字、顺序、直方图名、gauge 名、`model="gateway"` 和 gauge 值 1。不得按 catalog id 再拆标签，不得把毫秒写进 gauge 的值。
- count 为 1 时 p50 用 `_sum`。比较式是五段之和严格大于 `0.5 * S`。并列时取 `gw_in`、`route`、`auth`、`fwd`、`upstream` 中靠后的名字。
- 样本上限是 4096。超过之后不再输出主导段。直方图累计 `_count` 不回退。
- 不把 `stage_timing::enabled()` 改成恒真，不改 `to_log_line`，不让本方案创建 `/tmp/mortred_stage_timing`。
- 不改 `mortred_http_request_duration_ms`，不把新序列放进模型进程的 `/metrics`。
- 不改推理 JSON，不改写响应里的 `server_time_ms`。`/jobs` 不记这五段。
- 不改限流顺序，不打开限流开关，不改 `worker_nums`，不共享 TensorRT execution context，不增加 context 池，不改成动态 shape 或动态 batch。
- 不跑 calibrate，不跑 `http_infer_rps.py`，不重开 P0-1，不把其余八个覆盖集 id 加进本条验收。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v1 | `cowork/ledger/round-001/issues/p1-1/03-discussion/v1-sign-pm.md` |
| RD | 同意 | 无 | v1 | `cowork/ledger/round-001/issues/p1-1/03-discussion/v1-sign-rd.md` |
| 技术负责人 | 同意 | 无 | v1 | `cowork/ledger/round-001/issues/p1-1/03-discussion/v1-sign-tech-lead.md` |
