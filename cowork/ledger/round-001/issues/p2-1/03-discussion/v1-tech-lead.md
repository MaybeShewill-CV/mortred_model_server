# 技术方案

- 问题: P2-1
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` v3 的 P2-1

## 边界

本方案只补这一件事：覆盖集 server toml 的 `max_batch_size=1` 时，同步请求里每一张已经跑完的图，向现有直方图记一次值为 1 的 batch 观测。一次提交 N 张合法图片后，响应 `results` 长度仍是 N。随后在这个模型进程上 `GET /metrics`，`mortred_batch_size` 的样本都是 1，count 等于这次已经跑完的张数，因此 sum 等于 count。

外部样本固定为已经能成功推理的 `yolov8s`，N 固定为 2。使用现有 `conf/server/object_detection/yolov8/yolov8s_server_config.toml`，其中 `max_batch_size=1`、`worker_nums=1`。九个覆盖集 id 共用 `BaseAiServerImpl` 的这一段同步路径，其余八个 id 不各做一次外部验收。

TensorRT 保持 static shape、batch=1。每个 worker 仍各自持有一个 execution context。不跑 calibrate，不跑 `http_infer_rps.py`。

「这次」指：一个新拉起的 `yolov8s` 进程，在本次推理之前没有执行过图片。count 与 sum 是进程启动以来的累计值；这次之前没有样本，所以读数就等于这次跑完的张数。

## 做法

### 推荐

`max_batch_size<=1` 时继续走现有的 `schedule_sync_request` → `start_sync_item`，逐张 `run_one`，不启动 `BatchCollector`。一张图的 `run_one` 已经返回、该张被标成跑完之后，调用现有的 `observe_batch_size(1)` 恰好一次。门闩与现有的逐张推理耗时观测相同：同一张只记一次；回调的多条分支不能再记第二次。没进入 `run_one`，或 `run_one` 尚未返回的张，不记。

观测值固定为 1，不是 N，也不是这一次 HTTP 请求记一条。N 张都跑完时，直方图 count 增加 N，sum 增加 N。桶上界仍是现有的 `1, 2, 4, 8, 16, 32`。值为 1 时，`le="1"` 与更高的累计桶一起增加。

代价：每张跑完的图在现有指标锁里做一次直方图累加，发生在 `run_one` 返回之后。不增加线程，不增加显存，不新建 execution context，不改变这一请求 checkout worker 的次数。第一次观测之后，`/metrics` 多出 `mortred_batch_size` 的样本行；在此之前渲染器仍只输出这条直方图的 HELP 和 TYPE。

### 否决

把 `max_batch_size<=1` 改走 `BatchCollector`，用它现有的 `observe_batch_size(batch.size())`。`start()` 在 `max_batch_size<=1` 时直接返回；要让它记数，就得把收集线程拉起来，并按 `max_batch_delay_ms` 等窗口，再改用 `run_batch`。那次观测发生在真正执行之前，checkout 失败的张也会被记进 count。覆盖集 toml 的上限仍是 1，凑批不会发生，多出来的是线程、窗口等待和另一套超时语义。`mortred_batch_window_wait_ms` 也会出现样本。

按一次 HTTP 请求记一条。值记成 1 时，count 是 1，对不上这次跑完的张数。值记成 N 时，样本不是 1，sum 与 count 也不相等。在回复时按 `n_items` 连记 N 次 1，会把还没跑完、甚至没进入 `run_one` 的张算进去；响应长度本来就是按 `n_items` 组装的，那条路径现在已经满足，不能拿来充当执行张数。

## 改动面

- `src/server/base_server_impl.h`：只在 `max_batch_size<=1` 的同步逐张完成路径上，于该张 `run_one` 返回且已标成跑完时记 `observe_batch_size(1)`。调度、checkout、回复和 `results` 组装保持原路径。
- `test/server_e2e_contract_test.cc`：按下述测试义务增加两个用例。

## 明确不改

- `src/server/batch_collector.h`。`max_batch_size>1` 时仍在收集到一批之后记 `observe_batch_size(batch.size())`，并仍记窗口等待。`max_batch_size<=1` 时 `start()` 仍直接返回，收集线程不启动。
- `src/server/prometheus_metrics.h`。`observe_batch_size` 的签名、桶上界、HELP 文案和「没有观测过的序列不渲染样本行」保持现状。`model` 标签仍是 `set_model` 写入的 server URI。
- `src/server/item_exec.h` 的 `run_one`、`run_items`。指标不放进单张执行原语。
- `async_run_job` 与 `do_work`。二者不在这次同步提交的路径上。
- 响应组装：`results` 长度仍按 `n_items`。单张失败不填 `data`，超时且已有成功项时顶层仍为 `DEADLINE_EXCEEDED_PARTIAL`。
- `conf/` 下全部 toml，包括九个覆盖集 server toml 的 `max_batch_size` 与 `worker_nums`。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。
- worker 池、TensorRT execution context 的创建方式。不共享 context，不加 context 池。engine 文件与 `docs/benchmark-h3/` 不改，不执行 `mortredctl prepare`。
- gateway、`docs/openapi.json`、监控指南。请求与响应的 JSON 字段不增删。

## 契约

推理请求与响应的字段不变。一次 JSON 请求的 `images` 有 N 张时，响应 `results` 长度是 N。

`GET /metrics` 仍走模型进程现有鉴权：配置了 token 时必须带同一个 Bearer，状态 200，`Content-Type` 仍是 `text/plain; version=0.0.4; charset=utf-8`。不新增 metrics token。

`mortred_batch_size` 仍是直方图，桶上界仍是 `1`、`2`、`4`、`8`、`16`、`32` 和 `+Inf`。`max_batch_size<=1` 的同步路径上，每一张 `run_one` 已返回的图增加一次观测，值恰好是 1。因此：

- `_count` 增加 1
- `_sum` 增加 1
- `le="1"` 的累计桶增加 1，更高的累计桶同样增加 1

没返回的张不增加这些数。`model` 标签仍是 server URI。`yolov8s` 上是 `model="/mortred_ai_server_v1/obj_detection/yolov8s"`。验收看 count、sum 和 `le="1"` 的数值。

这条路径不调用 `observe_batch_window_wait_ms`。进程在本次请求之前没有别的 batch 观测时，正文里不出现 `mortred_batch_window_wait_ms_count{`。HELP 与 TYPE 行可以仍在。

`max_batch_size>1` 的收集路径仍是一批记一次，值是该批的张数。现有四张请求凑成少于四批时，sum 为 4 且 count 小于 4。

送入会话的仍是现有 static shape、batch=1 的 engine，每张图一次 `run_one`。这条路径不调用 `run_batch`。

## 性能与资源

常驻的仍是一个模型进程。自测的 `yolov8s` 使用 toml 里的 `worker_nums=1`，进程内一个 worker，持有它自己的那一个 execution context。`BatchCollector` 的线程在 `max_batch_size=1` 时不存在。

直方图累加在 `run_one` 返回之后，不包住 GPU 执行。不分配新的显存，不生成新的 engine。产品文件仍指向 `yolov8s.engine.static_bs1.fp16`。

## 验收标准

1. `conf/server/object_detection/yolov8/yolov8s_server_config.toml` 中 `max_batch_size=1` 且 `worker_nums=1`。进程启动时环境里没有 `MORTRED_WORKER_NUMS` 和 `MORTRED_DECODE_AUTO`。`127.0.0.1:9056/ready` 在本次推理前返回成功。
2. 本次推理之前，用该进程的 Bearer 对 `127.0.0.1:9056/metrics` 做一次 GET，状态 200。正文没有以 `mortred_batch_size_count{` 开头的样本行。
3. 向 toml 中的 `server_uri` `/mortred_ai_server_v1/obj_detection/yolov8s` POST 一次 JSON。`images` 是两张元素，都是 `demo_data/model_test_input/object_detection/bus.jpg` 文件字节的标准 base64，不换行。HTTP 状态 200。响应 JSON 的 `status` 为 0，`results` 长度为 2，两条 `results[].status` 都为 0。
4. 紧接着再 GET 一次同一 `/metrics`。恰好一条 `mortred_batch_size_count{...}`，值为 2。恰好一条对应的 `mortred_batch_size_sum{...}`，值等于该 count。恰好一条带 `le="1"` 的 `mortred_batch_size_bucket{...}`，值等于该 count。
5. 同一次正文没有以 `mortred_batch_window_wait_ms_count{` 开头的样本行。

## 测试义务

QA 先跑 `server_e2e_contract_test` 里下面四个用例，再按验收标准在 WSL 上对 `yolov8s` 做一次外部验收。外部验收只打 `yolov8s` 这一次，N 为 2。不对其余八个覆盖集 id 重复，不跑 calibrate，不跑 `http_infer_rps.py`，不跑 `mortredctl prepare`。

- `batch_size_one_counts_each_executed_image`：`start_server` 不写 `max_batch_size`，因此运行配置为 1。推理前 `/metrics` 没有 `mortred_batch_size_count{`。POST 两张图，HTTP 200，`results` 长度为 2，两条 item status 都为 0。随后 count 为 2，sum 为 2，`le="1"` 桶为 2。正文没有 `mortred_batch_window_wait_ms_count{`。
- `batch_size_one_omits_item_still_inside_run`：`model_run_timeout=100`，`fake_delay_ms=1000`，不写 `max_batch_size`。POST 两张图，HTTP 504，`results` 长度为 2。响应返回后立刻读 `/metrics`，此时第一张仍在 fake delay 内，正文没有 `mortred_batch_size_count{`。随后轮询直到 `mortred_inference_duration_ms_count` 为 1，期限 3 秒。那时 batch 的 count、sum 和 `le="1"` 都是 1。再等 200 毫秒，这三个值仍是 1。
- 已有的 `batch_collects_and_distributes_per_request_results` 仍通过：四次请求的 `mortred_batch_size_sum` 为 4，count 大于等于 1 且小于 4。
- 已有的 `model_timeout_n_items_keeps_results_length` 仍通过：两张图的 `results` 长度为 2。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面每条的期望退出码都是 0。

第一条先确认两个新用例已经编进二进制，再跑整份 `server_e2e_contract_test`。

```bash
cd /mnt/g/Codex/mortred_model_server
cmake --build build/full --target server_e2e_contract_test
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./_bin/server_e2e_contract_test --gtest_list_tests > /tmp/p2-1-list.txt
grep -F 'batch_size_one_counts_each_executed_image' /tmp/p2-1-list.txt
grep -F 'batch_size_one_omits_item_still_inside_run' /tmp/p2-1-list.txt
scripts/run_tests.sh build/full -R '^server_e2e_contract_test$' --output-on-failure
```

第二条覆盖验收标准 1 至 5。开始时 `127.0.0.1:9056/ready` 必须连不上。模型工作目录是仓库的 `_bin`，可执行文件是 `_bin/mortred-model-server.out`。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_MODEL_CONFIG_FILE`。不执行 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。

```bash
cd /mnt/g/Codex/mortred_model_server
grep -q '^max_batch_size=1$' conf/server/object_detection/yolov8/yolov8s_server_config.toml
grep -q '^worker_nums=1$' conf/server/object_detection/yolov8/yolov8s_server_config.toml
test -f demo_data/model_test_input/object_detection/bus.jpg
curl -sf --max-time 1 http://127.0.0.1:9056/ready >/dev/null && exit 1
model_pid=""
cleanup() {
  if [ -n "$model_pid" ]; then kill "$model_pid" 2>/dev/null || true; wait "$model_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
cd /mnt/g/Codex/mortred_model_server/_bin
env -u MORTRED_WORKER_NUMS -u MORTRED_DECODE_AUTO -u MORTRED_MODEL_CONFIG_FILE \
  MORTRED_AUTH_TOKEN=p2-1-model \
  ./mortred-model-server.out --model yolov8s \
  /mnt/g/Codex/mortred_model_server/conf/server/object_detection/yolov8/yolov8s_server_config.toml \
  >/tmp/p2-1-model.log 2>&1 &
model_pid=$!
ready=0
for _ in $(seq 1 90); do
  if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:9056/ready; then ready=1; break; fi
  sleep 1
done
test "$ready" -eq 1
curl -sS -H "Authorization: Bearer p2-1-model" \
  http://127.0.0.1:9056/metrics -o /tmp/p2-1-before.metrics
python3 - <<'PY'
import base64, json
from pathlib import Path
raw = Path("/mnt/g/Codex/mortred_model_server/demo_data/model_test_input/object_detection/bus.jpg").read_bytes()
b64 = base64.b64encode(raw).decode("ascii")
Path("/tmp/p2-1.jsonreq").write_text(
    json.dumps({"images": [b64, b64], "req_id": "p2-1"}), encoding="utf-8")
PY
curl -sS -D /tmp/p2-1.hdr -o /tmp/p2-1.json \
  -H "Authorization: Bearer p2-1-model" -H "Content-Type: application/json" \
  --data-binary @/tmp/p2-1.jsonreq \
  http://127.0.0.1:9056/mortred_ai_server_v1/obj_detection/yolov8s
curl -sS -H "Authorization: Bearer p2-1-model" \
  http://127.0.0.1:9056/metrics -o /tmp/p2-1-after.metrics
python3 - <<'PY'
import json
from pathlib import Path
hdr = Path("/tmp/p2-1.hdr").read_text(encoding="utf-8", errors="replace")
code = int(hdr.splitlines()[0].split()[1])
body = json.loads(Path("/tmp/p2-1.json").read_text(encoding="utf-8"))
before = Path("/tmp/p2-1-before.metrics").read_text(encoding="utf-8")
after = Path("/tmp/p2-1-after.metrics").read_text(encoding="utf-8")

def values(text, name):
    found = []
    prefix = name + "{"
    for line in text.splitlines():
        if line.startswith(prefix):
            found.append(float(line.rsplit(" ", 1)[1]))
    return found

assert not any(line.startswith("mortred_batch_size_count{") for line in before.splitlines())
assert code == 200
assert body["status"] == 0
assert len(body["results"]) == 2
assert body["results"][0]["status"] == 0
assert body["results"][1]["status"] == 0
counts = values(after, "mortred_batch_size_count")
sums = values(after, "mortred_batch_size_sum")
le1 = []
for line in after.splitlines():
    if line.startswith("mortred_batch_size_bucket{") and 'le="1"' in line:
        le1.append(float(line.rsplit(" ", 1)[1]))
assert counts == [2.0]
assert sums == [2.0]
assert le1 == [2.0]
assert not any(line.startswith("mortred_batch_window_wait_ms_count{") for line in after.splitlines())
print("p2-1-ok")
PY
```

## RD 不得自行决定

- 观测值是 1。每张 `run_one` 已返回的图记一次。一次 HTTP 请求只记一条，或把值记成 N，都不符合本方案。
- 没进入 `run_one`，或 `run_one` 尚未返回的张，不记。回复时按 `n_items` 补记，不符合本方案。
- 不把 `max_batch_size<=1` 改走 `BatchCollector`，不在这条路径上记 `mortred_batch_window_wait_ms`。
- 不改直方图桶、渲染条件和 `model` 标签。不改 `observe_batch_size` 的签名。
- 不改 `async_run_job`、`do_work`、`run_items`。
- 不改任何 server toml 的 `max_batch_size`、`worker_nums`、`model_run_timeout`。外部验收的 N 是 2。不得靠加大超时或提高 `max_batch_size` 让命令通过。
- 不改成动态 batch 或动态 shape。不让多个 worker 共享一个 TensorRT execution context，不加 context 池。不跑 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。
- 不把其余八个覆盖集 id 加进本条外部验收。
