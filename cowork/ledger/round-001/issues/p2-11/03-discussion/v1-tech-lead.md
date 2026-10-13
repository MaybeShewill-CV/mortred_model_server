# 技术方案

- 问题: P2-11
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-11

## 边界

本条没有上一版技术方案。队列 v3 的用户决定不把 v2 多出的「每个 worker 仍各自持有 TensorRT execution context」放进通过条件。评分复核里 P1-1 已关闭；PF4 的报告半边仍在本条。这些都写进正文。

锁定的可观察变化是一张清单。覆盖集九个 `type=tensorrt` 的 id 没有一份按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告（PF1、PF2、PF5、TP1）。PF4 在本条只保留报告半边：同一次压测要能读到 gateway 阶段耗时和模型 `server_time_ms`。`365b5dcd` 的 calibrator v4 已在仓库里，它不是本条的关闭。

P2-8 留在附录，不恢复。P0-1、P1-1、TP2（P2-4）已关闭，不重做。九个 id 各自 prepare、拉起、推理。驻留集仍是 `mortredctl doctor --strict` 允许的那一组，至少 `yolov8s` 与 `libface` 同时常驻；那是已关闭的 TP2，不写进本条通过条件。不要求九个 id 同时常驻。

实现不得改成共享 context，也不得增加跨 worker 的 context 池。这句话是结构约束，不是清单勾销条件。

## 做法

### 推荐

新增 `scripts/coverage_http_report.py`。它按锁定顺序逐个处理九个 id：`mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`。同一时刻只准备、只拉起、只压测当前这一个 id。通过之后把一张清单写到 `docs/perf/coverage_set_http_rps.md`。打不开这一张文件，就不能判定。

每个 id 的顺序固定为四步。

1. `mortredctl prepare --id <id>`。引擎保持现有 static shape、batch=1。已有引擎时沿用现有 prepare 的跳过规则。
2. 在仓库外写一份只含 `[pack.<id>]` 的临时 pack，初值 `worker_nums = 1`，不写 `model_config`。然后执行 `mortredctl calibrate --pack <该文件> --write-pack`，不传 `--duration`，因此用 v4 的默认 30 秒，结论才会写入 pack。不传 `--from-json`，不传 `--workers`。v4 内部的 worker 数扫描和 JPEG 解码投票已经走 `--raw`。脚本不增加 base64 路径，也不改 `scripts/calibrate_pack.py`。
3. 从写回的 `[pack.<id>]` 读出 `worker_nums` 和 `decode_auto`。`decode_auto` 必须是 `cpu` 或 `gpu`。再只把这一份 pack 交给 supervisor（`MORTRED_PACK`）。`conf/mortred.toml` 的 `autostart_default = false` 保持不变，因此这次只拉起 gateway 和这一个模型。进程起来之后，模型 `/metrics` 上 `mortred_workers_available` 与 `mortred_workers_busy` 之和等于该 `worker_nums`；该进程环境里的 `MORTRED_WORKER_NUMS` 等于该 `worker_nums`，`MORTRED_DECODE_AUTO` 等于该 `decode_auto`。
4. 对 gateway 跑 `scripts/server/http_infer_rps.py --raw`，`--model <id>`，`-c` 等于已部署的 `worker_nums`，`-d 30s`，`--warmup 0s`，不设 `--qps`。测量窗的 `wall_s` 不少于 30。成功 RPS 用该次脚本的 rps：测量窗内 HTTP 200 的次数除以墙钟。现有映射里 HTTP 200 对应状态 OK。

`realesrgan` 的 calibrate 子进程和这次官方压测使用同一张合法 JPEG：无方向标记、彩色、解码后宽 256、高 256。文件放在已忽略的 `logs/p2-11-realesrgan-256.jpg`，只在这个 id 的子进程里设置 `MORTRED_DEMO_IMAGE` 指向它，使 v4 的 `resolve_demo_image` 和 `Path.relative_to` 都能用。其他 id 不设置这个变量，继续用各自现有 demo。v4 的 16 张解码投票网格保持原样。

官方压测打 `http://127.0.0.1:8080/v1/models/<id>/infer`，Bearer 用现有推理 token。压测前后各刮一次该模型进程的 `/metrics`，Bearer 用 supervisor 注入该子进程的 `MORTRED_AUTH_TOKEN`，只留在内存里。gateway `/metrics` 用现有 metrics token。三个 token 都不写入清单，也不提交。

清单每一行（`libface` 允许空白）从这次运行的 pack、进程环境、压测输出和两次 `/metrics` 写成，不手填。`libface` 在 P0-1 已关闭的前提下单独跑；这一行拿不到成功 RPS 时保持空白，其余已经写明字段的行仍勾销。除 `libface` 外有一行不能勾销时，脚本以非 0 退出，并且不把 `docs/perf/coverage_set_http_rps.md` 留成可勾销清单。

代价：九个 id 串行。每个 id 含 v4 默认 30 秒的 worker 网格、v4 现有的解码投票（每张图 10 秒，cpu 与 gpu 各一轮），再加上部署后不少于 30 秒的官方压测。这一张卡上要连续跑一段时间。同时常驻的只有 gateway 和当前这一个模型。HTTP 契约和引擎形状不变。机器 pack、256×256 临时 JPEG 和 calibrate 日志不进提交。

### 否决

用 calibrate 内部采样代替部署后的官方压测。那些采样打在模型端口上，发生在 pack 写回并部署之前，读不到 gateway 阶段，也不能证明进程里的 worker 数和 `MORTRED_DECODE_AUTO`。少跑一轮部署后的 30 秒，换来的数字对不上锁定顺序。

手改 `docs/benchmark-set-a.md`、`docs/perf/yolov8s_decode_backend_rps.md` 或 `README.md` 里的示例命令来凑字段。这些记录是 15 秒、并发与 `worker_nums` 不一致，或缺少 `decode_auto`、p50、错误率和比值。把 `worker_nums` 写进 `conf/server`，或把九个 id 放进同一份常驻 pack，都越过锁定边界。改写 calibrator v4，或重做 gateway 阶段计时，会动已经提交且明确不回退的 v4，以及已关闭的 P1-1。

## 改动面

- 新增 `scripts/coverage_http_report.py`：一次一个 id 的 prepare、calibrate、部署、官方压测和清单写入。`--check` 只跑判定，不拉起进程。
- `scripts/server/http_infer_rps.py`：同一次 `--raw` 测量窗里，从 HTTP 200 响应体的顶层 `server_time_ms` 收集数值，并在打印和 `--out` JSON 里给出这些数值的 p50。缺字段的响应不进入这个 p50。请求编码、并发、时长和 HTTP 200 的成功定义保持现有行为。现有自测继续通过，并补一条进程内响应带 `server_time_ms` 的检查。
- 新增 `docs/perf/coverage_set_http_rps.md`：唯一的勾销清单。

## 明确不改

`scripts/calibrate_pack.py` 与 `365b5dcd` 的 calibrator v4，包括 `--raw` 扫描、解码投票、默认 30 秒才写 pack、拒绝写入 `conf/server` 和 `conf/packs`。`scripts/product_static_bs1.py` 与 `conf/product_static_bs1.json`。九份 `conf/server` 的 `worker_nums` 保持 1。`conf/packs/` 四份示例 pack 保持 seed，不写 `decode_auto`。`conf/model` 产品 toml 保持 `type=tensorrt`，引擎文件名仍是 static batch=1。`src/models/backend/trt_session.cpp` 里每个 session 一次 `createExecutionContext()`。gateway 阶段计时实现，不创建 `/tmp/mortred_stage_timing`。P2-8 的 `conf/onnx_sources.json`。

## 契约

临时 pack 的 `[pack.<id>]` 增加或覆盖 `worker_nums` 和 `decode_auto`（`cpu` 或 `gpu`）。v4 还会写占用戳；这些戳留在 `/tmp` 的那份 pack 里，不进清单的通过字段，也不提交。`conf/server` 的 `worker_nums` 仍是 1。不写 `model_config`，因此 `mobilenetv2` 不会走到 `demo.toml` 里的 `mobilenetv2_cpu.toml`。

HTTP 仍是 gateway 上的 `--raw` 图片字节。响应仍带顶层 `server_time_ms`。测量脚本只读取它。tensor 形状保持 static shape、batch=1。

进程中的 worker 数用该模型 `/metrics` 的 `mortred_workers_available{model="<id>"}` 与 `mortred_workers_busy{model="<id>"}` 之和。`MORTRED_DECODE_AUTO` 与 `MORTRED_WORKER_NUMS` 读该模型进程环境。

错误率 =（该次 `errors` − HTTP 429 次数）/ 请求数。429 次数取 `by_http` 的 `429`，并在行内单独写出；次数为 0 也要写出。这个比率高于 1% 的行不勾销。等于 1% 仍可勾销。

排队直方图是 `mortred_queue_wait_duration_ms`，推理直方图是 `mortred_inference_duration_ms`。行内写下这两次刮取之间各自的 count 增量。两个名字都要出现，排队等待不并进推理耗时。

`mortred_jpeg_decode_total` 的落点：`decode_auto=gpu` 时 `backend="jpeggpu"` 的增量大于 0，并且严格大于 `cpu-reduced` 与 `cpu-full` 之和；`decode_auto=cpu` 时后者之和大于 0，并且严格大于 `jpeggpu`。`fallback` 不算这两条路径。

gateway 阶段 p50 是同一次压测结束后、进程停止前，gateway `/metrics` 上五段 `mortred_gateway_stage_duration_ms{model="gateway",stage="..."}` 的 p50 之和。五段顺序是 `gw_in`、`route`、`auth`、`fwd`、`upstream`。现有按 `model` 紧跟 `le` 的读数对不上带 `stage` 的标签；报告脚本先按 `stage` 取出 bucket 和 count，再交给现有 `histogram_p50`。模型 `server_time_ms` p50 是这次测量窗里 HTTP 200 响应体顶层数值的 p50。五段之和大于该 p50 的一半时，行内写明时间主要花在 p50 最大的那一段；并列取顺序里靠后的名字。已关闭的 P1-1 在累计样本超过 4096 后不再输出 `mortred_gateway_stage_dominant`；30 秒压测可以没有这只 gauge。gauge 在场时，它的 `stage` 必须与行内主导段相同。gauge 不在时，行内按直方图 p50 写明的主导段仍然有效。

`yolov8s` 的比值分母是 `docs/benchmark-h3/yolov8s.txt` 的 Throughput `400.677` qps。其他 id 的分母是各自 `docs/benchmark-h3/<id>.txt` 里 Throughput 那一行的数字。比值是成功 RPS 除以该 Throughput。不设比值下限。

PF5 尚无已记录的合格 HTTP RPS。这张清单就是第一次记录。不要求再跑一次才算通过。

## 性能与资源

一次只有 gateway（:8080）和当前模型进程。supervisor 仍在 :8787。每个 worker 继续持有各自的 TensorRT execution context；本条不把这件事改成共享 context 或 context 池，也不把它当作勾销条件。显存只够当前这一个 id 的 calibrate 与部署。官方压测并发等于该 id 已部署的 `worker_nums`，测量窗 30 秒，不把并发改成固定 8、16 或 32。引擎保持 static shape、batch=1。不在 cpu profile 上取这些数字。

开始每个 id 之前，`127.0.0.1:8080`、`:8787` 和该 id 的模型端口必须连不上。连得上则退出码非 0，不杀掉已有进程，不写清单。本 id 结束时只停止这次脚本拉起的 supervisor，并等到这三个端口再次连不上，再进入下一个 id。

## 验收标准

1. 打开一张清单即可判定。清单列出 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`。每一行单独勾销：`mortredctl calibrate --pack <pack> --write-pack` 的 worker 数扫描和 JPEG 解码投票都走 `--raw`；`[pack.<id>]` 写出 `worker_nums`，以及 `decode_auto` 为 `cpu` 或 `gpu`；`conf/server` 的 `worker_nums` 仍是 1。按该 pack 部署并拉起后，进程中的 worker 数等于 `worker_nums`，`MORTRED_DECODE_AUTO` 等于 `decode_auto`。`scripts/server/http_infer_rps.py --raw` 的并发数等于已部署的 `worker_nums`，测量不少于 30 秒。该行写明硬件、日期、这两项、压测命令、成功 RPS、p50、p99、错误率，以及 `docs/benchmark-h3/<id>.txt` 的 Throughput 和二者比值。429 单独计数，不计入错误率的 1%。同一行引用该次模型进程 `/metrics` 上分开的排队直方图与推理直方图，并且 `mortred_jpeg_decode_total` 落在该行的 `decode_auto` 上。报告在本机 WSL Ubuntu（`/mnt/g/Codex/mortred_model_server`）取得。引擎保持 static shape、batch=1。`yolov8s` 这一行的比值用 400.677 qps。`realesrgan` 这一行用解码后为 256×256 的合法 JPEG 取得成功 RPS。`libface` 的成功 RPS 在 P0-1 关闭后单独勾销；这一行空着时，其余已经写明上述字段的行保持勾销。

2. 这张清单就是 `docs/perf/coverage_set_http_rps.md`。九个 id 按第 1 条的顺序各占一行。除允许空白的 `libface` 外，每一行都能直接读到：硬件（这次 WSL 的 GPU 名称与显存）、压测日期、`worker_nums`、`decode_auto`、对应 `conf/server` 路径及其 `worker_nums` 为 1、calibrate 命令、进程 worker 数、`MORTRED_DECODE_AUTO`、带 `--raw` 且 `-c` 等于 `worker_nums` 且 `-d 30s` 的压测命令、成功 RPS、p50、p99、单独的 429 次数、不含 429 的错误率、Throughput、比值、`mortred_queue_wait_duration_ms` 与 `mortred_inference_duration_ms` 的 count 增量、`mortred_jpeg_decode_total` 的落点。`yolov8s` 行的 Throughput 是 400.677。`realesrgan` 行写明解码后宽 256、高 256。错误率高于 1% 的行不勾销。

3. 同一次官方压测的这一行能读到 gateway 五段耗时和模型 `server_time_ms` p50。gateway 阶段 p50 大于模型 `server_time_ms` p50 的一半时，该行写明时间主要花在哪一段。不创建 `/tmp/mortred_stage_timing`。

4. 这份清单是第一次合格 HTTP RPS 记录。不要求第二次复测才勾销。

5. 清单不要求九个 id 同时出现在同一次 supervisor 状态里，也不要求 `yolov8s` 与 `libface` 同时 ready。

## 测试义务

QA 做白盒。先打开 `docs/perf/coverage_set_http_rps.md`，按验收标准 1 至 5 逐行判定。再读九份被该行点名的 `conf/server` 文件，确认 `worker_nums` 仍是 1。读 `docs/benchmark-h3/yolov8s.txt`，确认 Throughput 行是 400.677 qps。

白盒确认：脚本对每个 id 调用 `mortredctl prepare --id` 和 `mortredctl calibrate --pack <pack> --write-pack`；官方压测命令含 `--raw`，并发等于该行 `worker_nums`，时长 30 秒，URL 在 gateway。`scripts/calibrate_pack.py` 的 worker 扫描和 JPEG 解码投票仍传 `raw=True`。diff 不改 `scripts/calibrate_pack.py`、`scripts/product_static_bs1.py`、`src/models/backend/trt_session.cpp`、gateway 阶段计时，不写 `conf/server`、`conf/packs/`、`conf/local/`、weights 或密钥。清单正文没有 token。`libface` 行空白时，其余已写明字段的行仍算勾销。

QA 执行下面前三条自测，期望退出码都是 0。不把第二次完整九 id 压测当作通过条件。第 4 条由 RD 在取得清单时执行；QA 以打开清单为准，不要求为 PF5 再跑一遍。

## 自测命令

在本机 WSL Ubuntu 执行，仓库路径 `/mnt/g/Codex/mortred_model_server`。Windows 壳跑不了 GPU 命令不算缺 GPU。四条期望退出码都是 0。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 scripts/calibrate_pack.py --self-test
```

```bash
cd /mnt/g/Codex/mortred_model_server
python3 scripts/server/http_infer_rps.py --self-test
```

```bash
cd /mnt/g/Codex/mortred_model_server
python3 scripts/coverage_http_report.py --check
```

```bash
cd /mnt/g/Codex/mortred_model_server
python3 scripts/coverage_http_report.py
```

第 4 条会 prepare、calibrate、拉起并压测。它成功结束时，`docs/perf/coverage_set_http_rps.md` 存在，并且除允许空白的 `libface` 外各行满足验收标准 1 至 3。`--check` 不拉起进程：样例行字段齐全且错误率不高于 1% 为通过；429 被并进 1%、时长低于 30 秒、并发不等于 `worker_nums`、缺少两只直方图、`yolov8s` 的分母不是 400.677、`decode_auto` 不是 `cpu` 或 `gpu`，都为不通过；只有 `libface` 空白且其余八行齐全为通过；其他 id 空白为不通过。

## RD 不得自行决定

- 不得回退 `365b5dcd`，不得修改 `scripts/calibrate_pack.py` 的扫描、投票、写 pack 规则或内部选 worker 门槛，也不得把那些内部门槛写成清单通过条件。
- 不得把结论写入 `conf/server` 或 `conf/packs/`。`conf/server` 的 `worker_nums` 保持 1。不得提交 `conf/local/`、weights、密钥、`/tmp` pack、`logs/` 下的 JPEG 和 calibrate 日志。
- 不得改 `scripts/product_static_bs1.py`，不得改每个 worker 各自的 `IExecutionContext`，不得做成共享 context 或跨 worker context 池。
- 不得重做 gateway 阶段计时，不得创建 `/tmp/mortred_stage_timing`。
- 不得要求 70%、50%、300 RPS、0.1% 错误、60 秒、gateway ≤2ms 或 10%、回归 90%/125%，也不得要求第二次复测。错误率只用 1%，429 单独计数。
- 不得用 base64 JSON 做 worker 扫描、解码投票或官方压测。不得把 calibrate 内部采样的 RPS 写成该行的成功 RPS。
- 不得手写 `worker_nums` 或 `decode_auto`。不得让九个 id 同时常驻。不得把 `yolov8s` 与 `libface` 同时 ready 写进本条通过条件。
- 不得用非 256×256 的图为 `realesrgan` 取得成功 RPS。`yolov8s` 的比值分母不得换成 400.677 以外的数。
- `libface` 必须单独跑。跑不出成功 RPS 时该行留空，不得借其他 id 的数字填上，也不得因此勾销失败其余已写明字段的行。
- 不得恢复 P2-8。不得杀掉不属于这次脚本拉起的进程来腾端口。
