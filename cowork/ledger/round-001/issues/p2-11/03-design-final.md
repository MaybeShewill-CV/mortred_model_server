# 技术方案

- 问题: P2-11
- 版本: 用户选定（以 v2 为底，2026-10-14 重跑）
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-11

## 边界

v1、v2 都未三方通过。不再开第三版确认。PM 对 v2 无条件同意的清单范围留在本版。v2 已经写入的标签事实留在本版，不否决：`src/server/base_server_impl.h` 的 `serve_process` 在渲染前执行 `_m_metrics.set_model(_m_server_uri)`，因此模型进程 `/metrics` 的 `model` 标签是该 id 的 `server_uri`。`scripts/calibrate_pack.py` 的 `_metric_series` 只匹配 `{model="<传入字符串>",le="..."}`。`run_one_point` 现在传入的是目录 id。这两处对不上时，成功请求被标成 `metrics_missing`。

2026-10-14 重跑决定写入做法和验收标准，覆盖本版里与之冲突的 30 秒、并发等于 `worker_nums`、`--warmup 0s`，以及 `pphuman_mobile` 使用 `human_image.jpg` 的句子。选定行为是：

- 删掉上次执行留下的日志、临时 pack 和未完成清单，九个 id 从头再跑。
- `pphuman_mobile` 的 calibrate 与官方压测都使用 `demo_data/model_test_input/scene_segmentation/face_w_mask.jpg`。该文件是 `SOF0` 一次扫描 JPEG，宽 380、高 381，45770 字节。不使用 `human_image.jpg`。不改 `human_image.jpg` 的字节，不改解码路径，不改 16 张解码投票网格。
- worker 数扫描和官方压测的并发等于该点 `worker_nums` 的两倍，测量时长 60 秒，warmup 为 5 秒。`wall_s` 不少于 60。解码投票仍用现有 `DECODE_VOTE_DURATION_S` 10 秒，不把投票网格改成 60 秒。
- 每个 id 完成 calibrate 和官方压测并得到可写入的一行后，立刻把这一行写入 `docs/perf/coverage_set_http_rps.md`，再开始下一个 id。后面的 id 失败时，已写入的行保留。一次新的实跑开始时先清空这份文档。

原先选定的两处读数仍然有效。选定行为是：

- 进程中的 worker 数取官方压测已经产生完成响应、模型进程仍在时的刮取：`mortred_workers_available{model="<server_uri>"}` 与 `mortred_workers_busy{model="<server_uri>"}` 之和等于 pack 里的 `worker_nums`。压测前那一次刮取只做增量基线，不作为 worker 数。`MORTRED_WORKER_NUMS` 与 `MORTRED_DECODE_AUTO` 在进程拉起后即可读。
- `run_one_point` 在 `run_load` 之前抓到的 `/metrics` 里，排队直方图和推理直方图还没有样本。调用 `_hist_delta` 之前，空的 before 按 bucket 为空、count 为 0。压测后对得上 `server_uri` 且增量不少于成功请求一半时，不再因为 before 没有样本被记成 `metrics_missing`。默认 30 秒的 `--write-pack` 因此能写出 `worker_nums` 与 `decode_auto`，解码投票按现有路径运行。after 仍然没有对上的样本时，差值仍为空，`metrics_missing` 仍为真。

九个 id 的 `server_uri` 分别是 `/mortred_ai_server_v1/classification/mobilenetv2`、`/mortred_ai_server_v1/obj_detection/yolov8s`、`/mortred_ai_server_v1/obj_detection/libface`、`/mortred_ai_server_v1/scene_segmentation/pphuman_mobile`、`/mortred_ai_server_v1/ocr/dbnet`、`/mortred_ai_server_v1/feature_embedding/dinov2_vits14`、`/mortred_ai_server_v1/feature_point/superpoint`、`/mortred_ai_server_v1/enhancement/realesrgan`、`/mortred_ai_server_v1/mono_depth_estimation/metric3d_512`。

锁定的可观察变化仍是那一张清单。覆盖集九个 `type=tensorrt` 的 id 没有一份按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告（PF1、PF2、PF5、TP1）。PF4 在本条只保留报告半边：同一次压测要能读到 gateway 阶段耗时和模型 `server_time_ms`。`365b5dcd` 的 calibrator v4 已在仓库里。本条改两处：`run_one_point` 读取直方图时传入 `server_uri`；空的 before 直方图按 0 进入现有差值。不回退它的选 worker 门槛、默认 30 秒才写 pack、拒绝写入 `conf/server` 和 `conf/packs`。不改 `_hist_delta` 的函数体。

P2-8 留在附录，不恢复。P0-1、P1-1、TP2（P2-4）已关闭，不重做。九个 id 各自 prepare、拉起、推理。驻留集仍是 `mortredctl doctor --strict` 允许的那一组，至少 `yolov8s` 与 `libface` 同时常驻；那是已关闭的 TP2，不写进本条通过条件。不要求九个 id 同时常驻。

实现不得改成共享 context，也不得增加跨 worker 的 context 池。这句话是结构约束，不是清单勾销条件。

## 做法

### 推荐

新增 `scripts/coverage_http_report.py`。它按锁定顺序逐个处理九个 id：`mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`。同一时刻只准备、只拉起、只压测当前这一个 id。通过之后把一张清单写到 `docs/perf/coverage_set_http_rps.md`。打不开这一张文件，就不能判定。

每个 id 的顺序固定为四步。

1. `mortredctl prepare --id <id>`。引擎保持现有 static shape、batch=1。已有引擎时沿用现有 prepare 的跳过规则。
2. 在仓库外写一份只含 `[pack.<id>]` 的临时 pack，初值 `worker_nums = 1`，不写 `model_config`。然后执行 `mortredctl calibrate --pack <该文件> --write-pack --duration 60s`。60 秒不低于 v4 的 30 秒写 pack 门槛，结论才会写入 pack。不传 `--from-json`，不传 `--workers`。worker 数扫描的并发等于该点 `worker_nums` 的两倍，warmup 为 5 秒，测量时长用这次传入的 60 秒。JPEG 解码投票仍走 `--raw`，时长仍是现有 10 秒。脚本不增加 base64 路径。`pphuman_mobile` 这次 calibrate 设置 `MORTRED_DEMO_IMAGE=demo_data/model_test_input/scene_segmentation/face_w_mask.jpg`。`dbnet` 这次 calibrate 设置 `MORTRED_DEMO_IMAGE=demo_data/model_test_input/ocr/coco_text_test.jpg`。
3. 从写回的 `[pack.<id>]` 读出 `worker_nums` 和 `decode_auto`。`decode_auto` 必须是 `cpu` 或 `gpu`。再只把这一份 pack 交给 supervisor（`MORTRED_PACK`）。`conf/mortred.toml` 的 `autostart_default = false` 保持不变，因此这次只拉起 gateway 和这一个模型。进程起来之后，读该进程环境里的 `MORTRED_WORKER_NUMS` 与 `MORTRED_DECODE_AUTO`。这两项在拉起后即可读，分别等于该 `worker_nums` 与该 `decode_auto`。这一步不读 worker gauge，不把它们的和写成进程中的 worker 数。
4. 官方压测开始前，刮一次该模型进程的 `/metrics`。这次只作为排队直方图、推理直方图和 `mortred_jpeg_decode_total` 的增量基线。基线里没有的序列按 0。这次刮取上的 worker gauge 即使是 0，也不写入清单的进程 worker 数。然后对 gateway 跑 `scripts/server/http_infer_rps.py --raw`，`--model <id>`，`-c` 等于已部署 `worker_nums` 的两倍，`-d 60s`，`--warmup 5s`，不设 `--qps`。`pphuman_mobile` 增加 `--image demo_data/model_test_input/scene_segmentation/face_w_mask.jpg`。`dbnet` 增加 `--image demo_data/model_test_input/ocr/coco_text_test.jpg`。测量窗的 `wall_s` 不少于 60。成功 RPS 用该次脚本的 rps：测量窗内 HTTP 200 的次数除以墙钟。现有映射里 HTTP 200 对应状态 OK。压测已经产生完成响应、并且这次拉起的模型进程还在时，再刮一次 `/metrics`。进程中的 worker 数是这一次的 `mortred_workers_available{model="<server_uri>"}` 与 `mortred_workers_busy{model="<server_uri>"}` 之和，等于该 `worker_nums`。两只直方图的 count 增量，以及 `mortred_jpeg_decode_total` 的各 `backend` 增量，都是这一次减去基线；基线没有的序列按 0。然后才停止这次拉起的进程。

标签只改 `scripts/calibrate_pack.py` 的 `run_one_point`。该函数已经用 `server_listen` 取出 `uri`。四次调用 `_metric_series` 时，第三个参数改为这个 `uri`，不再传目录 id。`_metric_series` 的正则不改，继续只匹配 `{model="<第三个参数>",le="..."}` 的 bucket，以及 `{model="<第三个参数>"}` 的 `_count`。第三个参数是上面列出的 `server_uri`。排队直方图名仍是 `mortred_queue_wait_duration_ms`，推理直方图名仍是 `mortred_inference_duration_ms`。

`run_one_point` 仍在 `run_load` 之前抓 before。调用 `_hist_delta` 之前，若某一侧 before 的 `_metric_series` 返回空，则传入 `{"buckets": [], "count": 0.0}`。after 侧仍原样传入，不把空 after 换成 0。`_hist_delta` 的函数体不改：after 为空时仍返回空。before 按 0、after 的 count 不少于该次成功请求一半时，该点不再因为 before 没有样本而被记成 `metrics_missing`。目录 id 对不上、或 after 没有样本时，仍记成 `metrics_missing`。不在 `run_load` 之前另发一条推理去制造 before 样本。

`assess_point`、`pick_w_star` 的停止条件保持 `365b5dcd` 已提交的原样，不把那些内部门槛写进清单勾销。某个 id 因这些已提交条件没有写出 `worker_nums` 与 `decode_auto` 时，该行不勾销。除允许空白的 `libface` 外，脚本以非 0 退出，并且不把 `docs/perf/coverage_set_http_rps.md` 留成可勾销清单。`eligible` 为真时，解码投票仍按 v4 现有路径运行。

`realesrgan` 的 calibrate 子进程和这次官方压测使用同一张合法 JPEG：无方向标记、彩色、解码后宽 256、高 256。文件放在已忽略的 `logs/p2-11-realesrgan-256.jpg`，只在这个 id 的子进程里设置 `MORTRED_DEMO_IMAGE` 指向它，使 v4 的 `resolve_demo_image` 和 `Path.relative_to` 都能用。`pphuman_mobile` 的 calibrate 子进程和官方压测使用 `demo_data/model_test_input/scene_segmentation/face_w_mask.jpg`，同样只在这个 id 设置 `MORTRED_DEMO_IMAGE`，官方压测再传 `--image` 指向同一文件。`dbnet` 的 calibrate 子进程和官方压测使用 `demo_data/model_test_input/ocr/coco_text_test.jpg`，只在这个 id 设置 `MORTRED_DEMO_IMAGE`，官方压测再传 `--image` 指向同一文件。该文件是 `SOF0` 一次扫描 JPEG，宽 746、高 530，109127 字节。不改 `railway_ticket.png`。其余 id 不设置这个变量，继续用各自现有 demo。v4 的 16 张解码投票网格保持原样。

官方压测打 `http://127.0.0.1:8080/v1/models/<id>/infer`，Bearer 用现有推理 token。模型进程 `/metrics` 的 Bearer 用 supervisor 注入该子进程的 `MORTRED_AUTH_TOKEN`，只留在内存里。gateway `/metrics` 用现有 metrics token。三个 token 都不写入清单，也不提交。

清单每一行（`libface` 允许空白）从这次运行的 pack、进程环境、压测输出和上面规定的两次 `/metrics` 写成，不手填。进程 worker 数只来自压测后的那一次 gauge 之和。`libface` 在 P0-1 已关闭的前提下单独跑；这一行拿不到成功 RPS 时保持空白，其余已经写明字段的行仍勾销。

每个 id 的一行通过字段检查后立刻写入 `docs/perf/coverage_set_http_rps.md`，再开始下一个 id。一次新的实跑开始时先清空该文件。后面的 id 失败时，已写入的行保留，脚本以非 0 退出。失败行不写入。`libface` 留空时，空白行也立刻写入。

代价：九个 id 串行。每个 id 含 60 秒、并发为该点 worker 数两倍、warmup 5 秒的 worker 网格，加上 v4 现有的解码投票（每张图 10 秒，cpu 与 gpu 各一轮），再加上部署后 60 秒、warmup 5 秒、并发为已部署 `worker_nums` 两倍的官方压测。这一张卡上要连续跑一段时间。同时常驻的只有 gateway 和当前这一个模型。HTTP 契约和引擎形状不变。`run_one_point` 改传入标签，并把空的 before 直方图按 0 传入现有差值。不改选 worker 门槛。机器 pack、256×256 临时 JPEG、口罩图副本和 calibrate 日志不进提交。

### 否决

改 `serve_process` 里的 `_m_metrics.set_model(_m_server_uri)`，让进程改吐目录 id。代价是模型进程上每一条带 `model` 的序列都改标签。最小改动是让 `run_one_point` 传入已经渲染的 `server_uri`。改渲染侧不能关掉本条。

进程刚 ready、官方压测还没有完成响应时，就把两只 worker gauge 相加当作进程中的 worker 数。这两只 gauge 只在推理完成路径里离开初值 0，这次相加在正确部署上仍是 0。

保持「只改标签、空 before 仍让 `_hist_delta` 得到空」。压测前直方图没有样本，成功请求仍被记成 `metrics_missing`，默认 30 秒的 `--write-pack` 仍不写 `worker_nums` 与 `decode_auto`。

在 `run_load` 之前先打一条推理，用来让 before 直方图出现。用户选定的是空 before 按 0，不增加这一次请求。

用 calibrate 内部采样代替部署后的官方压测。那些采样打在模型端口上，发生在 pack 写回并部署之前，读不到 gateway 阶段，也不能证明进程里的 worker 数和 `MORTRED_DECODE_AUTO`。手改已有基准文档来凑字段，对不上锁定顺序。把 `worker_nums` 写进 `conf/server`，或把九个 id 放进同一份常驻 pack，都越过锁定边界。放宽 `assess_point` 或 `pick_w_star`，改 `_hist_delta` 的函数体，或重做 gateway 阶段计时，会动已经提交且明确不回退的 v4 门槛，以及已关闭的 P1-1。

## 改动面

- `scripts/calibrate_pack.py` 的 `run_one_point`：四次 `_metric_series` 的第三个参数改为同一次 `server_listen` 返回的 `uri`。调用 `_hist_delta` 前，空的 before 换成 `{"buckets": [], "count": 0.0}`。`_hist_delta` 函数体不改。`self_test` 增加断言：`server_uri` 对得上这条直方图，目录 id 对不上；空 before 加有样本的 after，count 增量等于 after 的 count；after 为空时增量仍为空。已有停止原因断言、拒绝写入 `conf/server` 和 `conf/packs` 的断言保持原样。
- 新增 `scripts/coverage_http_report.py`：一次一个 id 的 prepare、calibrate、部署、官方压测和清单写入。读模型直方图时把该行 `server_uri` 交给现有 `_metric_series`，不把目录 id 传进去。直方图和 `mortred_jpeg_decode_total` 的增量以压测前刮取为基线，没有的序列按 0。worker 两只 gauge 只用压测后、进程仍在时的那一次、同一条 `server_uri` 相加。`--check` 只跑判定，不拉起进程。
- `scripts/server/http_infer_rps.py`：同一次 `--raw` 测量窗里，从 HTTP 200 响应体的顶层 `server_time_ms` 收集数值，并在打印和 `--out` JSON 里给出这些数值的 p50。缺字段的响应不进入这个 p50。请求编码、并发、时长和 HTTP 200 的成功定义保持现有行为。现有自测继续通过，并补一条进程内响应带 `server_time_ms` 的检查。
- 新增 `docs/perf/coverage_set_http_rps.md`：唯一的勾销清单。

## 明确不改

`_metric_series` 的正则。`_hist_delta` 的函数体。`assess_point`、`pick_w_star`、`occupancy_fields_from_report` 的跳过条件、`write_pack_fields`、`write_pack_worker_nums`。默认 30 秒才写 pack。拒绝写入 `conf/server` 和 `conf/packs`。worker 扫描与 JPEG 解码投票的 `--raw`。解码投票的网格和时长。`histogram_p50`。`scripts/product_static_bs1.py` 与 `conf/product_static_bs1.json`。九份 `conf/server` 的 `worker_nums` 保持 1。`conf/packs/` 四份示例 pack 保持 seed，不写 `decode_auto`。`conf/model` 产品 toml 保持 `type=tensorrt`，引擎文件名仍是 static batch=1。`src/server/base_server_impl.h` 的 `set_model(_m_server_uri)`。`src/models/backend/trt_session.cpp` 里每个 session 一次 `createExecutionContext()`。gateway 阶段计时实现，不创建 `/tmp/mortred_stage_timing`。P2-8 的 `conf/onnx_sources.json`。

## 契约

临时 pack 的 `[pack.<id>]` 增加或覆盖 `worker_nums` 和 `decode_auto`（`cpu` 或 `gpu`）。v4 还会写占用戳；这些戳留在 `/tmp` 的那份 pack 里，不进清单的通过字段，也不提交。`conf/server` 的 `worker_nums` 仍是 1。不写 `model_config`，因此 `mobilenetv2` 不会走到 `demo.toml` 里的 `mobilenetv2_cpu.toml`。

HTTP 仍是 gateway 上的 `--raw` 图片字节。响应仍带顶层 `server_time_ms`。测量脚本只读取它。tensor 形状保持 static shape、batch=1。

模型进程 `/metrics` 的 `model` 标签取该行点名的 `conf/server` 里的 `server_uri`。当前九个值是：

| id | conf/server | server_uri |
| --- | --- | --- |
| mobilenetv2 | `conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml` | `/mortred_ai_server_v1/classification/mobilenetv2` |
| yolov8s | `conf/server/object_detection/yolov8/yolov8s_server_config.toml` | `/mortred_ai_server_v1/obj_detection/yolov8s` |
| libface | `conf/server/object_detection/libface_det/libface_server_config.toml` | `/mortred_ai_server_v1/obj_detection/libface` |
| pphuman_mobile | `conf/server/scene_segmentation/pphuman_seg/pphuman_mobile_server_config.toml` | `/mortred_ai_server_v1/scene_segmentation/pphuman_mobile` |
| dbnet | `conf/server/ocr/dbnet/dbnet_server_config.toml` | `/mortred_ai_server_v1/ocr/dbnet` |
| dinov2_vits14 | `conf/server/feature_embedding/dinov2/dinov2_vits14_server_config.toml` | `/mortred_ai_server_v1/feature_embedding/dinov2_vits14` |
| superpoint | `conf/server/feature_point/superpoint/superpoint_server_config.toml` | `/mortred_ai_server_v1/feature_point/superpoint` |
| realesrgan | `conf/server/enhancement/real_esrgan/realesrgan_server_config.toml` | `/mortred_ai_server_v1/enhancement/realesrgan` |
| metric3d_512 | `conf/server/mono_depth_estimation/metric3d/metric3d_512_server_config.toml` | `/mortred_ai_server_v1/mono_depth_estimation/metric3d_512` |

进程中的 worker 数是官方压测已经产生完成响应、模型进程仍在时刮到的 `mortred_workers_available{model="<server_uri>"}` 与 `mortred_workers_busy{model="<server_uri>"}` 之和。压测前那一次的这两只 gauge 不进入这个和。排队直方图与推理直方图的 bucket 形状是 `{model="<server_uri>",le="..."}`，count 形状是 `{model="<server_uri>"}`。目录 id 上的同名序列不进入 worker 之和，也不进入这两只直方图。`MORTRED_DECODE_AUTO` 与 `MORTRED_WORKER_NUMS` 读该模型进程环境，拉起后即可读。

错误率 =（该次 `errors` − HTTP 429 次数）/ 请求数。429 次数取 `by_http` 的 `429`，并在行内单独写出；次数为 0 也要写出。这个比率高于 1% 的行不勾销。等于 1% 仍可勾销。

行内写下压测后刮取减去压测前基线之后，排队直方图与推理直方图各自的 count 增量。压测前没有该序列时，增量等于压测后的 count。两个名字都要出现，排队等待不并进推理耗时。

`mortred_jpeg_decode_total` 没有 `model` 标签，只有 `backend`。计数为 0 的 backend 不出现在正文里，压测前因此可以没有对应行，增量按 0 起算。落点：`decode_auto=gpu` 时 `backend="jpeggpu"` 的增量大于 0，并且严格大于 `cpu-reduced` 与 `cpu-full` 之和；`decode_auto=cpu` 时后者之和大于 0，并且严格大于 `jpeggpu`。`fallback` 不算这两条路径。

gateway 阶段 p50 是同一次压测结束后、进程停止前，gateway `/metrics` 上五段 `mortred_gateway_stage_duration_ms{model="gateway",stage="..."}` 的 p50 之和。五段顺序是 `gw_in`、`route`、`auth`、`fwd`、`upstream`。现有按 `model` 紧跟 `le` 的读数对不上带 `stage` 的标签；报告脚本先按 `stage` 取出 bucket 和 count，再交给现有 `histogram_p50`。不改 `_metric_series` 去吞带 `stage` 的标签。模型 `server_time_ms` p50 是这次测量窗里 HTTP 200 响应体顶层数值的 p50。五段之和大于该 p50 的一半时，行内写明时间主要花在 p50 最大的那一段；并列取顺序里靠后的名字。已关闭的 P1-1 在累计样本超过 4096 后不再输出 `mortred_gateway_stage_dominant`；30 秒压测可以没有这只 gauge。gauge 在场时，它的 `stage` 必须与行内主导段相同。gauge 不在时，行内按直方图 p50 写明的主导段仍然有效。

`yolov8s` 的比值分母是 `docs/benchmark-h3/yolov8s.txt` 的 Throughput `400.677` qps。其他 id 的分母是各自 `docs/benchmark-h3/<id>.txt` 里 Throughput 那一行的数字。比值是成功 RPS 除以该 Throughput。不设比值下限。

PF5 尚无已记录的合格 HTTP RPS。这张清单就是第一次记录。不要求再跑一次才算通过。

## 性能与资源

一次只有 gateway（:8080）和当前模型进程。supervisor 仍在 :8787。每个 worker 继续持有各自的 TensorRT execution context。本条不把这件事写成勾销条件。实现不得改成共享 context，也不得增加跨 worker 的 context 池。显存只够当前这一个 id 的 calibrate 与部署。官方压测并发等于该 id 已部署 `worker_nums` 的两倍，测量窗 60 秒，warmup 5 秒。并发随该 id 的 `worker_nums` 变化，不是固定的 8、16 或 32。引擎保持 static shape、batch=1。不在 cpu profile 上取这些数字。

开始每个 id 之前，`127.0.0.1:8080`、`:8787` 和该 id 的模型端口必须连不上。连得上则退出码非 0，不杀掉已有进程，不写清单。本 id 结束时只停止这次脚本拉起的 supervisor，并等到这三个端口再次连不上，再进入下一个 id。

## 验收标准

1. 打开一张清单即可判定。清单列出 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`。每一行单独勾销：`mortredctl calibrate --pack <pack> --write-pack` 的 worker 数扫描和 JPEG 解码投票都走 `--raw`；`[pack.<id>]` 写出 `worker_nums`，以及 `decode_auto` 为 `cpu` 或 `gpu`；`conf/server` 的 `worker_nums` 仍是 1。按该 pack 部署并拉起后，`MORTRED_WORKER_NUMS` 等于 `worker_nums`，`MORTRED_DECODE_AUTO` 等于 `decode_auto`。进程中的 worker 数等于 `worker_nums`，并且这个数来自官方压测已经产生完成响应、进程停止之前的那一次 gauge 之和；压测前的刮取不作为这个数。`scripts/server/http_infer_rps.py --raw` 的并发数等于已部署 `worker_nums` 的两倍，测量不少于 60 秒，warmup 为 5 秒。`pphuman_mobile` 这一行的压测图是 `demo_data/model_test_input/scene_segmentation/face_w_mask.jpg`。该行写明硬件、日期、这两项、压测命令、成功 RPS、p50、p99、错误率，以及 `docs/benchmark-h3/<id>.txt` 的 Throughput 和二者比值。429 单独计数，不计入错误率的 1%。同一行引用该次模型进程 `/metrics` 上分开的排队直方图与推理直方图，并且 `mortred_jpeg_decode_total` 落在该行的 `decode_auto` 上。报告在本机 WSL Ubuntu 的问题 worktree（`/mnt/g/Codex/mortred_wt_r1_p2_11`）取得。引擎保持 static shape、batch=1。`yolov8s` 这一行的比值用 400.677 qps。`realesrgan` 这一行用解码后为 256×256 的合法 JPEG 取得成功 RPS。`libface` 的成功 RPS 在 P0-1 关闭后单独勾销；这一行空着时，其余已经写明上述字段的行保持勾销。

2. 这张清单就是 `docs/perf/coverage_set_http_rps.md`。九个 id 按第 1 条的顺序各占一行。除允许空白的 `libface` 外，每一行都能直接读到：硬件（这次 WSL 的 GPU 名称与显存）、压测日期、`worker_nums`、`decode_auto`、对应 `conf/server` 路径及其 `worker_nums` 为 1、该文件中的 `server_uri`、calibrate 命令、进程 worker 数、`MORTRED_DECODE_AUTO`、带 `--raw` 且 `-c` 等于 `worker_nums` 的两倍且 `-d 60s` 且 `--warmup 5s` 的压测命令、成功 RPS、p50、p99、单独的 429 次数、不含 429 的错误率、Throughput、比值、`mortred_queue_wait_duration_ms` 与 `mortred_inference_duration_ms` 的 count 增量、`mortred_jpeg_decode_total` 的落点。进程 worker 数等于压测后刮取的两只 gauge 之和，并写明该数不是压测前刮取。worker 之和与这两只直方图的 `model` 标签等于该行写出的 `server_uri`，并等于契约表里该 id 的值。目录 id 上的同名序列不计入这些字段。压测前没有直方图样本时，行内 count 增量等于压测后的 count。`yolov8s` 行的 Throughput 是 400.677。`realesrgan` 行写明解码后宽 256、高 256。错误率高于 1% 的行不勾销。

3. 同一次官方压测的这一行能读到 gateway 五段耗时和模型 `server_time_ms` p50。gateway 阶段 p50 大于模型 `server_time_ms` p50 的一半时，该行写明时间主要花在哪一段。不创建 `/tmp/mortred_stage_timing`。

4. 这份清单是第一次合格 HTTP RPS 记录。不要求第二次复测才勾销。

5. 清单不要求九个 id 同时出现在同一次 supervisor 状态里，也不要求 `yolov8s` 与 `libface` 同时 ready。

## 测试义务

QA 做白盒。先打开 `docs/perf/coverage_set_http_rps.md`，按验收标准 1 至 5 逐行判定。再读九份被该行点名的 `conf/server` 文件，确认 `worker_nums` 仍是 1，且文件里的 `server_uri` 与该行写出的标签一致，并与契约表一致。读 `docs/benchmark-h3/yolov8s.txt`，确认 Throughput 行是 400.677 qps。

白盒确认：脚本对每个 id 调用 `mortredctl prepare --id` 和 `mortredctl calibrate --pack <pack> --write-pack`；官方压测命令含 `--raw`，并发等于该行 `worker_nums` 的两倍，时长 60 秒，warmup 5 秒，URL 在 gateway。`pphuman_mobile` 的 `--image` 是 `face_w_mask.jpg`。`run_one_point` 里四次 `_metric_series` 的第三个参数是 `server_listen` 得到的 `uri`。空的 before 在调用 `_hist_delta` 前换成 count 为 0 的序列，`_hist_delta` 的函数体没有改。`_metric_series` 的正则、`assess_point`、`pick_w_star`、写 pack 的拒绝路径、默认 30 秒才写 pack、worker 扫描和 JPEG 解码投票的 `raw=True` 都没有改。`serve_process` 仍在渲染前执行 `_m_metrics.set_model(_m_server_uri)`。清单脚本把进程 worker 数只从压测后、进程仍在时的 gauge 之和写入；压测前刮取不写入该字段。diff 不改 `scripts/product_static_bs1.py`、`src/models/backend/trt_session.cpp`、gateway 阶段计时，不写 `conf/server`、`conf/packs/`、`conf/local/`、weights 或密钥。清单正文没有 token。`libface` 行空白时，其余已写明字段的行仍算勾销。

QA 执行下面前三条自测，期望退出码都是 0。不把第二次完整九 id 压测当作通过条件。第 4 条由 RD 在取得清单时执行；QA 以打开清单为准，不要求为 PF5 再跑一遍。

## 自测命令

在本机 WSL Ubuntu 执行。第 4 步的 worktree 路径是 `/mnt/g/Codex/mortred_wt_r1_p2_11`。Windows 壳跑不了 GPU 命令不算缺 GPU。四条期望退出码都是 0。

```bash
cd /mnt/g/Codex/mortred_wt_r1_p2_11
python3 scripts/calibrate_pack.py --self-test
```

期望退出码 0。除原有断言外：一条合成的 `/metrics` 正文使用 `model="/mortred_ai_server_v1/classification/mobilenetv2"` 的 `mortred_queue_wait_duration_ms` bucket 与 count。把这个 `server_uri` 交给 `_metric_series` 能得到该序列；把 `mobilenetv2` 交给同一次调用得到空。`_hist_delta({"buckets": [], "count": 0.0}, 该序列)` 的 count 等于该序列的 count。`_hist_delta` 的第二个参数为 `None` 时返回 `None`。已有的 `pick_w_star` 停止原因断言保持原样。拒绝写入 `conf/server` 和 `conf/packs` 的断言保持原样。

```bash
cd /mnt/g/Codex/mortred_wt_r1_p2_11
python3 scripts/server/http_infer_rps.py --self-test
```

期望退出码 0。

```bash
cd /mnt/g/Codex/mortred_wt_r1_p2_11
python3 scripts/coverage_http_report.py --check
```

期望退出码 0。`--check` 不拉起进程。样例行字段齐全、写出的 `server_uri` 与契约表一致、错误率不高于 1% 为通过。进程 worker 数等于样例里压测后那一次的 gauge 之和，并标明不是压测前刮取；把压测前的 0 当作进程 worker 数则不通过。worker 与两只直方图只认该 `server_uri`；同一份样例若只有 `{model="<目录 id>"}` 则不通过。压测前直方图为空、压测后 count 为正时，行内增量等于压测后的 count 为通过。429 被并进 1%、时长低于 60 秒、warmup 不是 5 秒、并发不等于 `worker_nums` 的两倍、缺少两只直方图、`yolov8s` 的分母不是 400.677、`decode_auto` 不是 `cpu` 或 `gpu`，都为不通过。`pphuman_mobile` 的压测图不是 `face_w_mask.jpg` 为不通过。只有 `libface` 空白且其余八行齐全为通过；其他 id 空白为不通过。

```bash
cd /mnt/g/Codex/mortred_wt_r1_p2_11
python3 scripts/coverage_http_report.py
```

期望退出码 0。第 4 条会 prepare、calibrate、拉起并压测。它成功结束时，`docs/perf/coverage_set_http_rps.md` 存在，并且除允许空白的 `libface` 外各行满足验收标准 1 至 3。

## RD 不得自行决定

- 不得回退 `365b5dcd` 的选 worker 门槛、默认不少于 30 秒才写 pack、拒绝写入 `conf/server` 和 `conf/packs`。这次实跑传入 `--duration 60s`，仍满足该门槛。`scripts/calibrate_pack.py` 里允许的产品行为改动是：`run_one_point` 传给 `_metric_series` 的标签改为 `server_uri`；调用 `_hist_delta` 前把空的 before 换成 count 为 0 的序列；本次 worker 数扫描的并发改为该点 worker 数的两倍，warmup 改为 5 秒。不得修改 `_hist_delta` 函数体、解码投票网格和 10 秒时长、写 pack 规则或 `assess_point`、`pick_w_star`。不得把那些内部门槛写成清单通过条件。
- 不得在 `run_load` 之前另发推理去制造 before 样本。不得把空的 after 换成 0。
- 清单上的进程 worker 数只取官方压测已经产生完成响应、进程停止之前的 gauge 之和。压测前那一次不得写入这个字段。`MORTRED_WORKER_NUMS` 与 `MORTRED_DECODE_AUTO` 在拉起后读取。
- 不得改 `serve_process` 的 `_m_metrics.set_model(_m_server_uri)`。不得让清单或 calibrate 用目录 id 去匹配 `model` 标签。
- 不得把结论写入 `conf/server` 或 `conf/packs/`。`conf/server` 的 `worker_nums` 保持 1。不得提交 `conf/local/`、weights、密钥、`/tmp` pack、`logs/` 下的 JPEG 和 calibrate 日志。
- 不得改 `scripts/product_static_bs1.py`，不得改每个 worker 各自的 `IExecutionContext`，不得做成共享 context 或跨 worker context 池。共享 context 不是清单通过条件。
- 不得重做 gateway 阶段计时，不得创建 `/tmp/mortred_stage_timing`。不得改 `_metric_series` 去匹配带 `stage` 的 gateway 标签。
- 不得要求 70%、50%、300 RPS、0.1% 错误、gateway ≤2ms 或 10%、回归 90%/125%，也不得要求第二次复测。错误率只用 1%，429 单独计数。测量时长按本次决定是 60 秒，不是额外的性能门槛。
- 不得用 base64 JSON 做 worker 扫描、解码投票或官方压测。不得把 calibrate 内部采样的 RPS 写成该行的成功 RPS。
- 不得手写 `worker_nums` 或 `decode_auto`。不得让九个 id 同时常驻。不得把 `yolov8s` 与 `libface` 同时 ready 写进本条通过条件。
- 不得用非 256×256 的图为 `realesrgan` 取得成功 RPS。`yolov8s` 的比值分母不得换成 400.677 以外的数。
- `libface` 必须单独跑。跑不出成功 RPS 时该行留空，不得借其他 id 的数字填上，也不得因此勾销失败其余已写明字段的行。
- 不得恢复 P2-8。不得杀掉不属于这次脚本拉起的进程来腾端口。

## 确认

| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v2 | `cowork/ledger/round-001/issues/p2-11/03-discussion/v2-sign-pm.md` |
| 技术负责人 | 不同意 | `mortred_workers_available` 与 `mortred_workers_busy` 只在推理完成路径里写入。进程刚 ready、还没有完成一次推理时，两只 gauge 仍是初值 0，相加不等于 `worker_nums`。进程中的 worker 数必须取官方压测已经产生完成响应、模型进程仍在时的刮取；压测前那一次刮取只用来做直方图增量，不能当作 worker 数。 | v2 | `cowork/ledger/round-001/issues/p2-11/03-discussion/v2-sign-tech-lead.md` |
| RD | 不同意 | `run_one_point` 在 `run_load` 之前抓 `/metrics`。`mortred_queue_wait_duration_ms` 与 `mortred_inference_duration_ms` 在第一次 `observe_queue_wait_ms` / `observe_inference_duration_ms` 之前没有 bucket 和 count，`render_histogram_locked` 只遍历已经有样本的 `hist.buckets`。这次 before 上 `_metric_series` 返回空，`_hist_delta` 返回空，成功请求的 `metrics_missing` 仍为真。第三个参数改成 `server_listen` 的 `uri` 只能让 after 对上 `{model="<server_uri>",le="..."}` 和 `{model="<server_uri>"}`。方案只允许改这一个参数，不准改差值、`assess_point`、`pick_w_star` 和写 pack 规则。默认 30 秒的 `--write-pack` 因此仍不写 `worker_nums` 与 `decode_auto`，解码投票不跑。 | v2 | `cowork/ledger/round-001/issues/p2-11/03-discussion/v2-sign-rd.md` |
| 用户 | 选定 | 无 | 以 v2 为底，写入两处读数 | 本文件「用户决定」 |

## 用户决定

- 日期: 2026-10-14
- 选用版本: 以 v2 为底，写入两处读数。不再开第三版确认。状态保持 running，进入第 4 步。
- 决定原文: 采用你的建议方案解决冲突的地方，然后继续执行cowork工作。
- 未选用:

v1 未被选中。该版不改 `scripts/calibrate_pack.py`，部署后按目录 id 去加 worker gauge。RD 对 v1 的条件原文：

模型进程 `/metrics` 的 `model` 标签是 `server_uri`，不是目录 id。`serve_process` 在渲染前执行 `set_model(_m_server_uri)`，九个 id 的实际标签分别是 `/mortred_ai_server_v1/classification/mobilenetv2`、`/mortred_ai_server_v1/obj_detection/yolov8s`、`/mortred_ai_server_v1/obj_detection/libface`、`/mortred_ai_server_v1/scene_segmentation/pphuman_mobile`、`/mortred_ai_server_v1/ocr/dbnet`、`/mortred_ai_server_v1/feature_embedding/dinov2_vits14`、`/mortred_ai_server_v1/feature_point/superpoint`、`/mortred_ai_server_v1/enhancement/realesrgan`、`/mortred_ai_server_v1/mono_depth_estimation/metric3d_512`。不存在 `mortred_workers_available{model="<id>"}` 或 `mortred_workers_busy{model="<id>"}`。v4 的 `_metric_series` 只匹配 `{model="<id>",le="..."}`；成功请求因此被标成 `metrics_missing`，`eligible` 为假，`occupancy_fields_from_report` 不写 `worker_nums` 和 `decode_auto`，解码投票也不会跑。方案禁止改 `scripts/calibrate_pack.py`，改动面也不包含这条标签。

v2 未经这两处读数修订的句子未被选中：进程起来之后就把两只 worker gauge 相加，并当作进程中的 worker 数。`run_one_point` 只改传入标签，空的 before 仍使 `_hist_delta` 返回空。

只采纳技术负责人的条件、不处理空 before，未被选中。只采纳 RD 的条件、仍在压测前把 gauge 之和当作 worker 数，未被选中。停止本轮未被选中。

## 用户决定（重跑）

- 日期: 2026-10-14
- 选用: 在已选定方案上改测量参数和 `pphuman_mobile` 的测试图，删掉上次执行产物后从头重跑。不再开第三版确认。状态改为 running，从第 4 步重新实现。
- 决定原文: 删除 P2-11 执行过程中产生的一切中间记录，重新从头开始执行 P2-11。pphuman_mobile 的测试图像换成 `demo_data/model_test_input/scene_segmentation/face_w_mask.jpg`。calibrate 的 worker 数扫描和官方压测的并发数是该点 `worker_nums` 的两倍，duration 为 60 秒，warmup 为 5 秒。
- 未选用: 继续用 `human_image.jpg`；并发等于 `worker_nums`；测量 30 秒；`--warmup 0s`；把 16 张解码投票网格改成 60 秒。上次因渐进 JPEG 停住的偏差确认不再作为重跑的约束。

## 用户决定（继续）

- 日期: 2026-10-14
- 选用: 用已经另存的 `demo_data/model_test_input/ocr/coco_text_test.jpg` 作为 `dbnet` 的测试图，保留清单里已写入的四行，从 `dbnet` 继续。不再开第三版确认。状态改为 running，继续第 4 步。
- 决定原文: 继续执行。`dbnet` 的 calibrate 和官方压测都使用这张另存的 JPEG。不改 `railway_ticket.png`，不改解码路径，不改 16 张投票网格。`mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile` 四行保留，不重跑。从 `dbnet` 起跑完剩余 id。
- 未选用: 继续用 `railway_ticket.png`；清空前四行后从头重跑；把一次探测请求的 7 个框写进清单。那一次探测只说明这张 JPEG 能走出 `jpeggpu`，不能代替 60 秒官方压测。

## 用户决定（再跑 superpoint）

- 日期: 2026-10-14
- 选用: 保留清单已有 6 行，从 `superpoint` 再跑一次。`P99_CAP_MS["superpoint"]` 仍是 25.0。不改 `pick_w_star`，不改 `assess_point`，不改 demo `demo_data/model_test_input/feature_point/test.png`，不手填。状态改为 running，继续第 4 步。
- 决定原文: 再做一次 superpoint 模型。
- 做法: 执行 `python3 scripts/coverage_http_report.py --from superpoint`。该入口要求清单正好是 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`，并保留这些行。`superpoint` 写出 `worker_nums` 与 `decode_auto` 后，同一命令按原方案继续 `realesrgan`、`metric3d_512`。`superpoint` 再次因 `latency_budget` 写不出 `decode_auto` 时停住，不跑后面的 id，不改 25ms 上限，不手填。
- 未选用: 提高 25ms 上限；更换 `superpoint` 的 demo；手填这一行；跳过 `superpoint`；清空前 6 行后从头重跑。

## 用户决定（superpoint 上限 40）

- 日期: 2026-10-15
- 选用: 只把 `P99_CAP_MS["superpoint"]` 从 25.0 改为 40.0。不改 `pick_w_star`，不改 `assess_point` 的判定结构，不改 demo `demo_data/model_test_input/feature_point/test.png`，不改并发、60 秒和 5 秒 warmup。不改 `realesrgan` 的 200.0，不改 `metric3d_512` 的 150.0。保留清单已有 6 行，从 `superpoint` 续跑。状态改为 running，继续第 4 步。
- 决定原文: 按照你的意见继续推进 cowork 工作。
- 做法: 在问题 worktree 的 `scripts/calibrate_pack.py` 里把 `P99_CAP_MS["superpoint"]` 改为 40.0。其余种子不动。然后执行 `python3 scripts/coverage_http_report.py --from superpoint`。该入口保留 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`。`superpoint` 写出 `worker_nums` 与 `decode_auto` 后，同一命令继续 `realesrgan`、`metric3d_512`。后面的 id 若因各自已有种子上限停在 `latency_budget`，停住，不提前放宽那两个上限，不手填。
- 未选用: 保持 25.0；改成 35；改 `pick_w_star` 让 w=1 超限仍写入；更换 demo；重写 NMS 或描述子序列化；提前放宽 `realesrgan` 和 `metric3d_512`。

## 用户决定（superpoint JPEG）

- 日期: 2026-10-15
- 选用: 用另存的 `demo_data/model_test_input/feature_point/test.jpg` 作为 `superpoint` 的测试图。保留清单已有 6 行，从 `superpoint` 继续。`P99_CAP_MS["superpoint"]` 仍是 40.0。不改 `demo_data/model_test_input/feature_point/test.png` 的字节，不改解码路径，不改 16 张投票网格，不手填，不放宽落点。不改 `realesrgan` 的 200.0，不改 `metric3d_512` 的 150.0。不再开第三版确认。状态改为 running，继续第 4 步。
- 决定原文: 重新输出一张jpeg，superpoint测试重新使用这个jpeg。
- 做法: 这张 JPEG 由现有 `test.png` 的同一批像素重新编码。文件是 `SOF0` 一次扫描 JPEG，无 EXIF、无方向标记，宽 640、高 480，55892 字节，首两字节 `FF D8`。只在 `superpoint` 的 calibrate 子进程设置 `MORTRED_DEMO_IMAGE=demo_data/model_test_input/feature_point/test.jpg`，官方压测再传 `--image` 指向同一文件。不改 `test.png`。然后执行 `python3 scripts/coverage_http_report.py --from superpoint`。该入口保留 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`。`superpoint` 写出 `worker_nums` 与 `decode_auto` 后，同一命令继续 `realesrgan`、`metric3d_512`。后面的 id 若因各自已有种子上限停在 `latency_budget`，停住，不提前放宽那两个上限，不手填。
- 未选用: 继续把 `test.png` 的 PNG 字节当作官方 `--raw` 请求体；改写 `test.png` 文件本身；改解码路径；改 16 张投票网格；手填 `decode_auto`；放宽落点检查；改 40.0、200.0 或 150.0；清空前 6 行后从头重跑。

## 用户决定（realesrgan 上限 400 与 metric3d JPEG）

- 日期: 2026-10-15
- 选用: 只把 `P99_CAP_MS["realesrgan"]` 从 200.0 改为 400.0。`metric3d_512` 的 calibrate 和官方压测改用另存的 `demo_data/model_test_input/mono_depth_estimation/0000000005.jpg`。`P99_CAP_MS["metric3d_512"]` 仍是 150.0，`P99_CAP_MS["superpoint"]` 仍是 40.0。不改 `pick_w_star`，不改 `assess_point` 的判定结构，不改并发、60 秒和 5 秒 warmup，不改解码路径，不改 16 张投票网格，不手填，不放宽落点。不改 `0000000005.png` 的字节。保留清单已有 7 行，从 `realesrgan` 续跑。不再开第三版确认。状态改为 running，继续第 4 步。
- 决定原文: 按你的建议方案继续完成这一轮工作。
- 做法: 在问题 worktree 的 `scripts/calibrate_pack.py` 里把 `P99_CAP_MS["realesrgan"]` 改为 400.0。`superpoint` 保持 40.0，`metric3d_512` 保持 150.0。`0000000005.jpg` 由现有 `demo_data/model_test_input/mono_depth_estimation/0000000005.png` 的同一批像素重新编码，是 `SOF0` 一次扫描 JPEG，无 EXIF、无方向标记，宽 1242、高 375，147444 字节，首两字节 `FF D8`。不改这张 PNG。只在 `metric3d_512` 的 calibrate 子进程设置 `MORTRED_DEMO_IMAGE=demo_data/model_test_input/mono_depth_estimation/0000000005.jpg`，官方压测再传 `--image` 指向同一文件。然后执行 `python3 scripts/coverage_http_report.py --from realesrgan`。该入口保留 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`。`realesrgan` 写出 `worker_nums` 与 `decode_auto` 后，同一命令继续 `metric3d_512`。`metric3d_512` 若因已有 150.0 停在 `latency_budget`，停住，不提前放宽这个上限，不手填。
- 未选用: 保持 200.0；把上限改成 350；改 `pick_w_star` 让 w=1 超限仍写入；重写 Real-ESRGAN 的后处理；提前把 `metric3d_512` 的 150.0 放宽；继续用 `0000000005.png` 的 PNG 字节；改写这张 PNG；清空前 7 行后从头重跑。
