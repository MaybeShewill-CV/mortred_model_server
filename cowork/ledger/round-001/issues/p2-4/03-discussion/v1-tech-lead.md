# 技术方案

- 问题: P2-4
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-4

## 边界

锁定句照录。没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求、且两者都有成功响应的记录（TP2）。场景是驻留集里 `yolov8s` 与 `libface` 同时 ready，对两者并发请求。现在 `scripts/bench_set_a/http_raw.py` 与 `scripts/calibrate_pack.py` 的 `run_one_point()` 每次只拉起一个模型进程。`measure_joint_once()` 可以同时拉起多个进程，但不发推理。`test/gateway_e2e_test.cc` 的 `dead_upstream_maps_to_503` 只停一个 fake 上游。`test/` 下没有同时请求这两个 id 的用例。仓库里没有一份同时 ready、并发请求、两边都成功的记录。改完后能观察到的变化：同一份记录里，`yolov8s` 与 `libface` 同时为 ready，并发请求两者，两者都有成功响应。先停掉其中一个再压另一个，不能当作这次验收。

上一版审查意见无。

本方案只增加这一次实跑和它写出的一份记录，外加锁住「怎样才算这份记录」的判定样例。驻留集在这次实跑里就是 supervisor 用一份只含这两个 id 的 pack 拉起的进程。`conf/mortred.toml` 的 `start_concurrency = 1` 保持不变，两个模型仍按现有宽度先后进入 running；请求必须发生在同一份状态里两者都已经 ready 之后，并且两条请求的时间重叠。

不把 P2-5、P2-6、P2-7、P2-11 拉进来。不改 TensorRT execution context 的归属，不改动态 batch，不要求九个 id 同时驻留。每个 worker 仍各自持有一个 execution context。`TrtSession::init` 里每个 session 一次 `createExecutionContext()` 保持原样。

`mortredctl doctor --strict` 的退出码不是这次通过条件。`--strict` 还会因 token 长度和 token 相同而失败，那是 P2-5。这次不调用 `doctor`、`prepare`、`calibrate`、`http_infer_rps.py`。占用门闩保持现有 enforce：不设 `occupancy_policy=off`，不设 `MORTRED_OCCUPANCY_ENFORCE=0`。

实跑期望退出码 0 的前提是：本机 WSL 已有 `_bin/mortred-supervisor.out` 或 `bin/mortred-supervisor.out`，两份产品 engine 已在现有 toml 指向的路径上，`demo_data/model_test_input/object_detection/bus.jpg` 与 `face_wo_mask.jpg` 可读，并且 `conf/local/trust.env` 或环境里已有现有的管理 token、推理 token、metrics token，gateway 能按现有规则启动。缺任何一项时实跑退出码非 0，记录不得写成通过。本方案不补 engine，不改 token 规则。

## 做法

### 推荐

新增 `scripts/resident_pair_record.py`。实跑时它在本进程里写一份临时 pack，只含两节：`[pack.yolov8s]` 与 `[pack.libface]`。每节 `worker_nums = 1`、`gpu_mem_mib = 1800`、`gpu_mem_at_workers = 1`、`gpu_mem_source = "seed"`。不写 `model_config`、`decode_auto`、`gpu_name`、`gpu_memory_total_mib`。数值与现有 `conf/packs/yolov8.toml` 的 seed 相同，只供现有占用门闩放行，不当作本机实测显存，也不写回 `conf/packs/`。

脚本只给它自己拉起的 supervisor 设置 `MORTRED_PACK` 指向这份临时 pack，并按 `scripts/mortredctl_next.sh` 的现有方式设置 `LD_LIBRARY_PATH`（`_lib` 与 `3rd_party/libs`）。二进制先找 `_bin/mortred-supervisor.out`，再找 `bin/mortred-supervisor.out`。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。三个 token 只从已有环境或 `conf/local/trust.env` 读入，不打印、不写入记录。

supervisor 用现有 autostart 拉起 gateway（`127.0.0.1:8080`）和这两个模型（`yolov8s` 端口 9056，`libface` 端口 9051）。脚本轮询 `GET http://127.0.0.1:8787/api/v1/status`。读到同一份响应里两者 `ready` 都为 true 之后，用两个线程在同一道栏杆后各发一条请求，栏杆之前不发送：

- `POST http://127.0.0.1:8080/v1/models/yolov8s/infer`，body 为现有 JSON 信封，`images` 只有一张，内容是 `bus.jpg` 的 base64
- `POST http://127.0.0.1:8080/v1/models/libface/infer`，同样的信封，图片是 `face_wo_mask.jpg`

不走模型端口，不用 `--raw`，不调用 `run_load`。每条请求记单调时钟的开始和结束。两条都返回后，再读一次 `/api/v1/status`。把临时 pack 全文、前一份状态、两条请求、后一份状态写入 `logs/resident_pair_yolov8s_libface.json`。判定通过才以退出码 0 结束。不重试请求去凑重叠。

记录落盘并且判定已经完成后，脚本向该 supervisor 主进程发 SIGINT，等待现有收尾。收尾失败只在标准错误写一行，不把已经满足的记录改成失败，也不改停止顺序。`8787`、`8080`、`9051`、`9056` 已被占用时直接非 0 退出，不杀已有进程。

`--check` 不拉起进程，不写上述 json。它只用固定样例跑同一判定函数。

代价：一次实跑有四个短命进程，supervisor、gateway、以及各一个 worker 的 `yolov8s` 与 `libface`。显存是两份现有 static batch=1 engine，各一个 execution context。不生成 engine，不加 context 池。墙钟停在现有 autostart（gateway 15 秒、模型波 60 秒、`start_concurrency = 1`）加上两条重叠请求，不测 30 秒 RPS。占用门闩若因空闲显存低于 1800 MiB 拒绝第二个模型，实跑非 0，不得改 stamp 或关掉门闩来通过。

### 否决

把请求塞进 `run_one_point()`、`measure_joint_once()` 或 `scripts/bench_set_a/http_raw.py`。`run_one_point()` 结束会停掉该进程；`measure_joint_once()` 的职责是读显存、不发推理；`http_raw.py` 的并发是 `w*` 的倍数、时长 15 秒。这三个入口都要留给单 id 的占用和 HTTP 报告。把第二个模型加进去，会改它们的进程生命周期，也把 TP2 的记录和 P2-11 的单模型数字写在同一次输出里。

在 `test/gateway_e2e_test.cc` 里用两个 fake 上游分别叫 `yolov8s` 和 `libface`。fake 的 `/ready` 和 HTTP 200 可以进默认 ctest，也不占显存。那份结果里的 ready 不是驻留集里这两份 TensorRT 进程的 ready，响应也不是这两份产品引擎的推理结果。锁定句要的记录仍然没有。

## 改动面

- `scripts/resident_pair_record.py`：实跑、临时 pack、记录、判定函数、`--check`。
- `test/CMakeLists.txt`：增加 ctest `resident_pair_record_rules`，命令是 `python3 scripts/resident_pair_record.py --check`。它进入默认 `check`（`ctest -LE valgrind`）。实跑不注册进默认 `check`。

## 明确不改

- `src/models/backend/trt_session.cpp` 的 `createExecutionContext()`，以及每个 worker 一个 session、一个 context。
- `max_batch_size`、静态 shape、两份产品 engine 路径、`conf/model/object_detection/yolov8/yolov8s.toml`、`conf/model/object_detection/libfacedetection/libface.toml`。
- `conf/server/object_detection/yolov8/yolov8s_server_config.toml` 与 `conf/server/object_detection/libface_det/libface_server_config.toml` 的 `model`、`port`、`worker_nums`、`server_uri`、`max_batch_size`。
- `conf/mortred.toml` 的 `autostart_default`、`start_concurrency`、`api_port`、gateway 端口和 `[gateway.rate_limit]`。
- `conf/packs/demo.toml`、`conf/packs/yolov8.toml` 以及另外两份已有 pack。`docker-compose.yml` 的 `MORTRED_PACK` 仍指向 demo pack。`deploy/` 不改。
- `scripts/calibrate_pack.py`、`scripts/bench_set_a/http_raw.py`、`scripts/server/http_infer_rps.py`。
- `scripts/mortredctl_init.sh`、`scripts/mortredctl_prepare.sh`、`scripts/mortredctl_doctor.sh`。
- gateway 路由、token 启动条件、systemd 单元、停止信号顺序、`realesrgan`。
- `test/gateway_e2e_test.cc`。不新增 fake 双上游用例。

## 契约

管理接口仍是 `GET /api/v1/status`，鉴权仍是现有管理 token。状态对象仍用现有字段 `id`、`state`、`pid`、`ready`。gateway 地址仍是 `127.0.0.1:8080`。

推理仍是 `POST /v1/models/{id}/infer`，`id` 精确为 `yolov8s` 或 `libface`，没有大小写折叠。请求仍是现有 JSON：`images` 为长度 1 的数组，不使用 `img_data`，不使用原始字节。鉴权仍是现有推理 token。

成功响应沿用 `docs/api-contract.md`：HTTP 200，根上 `status` 为 0，`results` 长度等于 `images` 的长度，本条为 1，`results[0].status` 为 0。HTTP 200 且根上 `status` 为 68 不算这次成功。响应里的框、关键点、类别名不进入通过条件。

临时 pack 只在这次进程的环境里生效，不改仓库里的 pack，也不改产品 toml 的 `type = "tensorrt"`。

## 性能与资源

不生成 engine，不增加 execution context，不加 context 池，不改 `worker_nums`，不改 batch。常驻到记录写完为止的进程是这一次 supervisor、它拉起的 gateway、以及 `yolov8s` 与 `libface` 各一个模型进程。每个模型进程的 worker 数是 server toml 里的 1，每个 worker 仍持有自己的 execution context。两份 engine 仍是现有 static batch=1 文件。

临时 pack 的 1800 MiB 只参与现有占用门闩。不写 `gpu_memory_total_mib` 时，现有的「stamp 之和相对总显存」检查不启用；`nvidia-smi` 有空闲显存读数时，现有的「空闲低于 stamp 则拒绝 spawn」仍然生效。门闩拒绝时实跑失败。

不测量 RPS、p50、p99、错误率，不引用 `docs/benchmark-h3/` 的 Throughput。那些字段属于 P2-11。

## 验收标准

同一次外部验收先跑「自测命令」第 1 条，再跑第 2 条，然后只打开 `logs/resident_pair_yolov8s_libface.json`。下面 7 条都要满足。只满足第 1 条不通过。

1. `python3 scripts/resident_pair_record.py --check` 退出码为 0。它的样例同时覆盖这些判定：两条请求时间重叠且两边都是成功时判过；两条时间不重叠时判不过；请求前一份状态里有一边 `ready` 不为 true 时判不过；后一份状态的 pid 与前一份不一致时判不过；任一条 HTTP 状态不是 200，或根上 `status` 不是 0，或 `results[0].status` 不是 0 时判不过；pack 出现第三个 id 时判不过。这条命令不创建 `logs/resident_pair_yolov8s_libface.json`。

2. `python3 scripts/resident_pair_record.py` 退出码为 0，并写出 `logs/resident_pair_yolov8s_libface.json`。该文件的文本里不出现管理 token、推理 token、metrics token。

3. 该文件里有一个 `before` 对象，来自同一次 `/api/v1/status`。其中 `yolov8s` 与 `libface` 的 `ready` 都为 true，`state` 都为 `running`，`pid` 都大于 0；`gateway` 的 `state` 为 `running`，`ready` 为 true，`pid` 大于 0。

4. 同一文件里的 pack 全文只有 `[pack.yolov8s]` 与 `[pack.libface]`。两节都是 `worker_nums = 1`，都有 `gpu_mem_mib = 1800`、`gpu_mem_at_workers = 1`、`gpu_mem_source = "seed"`。没有第三个 id，没有 `model_config`，没有 `decode_auto`。

5. 同一文件里有两条请求。路径分别是 `/v1/models/yolov8s/infer` 与 `/v1/models/libface/infer`。两条的开始时间都晚于 `before` 的采样时间。`yolov8s` 的开始早于 `libface` 的结束，并且 `libface` 的开始早于 `yolov8s` 的结束。

6. 这两条的 HTTP 状态都是 200，根上 `status` 都是 0，`results` 长度都是 1，`results[0].status` 都是 0。任一条根上 `status` 为 68 则不通过。

7. 同一文件里的 `after` 仍是同一次 `/api/v1/status`。`yolov8s` 与 `libface` 的 `pid` 分别等于 `before` 里的 pid，`ready` 仍都为 true。文件里不存在「其中一个 pid 已经消失，然后才开始另一个请求」的记录。

## 测试义务

QA 按上面的顺序做这一次验收，逐条看退出码和那一份 json。`--check` 通过而实跑 json 不存在，或实跑退出码非 0，结论是不通过。不得用阅读 `supervisor.cpp`、`gateway_app.cpp` 或 `trt_session.cpp` 代替这次实跑。不得把先停掉其中一个再请求另一个的输出当成这份记录。不得把 `ctest` 里原有的 `gateway_e2e_test` 或 `dead_upstream_maps_to_503` 当成 TP2。

白盒只有 `resident_pair_record_rules`：默认 `check` 会跑到它，它必须退出码 0，并且它不拉起模型、不写实跑 json。实跑不进入默认 `check`。无 GPU 的 CI 跑过 `resident_pair_record_rules` 不得写成 TP2 已满足。

缺 engine、缺 demo 图片、缺 token 或端口被占用时，实跑非 0。QA 把这次验收记为未通过，不改记成 P2-5 或 P2-11 的结果，也不把跳过写成通过。

不跑 `mortredctl prepare`、`mortredctl calibrate`、`mortredctl doctor`、`http_infer_rps.py`。不要求九个 id 同时出现在状态里。不检查框的坐标和人脸关键点。

## 自测命令

在本机 WSL Ubuntu 执行，仓库路径 `/mnt/g/Codex/mortred_model_server`。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 scripts/resident_pair_record.py --check
```

期望退出码：0。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 scripts/resident_pair_record.py
```

期望退出码：0。完成后同一目录下有 `logs/resident_pair_yolov8s_libface.json`，并且该文件满足验收标准 2 至 7。前提缺一项时退出码非 0，该次不得改记为 0。

## RD 不得自行决定

- 成功的定义：HTTP 200，根上 `status` 为 0，`results` 长度为 1，`results[0].status` 为 0。根上 `status` 为 68、只看 HTTP 2xx、或只看模型端口，都不能算通过。
- 重叠的定义：两条开始时间都晚于同一次 `before`，并且每条的开始早于另一条的结束。不加重试，不插入睡眠去制造重叠。
- 图片只使用 `demo_data/model_test_input/object_detection/bus.jpg` 与 `face_wo_mask.jpg`。缺文件就失败，不换别的图。
- 请求只走 gateway 的 `/v1/models/yolov8s/infer` 与 `/v1/models/libface/infer`。
- pack 只有这两个 id，`worker_nums` 保持 1，seed 保持 1800。不得下调 stamp，不得关闭占用门闩，不得把 `model_config` 指到 cpu 或 onnx toml。
- 不改 `start_concurrency`，不放宽 autostart 的 15 秒与 60 秒期限。60 秒内没有同时 ready 就是实跑失败。
- 不改 execution context 的归属，不加 context 池，不改 `max_batch_size`，不把 engine 改成动态 batch。
- 不把 `doctor --strict`、calibrate、`--raw`、30 秒 RPS、p50、p99、Throughput 比值写进通过条件。
- 不改 token 规则、systemd、停止顺序、`realesrgan`，不拉起九个 id。
- 清理用 SIGINT 打到这次的 supervisor 主进程。清理失败不反写已经满足的记录，也不借此改 supervisor。
- 实跑不进默认 `check`。`--check` 的退出码 0 不能单独当作 TP2 通过。
