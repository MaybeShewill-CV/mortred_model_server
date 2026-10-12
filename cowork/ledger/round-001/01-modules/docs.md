# 模块评分

- id: docs
- 权重: 5
- 分数: 6.5
- 读过的路径: `cowork/ledger/rubric.md`、`docs/api-contract.md`、`docs/unsupported-boundaries.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`docs/openapi.json`、`docs/model-developer-guide.md`、`docs/how_to_add_new_server.md`、`docs/benchmark-set-a.md`、`docs/benchmark-h3/yolov8s.txt`、`docs/benchmark-h3/libface.txt`、`docs/benchmark-h3/mobilenetv2.txt`、`docs/perf/yolov8s_decode_backend_rps.md`、`README.md`、`README.zh-cn.md`、`CMakePresets.json`、`src/factory/classification_task.h`、`src/factory/cv_catalog.h`、`src/models/catalog/model_entry.h`、`scripts/mortredctl_next.sh`、`scripts/server/http_infer_rps.py`、覆盖集九个产品 toml 的 `type`

## 已具备

- `docs/api-contract.md` 写明 gateway `:8080`、请求只允许 `req_id` / `images` / `params` / `options`，`img_data` 与未知字段为 422，并给出指向 `/img_data`、改用 `images` 的错误体。`docs/openapi.json` 为覆盖集九个 id（`mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`）各有一份请求示例和一份带 `results[].data` 的响应示例。
- `docs/unsupported-boundaries.md` 把支持面收在 Linux x64、CUDA 12.x、TensorRT 10.x；`dbnet` 只做检测；engine 须本机 `mortredctl prepare`；产品 toml 不改成 `type=onnx`（已有 ONNX id 与 LIBFACE YuNet 除外）；cpu profile 上 `type=tensorrt` 应初始化失败。
- `README.md` 与 `docs/deployment.md`、`docs/deployment.zh-cn.md` 的首小时路径都是反复执行 `mortredctl next`，直到 `doctor --strict`。`scripts/mortredctl_next.sh` 的最后一条是 `mortredctl doctor --strict`。`scripts/bootstrap.sh`、`scripts/install_deps.sh`、`scripts/verify_deployment.sh`、`scripts/prepare_pack.sh`、`scripts/fetch_weights.py` 都在仓库里。
- 中英文 README 与 `docs/deployment.md` §7 的 GPU / CPU 源码构建命令都是 `cmake --preset full && cmake --build --preset full` 和 `cmake --preset full-cpu && cmake --build --preset full-cpu`，与 `CMakePresets.json` 的 `full`（`build/full`）和 `full-cpu`（`build/full-cpu`，`MORTRED_BUILD_PROFILE=cpu`）一致。UX1、UX2 这两行按现文件满足。
- `docs/how_to_add_new_server.md` 写明不要为每个模型新增 `src/apps/server/*.cpp`，统一二进制是 `mortred-model-server.out`。`docs/model-developer-guide.md` 的接入步骤是家族 `catalog()` 加一行，并用 `mortred-model-server.out --list` 检查。
- 覆盖集九个 id 的 `docs/benchmark-h3/<id>.txt` 都有硬件、日期、trtexec 命令和 Throughput。例如 `docs/benchmark-h3/yolov8s.txt` 是 RTX 2070 SUPER、2026-10-10、Throughput 400.677 qps。这是引擎参照，不是 HTTP 吞吐。

## 问题

1. **P1，完成标准 PF1、TP1、PF5。** 覆盖集九个 `type=tensorrt` id 没有一份按本次性能约束完成的 HTTP 报告。出处：`docs/benchmark-set-a.md` 表 3、表 4，`docs/perf/yolov8s_decode_backend_rps.md`，`README.md` 模型动物园下的示例命令，`docs/benchmark-h3/yolov8s.txt`。现在的行为：表 3 写明 `--duration 15s` 且「非发布 RPS」，没有 `--raw`；表 4 标成发布 HTTP 值，但 `yolov8s` 的 `w*` 是 8、并发是 32，`mobilenetv2` 的 `w*` 是 4、并发是 16，表里没有 p50、错误率、`decode_auto`，也没有相对 `docs/benchmark-h3/<id>.txt` Throughput 的比值；`libface` 这一行写成 `libface_640`。`docs/perf/yolov8s_decode_backend_rps.md` 虽用 `--raw`，时长是 15 秒，8 个 worker 配 `-c 16`，直接打模型端口，没有 p50、p99、错误率和 400.677 qps 的比值。`README.md` 有 `http_infer_rps.py --model yolov8s -c 8 -d 30s --raw`，这是一条示例命令，不是带硬件、日期、pack `worker_nums`、`decode_auto`、成功 RPS、p50、p99、错误率和比值的报告；另外八个 id 连这条示例都没有。`docs/deployment.md` 没有 `http_infer_rps`。尚无合格记录值，PF5 随 PF1 未满足。改完后一次外部验收即可确认的变化：覆盖集每个 `type=tensorrt` id 各有一份报告，能同时看到硬件、日期、pack 里的 `worker_nums` 与 `decode_auto`、`http_infer_rps.py --raw`、并发数等于该 `worker_nums`、时长不少于 30 秒、成功 RPS、p50、p99、错误率，以及对应 `docs/benchmark-h3/<id>.txt` 的 Throughput 和二者比值。抽查 `yolov8s` 时，并发数等于报告里的 `worker_nums`，比值用的是 `yolov8s.txt` 的 400.677 qps。

2. **P1，完成标准 EX1。** 加模型步骤和中文 README 的启动命令与当前 `catalog()` 不一致。出处：`docs/how_to_add_new_server.md` 的 catalog 示例和 `--model DENSENET`、`model = "DENSENET"`；`README.zh-cn.md` 第 219、234 行；`src/factory/classification_task.h` 的 `catalog()`；`src/factory/cv_catalog.h` 的 `family` 字段；`conf/server/classification/densenet/densenet_server_config.toml` 的 `model="densenet"`；`README.md` 与 `README.zh-cn.md` 写明 catalog id 为小写。现在的行为：指南示例把 `model_section` 写成 `DENSENET`，初始化列表在 `classification_param_specs()` 处结束，没有现有行末尾的 family 字符串 `"densenet"`；`cv_catalog.h` 写明 family 为空则该行不进 static batch=1 拆分。中文 README 用 `--model MOBILENETV2` 启动 benchmark 和 `mortred-model-server.out`，而 `catalog()` 注册的是 `mobilenetv2`。同一份中文 README 又写旧名已删除。中英文 README 的 cpu 精选模型写成 `resnet50`，`classification_task.h` 的 id 是 `resnet`，`docs/deployment.md` 写的是 `resnet`。指南没有要求为每个模型单独做一个服务器二进制。改完后一次外部验收即可确认的变化：`how_to_add_new_server.md` 示例的 id、family 和 `model =` 与 `classification_task.h` 里 `densenet` 那一行一致；`README.zh-cn.md` 的 `--model` 为 `mobilenetv2`；两份 README 的 cpu 精选 id 为 `resnet`。打开这三处，不再出现 `DENSENET`、`MOBILENETV2`、`resnet50` 作为要执行的 catalog id。

3. **P2。** 中文 README 的第一小时命令块没有 `mortredctl next`。出处：`README.zh-cn.md`「第一小时：mortredctl」命令块；对照 `README.md` 第 112 行、`docs/deployment.md` 第 10–13 行、`docs/deployment.zh-cn.md` 第 8–10 行。现在的行为：中文命令块从 `mortredctl init` 列到 `doctor`，全文检索没有 `mortredctl next`。英文 README 和两份部署文档都要求反复执行 `mortredctl next` 直到 `doctor --strict`，`scripts/mortredctl_next.sh` 也以这条结束。UX2 仍由英文 README 和 `docs/deployment.md` 满足，不把整行记成未满足。改完后一次外部验收即可确认的变化：`README.zh-cn.md` 第一小时命令块出现 `mortredctl next`，并且同一节能走到 `mortredctl doctor`。

## 范围外

- `README.md` Benchmark 节仍写 RTX 3080、CUDA 11.5、Driver 495。当前支持线在 `docs/unsupported-boundaries.md` 和快速开始表（CUDA 12.x、TensorRT 10.x）。这是旧的个人基准环境描述，不进入问题。
- `docs/benchmark-set-a.md` 里覆盖集以外的 id，不因九个代表 id 已有 trtexec 文件而进入完成标准。
- `msocrnet`、Docker 镜像用户、TensorRT 动态 batch、MNN 与 ONNX Runtime 的吞吐优化、多机调度，按 rubric 范围外，不写成问题。

## 依据

- 分数 6.5：部署、契约、边界和 OpenAPI 示例已经能让一个小组按文档把服务拉到 `doctor`，并看到九个覆盖 id 的请求与响应形状；trtexec 参照也在 `docs/benchmark-h3/`。按文档给出本次约束下的 HTTP 吞吐，以及按指南原样加一个已有家族的模型，这两件事对不上现文件，所以不到 8。
- UX1 满足：中英文 README 与 `CMakePresets.json` 的 `full` / `full-cpu` 命令一致。中文 README 另有一条 `cmake -B build -DMORTRED_BUILD_FULL=OFF`，那是 tests-only，不是这条完成标准里的 CPU/GPU 全量构建。
- UX4 的示例半边满足：九个 id 的请求和响应示例在 `docs/openapi.json`。`img_data` 返回 422 且错误指出改用 `images`，文档写在 `docs/api-contract.md`；这次没有跑服务，运行时是否如此是未验证判断，不写成已经发生的故障。
- PF2、PF4、TP2、UX3 不落在本模块的已发生故障。没有合格压测报告，因此不能从文档确认同一次压测是否读到分开的排队等待、推理耗时和 gateway 阶段耗时，也不能确认 `yolov8s` 与 `libface` 同时 ready 的并发结果。这些是未验证判断。UX3 的 OpenAPI 一致性检查在 CI，不在本模块文件里。
- 九个覆盖 id 的产品 toml 均为 `type=tensorrt`（含 `conf/model/object_detection/libfacedetection/libface.toml`），所以 PF1 覆盖这九个 id。

## 未覆盖

无
