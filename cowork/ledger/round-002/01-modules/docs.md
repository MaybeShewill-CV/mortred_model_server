# 模块评分

- id: docs
- 权重: 5
- 分数: 7.4
- 读过的路径: `cowork/ledger/rubric.md`；`docs/api-contract.md`；`docs/unsupported-boundaries.md`；`docs/deployment.md`；`docs/deployment.zh-cn.md`（第一小时与 `full` / `full-cpu` 段）；`README.md`；`README.zh-cn.md`；`docs/how_to_add_new_server.md`；`docs/how_to_add_new_server.zh-cn.md`；`docs/model-developer-guide.md`；`docs/model-developer-guide.zh-cn.md`；`docs/openapi.json`（覆盖集九个 id 的请求/响应 example，以及 `ValidationError`）；`docs/tutorials_of_model_servers.md`；`docs/tutorials_of_model_servers.zh-cn.md`；`docs/perf/coverage_set_http_rps.md`；`docs/benchmark-h3/` 下覆盖集九个 id 的 Throughput 行；`CMakePresets.json`；`scripts/mortredctl_next.sh`；`src/factory/classification_task.h`；`src/factory/obj_detection_task.h`；`src/factory/scene_segmentation_task.h`；`src/factory/enhancement_task.h`；`src/factory/feature_point_task.h`；`src/apps/model_server_main.cpp`；`src/apps/product_index.cpp`；`conf/server` 里上述教程 id 的 server toml；`test/server_e2e_contract_test.cc` 的 `img_data` 422 用例

## 已具备

UX1 已满足。中英文 README 与 `docs/deployment.md`、`docs/deployment.zh-cn.md` 的 GPU/CPU 源码构建都是 `cmake --preset full && cmake --build --preset full` 和 `cmake --preset full-cpu && cmake --build --preset full-cpu`。`CMakePresets.json` 里 `full` 输出到 `build/full`，`full-cpu` 输出到 `build/full-cpu`，并设置 `MORTRED_BUILD_PROFILE=cpu`。

UX2 已满足。`README.md` 与两份 deployment 的第一小时是反复执行 `mortredctl next`，直到 `doctor --strict`。`scripts/mortredctl_next.sh` 的最后一条是 `mortredctl doctor --strict`。文中的 `mortredctl` 子命令、`install_deps.sh`、`bootstrap.sh`、`fetch_weights.py`、`verify_deployment.sh`、`prepare_pack.sh`、`convert_trt_engines.sh`、`calibrate_pack.py`、`http_infer_rps.py` 都在仓库里。

UX4 已满足。`docs/openapi.json` 为 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512` 各有一条 `images` 请求 example 和一条 200 响应 example。`docs/api-contract.md` 写明已删除的 `img_data` 返回 422，`errors[].pointer` 为 `/img_data`，message 指向改用 `images`。`test/server_e2e_contract_test.cc` 的 `legacy_img_data_returns_422_with_migration_hint` 断言同一行为。

EX1 点名的两份英文文件已满足。`docs/how_to_add_new_server.md` 写明不要为每个模型新增 `src/apps/server/*.cpp` 或 `add_mortred_app`；示例 id、`--model`、`model=` 和 family 都是 `densenet`，与 `src/factory/classification_task.h` 的 `model_section` 一致。`docs/model-developer-guide.md` 用家族 `catalog()` 和 `mortred-model-server.out --list`，探针不得出现在 `--list` 里。

`docs/perf/coverage_set_http_rps.md` 九行都写了硬件、日期、`worker_nums`、`decode_auto`、`--raw` 压测命令、成功 RPS、p50、p99、错误率，以及与 `docs/benchmark-h3/<id>.txt` 一致的 Throughput 和比值。每行 `-c` 等于该行 `worker_nums` 的两倍，时长 60 秒。`conf/server` 的 `worker_nums` 仍是 1。排队与推理是分开的直方图计数；gateway 五段和 `server_time_ms` p50 都在，主导段为 `upstream`。本轮没有重跑这些命令。

`docs/unsupported-boundaries.md` 写明 Linux x64、CUDA 12.x、TensorRT 10.x、engine 须本机重建、`dbnet` 只做检测、`type=tensorrt` 在 cpu profile 上初始化失败。

## 问题

README 的「文档教程」链到 `docs/tutorials_of_model_servers.md` 与 `docs/tutorials_of_model_servers.zh-cn.md`。两份教程的启动命令是 `--model MOBILENETV2`、`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN`、`SUPERPOINT`。`src/apps/model_server_main.cpp` 写明 `--model` 大小写敏感，并且必须等于 catalog 的 `model_section`。当前 id 与 server toml 段名是 `mobilenetv2`、`yolov5l`、`bisenetv2`、`attentive_gan`、`superpoint`。`mobilenetv2` 和 `superpoint` 在覆盖集。按教程执行会落到 `unknown --model`。教程还写仓库自带配置的 `worker_nums` 可以在显存够时加大；`docs/deployment.md` 要求 `conf/server` 的 `worker_nums` 保持 1，校准结果写在机器 pack。

`README.zh-cn.md` 的添加入口是 `docs/how_to_add_new_server.zh-cn.md`。该页示例是 `Entry{"DENSENET", ...}`、`--model DENSENET`、`model = "DENSENET"`，并且没有 family `"densenet"`。`classification_task.h` 与 `conf/server/classification/densenet/densenet_server_config.toml` 的 id 是 `densenet`，family 也是 `densenet`。

`docs/api-contract.md` 与 `docs/api-contract.zh-cn.md` 的公共响应示例里 `model.name` 是 `MOBILENETV2`。`docs/openapi.json` 中 `mobilenetv2` 的响应 example 是 `mobilenetv2`。

改完后打开这五份文档即可确认：教程五条 `--model` 与对应 toml 段名、catalog id 一致；中文加服务示例的 id、`--model`、`model=` 和 family 都是 `densenet`；两份人读契约的 `model.name` 示例是 `mobilenetv2`；教程不再要求改 `conf/server` 的 `worker_nums`。

## 范围外

已关闭 P1-2 的重复描述：英文 `docs/how_to_add_new_server.md` 与 `docs/model-developer-guide.md` 已使用 `densenet`，且不要求单独的服务器二进制。该陈述不回到问题。

已关闭 P2-10 的重复描述：`README.zh-cn.md` 第一小时已列出 `mortredctl next`。不回到问题。

已关闭 P2-11 的重复描述：覆盖集九个 id 的 calibrate 与 `--raw` HTTP 清单已在 `docs/perf/coverage_set_http_rps.md`，并发是已部署 `worker_nums` 的两倍。不回到问题。

P2-8 保持在范围外，不入列。

英文加服务示例的 display name 是 `densenet classification`，`classification_task.h` 是 `Densenet classification`。`--list` 的 id 列仍是 `densenet`。不单列。

`README.md` 的 Benchmark 环境段仍写 CUDA 11.5 / RTX 3080。同一份 README 的快速开始和 `docs/unsupported-boundaries.md` 写的是 CUDA 12.x / TensorRT 10.x。安装命令不走那段。不单列。

`README.md` 的 yolov8s 示例是 `http_infer_rps.py --model yolov8s -c 8 -d 30s --raw`。清单里该 id 的 `worker_nums` 是 4，`-c 8` 即两倍。不单列。

UX3 落在 ci。PF/ST/TP/OP/EX2/EX3 的实现落在其它模块。Docker、多机、Triton、动态 batch、覆盖集以外的尺寸、`msocrnet`、SAM、diffusion、CLIP、LightGlue、RTDETR、MOT、HTTP OCR 识别，按 rubric 范围外。

## 依据

分数表示这套文档能不能让一个小组把这一台 Linux GPU 服务器用起来。第一小时、边界、OpenAPI 示例和九行 HTTP 报告已经能支撑上线、看时间构成和引用现有吞吐。README 直接链出的启动教程，以及中文添加入口，仍给出会被大小写敏感的 `--model` 拒绝的 id，所以到不了「按文档加上或拉起一个模型」。7.4 是这一档。

必读三份和点名的指南、OpenAPI 相关段都读过。`docs/ui-redesign/`、monitoring、about_model 配置长文、api-keys、async-jobs、onnx-interchange、benchmark-set-a、ci-golden、repository-layout、`contract_dump.json`，以及 `docs/api-contract.zh-cn.md`、`docs/deployment.zh-cn.md`、`docs/unsupported-boundaries.zh-cn.md` 的未核对段落，没有逐份读完。因此分数不超过 8.0。

## 未覆盖

`docs/ui-redesign/` 全部、`docs/ui-redesign-gallery.md`、`docs/monitoring-guide.md`、`docs/monitoring-guide.zh-cn.md`、`docs/about_model_configuration.md`、`docs/about_model_configuration.zh-cn.md`、`docs/about_model_server_configuration.md`、`docs/about_model_server_configuration.zh-cn.md`、`docs/api-keys.md`、`docs/api-keys.zh-cn.md`、`docs/async-jobs-customer-test.md`、`docs/async-jobs-customer-test.zh-cn.md`、`docs/onnx-interchange.md`、`docs/onnx-interchange.zh-cn.md`、`docs/benchmark-set-a.md`、`docs/benchmark-set-a.zh-cn.md`、`docs/ci-golden-regression.md`、`docs/repository-layout.md`、`docs/contract_dump.json`、`docs/perf/latency_iterations.md`、`docs/perf/trt-session-submit-tax.md`、`docs/perf/yolov8s_decode_backend_rps.md`、`docs/unsupported-boundaries.zh-cn.md`。`docs/api-contract.zh-cn.md` 与 `docs/deployment.zh-cn.md` 只核对了完成标准相关段。`docs/benchmark-h3/` 只核对了覆盖集九个 id 的 Throughput 行。
