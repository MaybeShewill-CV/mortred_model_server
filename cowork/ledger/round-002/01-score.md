# 项目评分

- 轮次: 2
- 自评原始分: 8.9
- 自评加权分: 8.9
- 上限规则: 无
- 相对上轮加权分: 7.1 → 8.9

## 模块
| id | 权重 | 分数 | 记录 |
| --- | --- | --- | --- |
| src-models | 15 | 9.0 | `cowork/ledger/round-002/01-modules/src-models.md` |
| src-server-apps | 20 | 9.4 | `cowork/ledger/round-002/01-modules/src-server-apps.md` |
| src-control | 15 | 9.0 | `cowork/ledger/round-002/01-modules/src-control.md` |
| src-factory-common | 10 | 9.2 | `cowork/ledger/round-002/01-modules/src-factory-common.md` |
| conf | 5 | 9.0 | `cowork/ledger/round-002/01-modules/conf.md` |
| ops | 15 | 8.8 | `cowork/ledger/round-002/01-modules/ops.md` |
| test | 10 | 8.8 | `cowork/ledger/round-002/01-modules/test.md` |
| docs | 5 | 7.4 | `cowork/ledger/round-002/01-modules/docs.md` |
| ci | 5 | 8.0 | `cowork/ledger/round-002/01-modules/ci.md` |

## P0
无

## P1
无

## P2
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| P2-12 | 英文模型服务教程的五条 `--model` 与 catalog id、server toml 的 `model=` 不一致。按教程启动会得到 `unknown --model`。 | 从 `README.md` 的文档教程打开 `docs/tutorials_of_model_servers.md`，按其中一条命令启动。 | 该文件五条命令是 `--model MOBILENETV2`、`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN`、`SUPERPOINT`。`src/apps/model_server_main.cpp` 写明 `--model` 大小写敏感，且必须等于 catalog 的 `model_section`。对应 toml 的 `model=` 与 catalog id 是 `mobilenetv2`、`yolov5l`、`bisenetv2`、`attentive_gan`、`superpoint`。`YOLOV5` 对不上 `yolov5l`，`ATTENTIVE_GAN_DERAIN` 对不上 `attentive_gan`。 | 只打开 `docs/tutorials_of_model_servers.md`。这五条 `--model` 分别是 `mobilenetv2`、`yolov5l`、`bisenetv2`、`attentive_gan`、`superpoint`。 | `cowork/ledger/round-002/01-modules/docs.md`；`docs/tutorials_of_model_servers.md`；`src/apps/model_server_main.cpp`；`src/factory/classification_task.h`；`src/factory/obj_detection_task.h`；`src/factory/scene_segmentation_task.h`；`src/factory/enhancement_task.h`；`src/factory/feature_point_task.h`；对应五份 `conf/server` toml | 无 |
| P2-13 | 中文模型服务教程的五条 `--model` 与 catalog id、server toml 的 `model=` 不一致。按教程启动会得到 `unknown --model`。 | 从 `README.zh-cn.md` 的模型服务教程打开 `docs/tutorials_of_model_servers.zh-cn.md`，按其中一条命令启动。 | 该文件五条命令是 `--model MOBILENETV2`、`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN`、`SUPERPOINT`。`--model` 大小写敏感，必须等于 `model_section`。对应 id 是 `mobilenetv2`、`yolov5l`、`bisenetv2`、`attentive_gan`、`superpoint`。 | 只打开 `docs/tutorials_of_model_servers.zh-cn.md`。这五条 `--model` 分别是 `mobilenetv2`、`yolov5l`、`bisenetv2`、`attentive_gan`、`superpoint`。 | `cowork/ledger/round-002/01-modules/docs.md`；`docs/tutorials_of_model_servers.zh-cn.md`；`README.zh-cn.md`；`src/apps/model_server_main.cpp`；上列家族 `catalog()` 与对应 `conf/server` toml | 无 |
| P2-14 | 英文教程要求在显存够时加大仓库自带配置的 `worker_nums`。`docs/deployment.md` 要求 `conf/server` 的 `worker_nums` 保持 1，校准结果写在机器 pack。 | 按 `docs/tutorials_of_model_servers.md` 调整 worker 数，再对照 `docs/deployment.md`。 | 该教程写 “Shipped configs default to `worker_nums=1`. Raise it only if you have GPU headroom.” `docs/deployment.md` 写明 `conf/server` 的 `worker_nums` 保持 1，并发写在机器 pack，经 `MORTRED_PACK` 注入。 | 只打开 `docs/tutorials_of_model_servers.md`。关于 `worker_nums` 的那一句不再要求加大仓库自带配置；它写明 `conf/server` 保持 1，校准出的数值写在机器 pack。 | `cowork/ledger/round-002/01-modules/docs.md`；`docs/tutorials_of_model_servers.md`；`docs/deployment.md` | 无 |
| P2-15 | 中文教程要求在显存够时加大仓库自带配置的 `worker_nums`。部署文档要求 `conf/server` 的 `worker_nums` 保持 1，校准结果写在机器 pack。 | 按 `docs/tutorials_of_model_servers.zh-cn.md` 调整 worker 数。 | 该教程写 “仓库自带配置默认 `worker_nums=1`，仅在显存足够时再加大。” `docs/deployment.md` 要求 `conf/server` 保持 1，校准结果放在机器 pack。 | 只打开 `docs/tutorials_of_model_servers.zh-cn.md`。关于 `worker_nums` 的那一句不再要求加大仓库自带配置；它写明 `conf/server` 保持 1，校准出的数值写在机器 pack。 | `cowork/ledger/round-002/01-modules/docs.md`；`docs/tutorials_of_model_servers.zh-cn.md`；`docs/deployment.md` | 无 |
| P2-16 | 中文「添加新服务」示例的 id、`--model`、`model=` 与 catalog 不一致，且示例没有 family `"densenet"`。 | 从 `README.zh-cn.md` 的「快速添加新的 DL 服务」打开 `docs/how_to_add_new_server.zh-cn.md`，按示例往分类家族加 densenet。 | 示例是 `Entry{"DENSENET", "densenet classification", "DENSENET_CLASSIFICATION_SERVER", ...}`，参数在 `classification_param_specs()` 结束，没有 family。下文是 `--model DENSENET` 和 `model = "DENSENET"`。`classification_task.h` 与 `conf/server/classification/densenet/densenet_server_config.toml` 的 id 和 family 都是 `densenet`。`--model` 大小写敏感。英文 `docs/how_to_add_new_server.md` 已是 `densenet`，不在本条。 | 只打开 `docs/how_to_add_new_server.zh-cn.md`。示例的 `model_section`、`--model`、`model=` 和 family 都是 `densenet`。 | `cowork/ledger/round-002/01-modules/docs.md`；`docs/how_to_add_new_server.zh-cn.md`；`README.zh-cn.md`；`src/factory/classification_task.h`；`conf/server/classification/densenet/densenet_server_config.toml` | 无 |
| P2-17 | 英文接口契约公共响应示例的 `model.name` 是 `MOBILENETV2`。`docs/openapi.json` 里 `mobilenetv2` 的响应示例是 `mobilenetv2`。 | 打开 `docs/api-contract.md` 的公共响应示例，对照 `docs/openapi.json` 中 `mobilenetv2` 的 200 示例。 | `docs/api-contract.md` 的示例是 `"model": { "name": "MOBILENETV2", "version": "" }`。`docs/openapi.json` 同一 id 的响应示例是 `"name": "mobilenetv2"`。 | 只打开 `docs/api-contract.md`。公共响应示例的 `model.name` 是 `mobilenetv2`。 | `cowork/ledger/round-002/01-modules/docs.md`；`docs/api-contract.md`；`docs/openapi.json` | 无 |
| P2-18 | 中文接口契约公共响应示例的 `model.name` 是 `MOBILENETV2`。`docs/openapi.json` 里 `mobilenetv2` 的响应示例是 `mobilenetv2`。 | 打开 `docs/api-contract.zh-cn.md` 的通用响应封装，对照 `docs/openapi.json` 中 `mobilenetv2` 的 200 示例。 | `docs/api-contract.zh-cn.md` 的示例是 `"model": { "name": "MOBILENETV2", "version": "" }`。`docs/openapi.json` 同一 id 的响应示例是 `"name": "mobilenetv2"`。 | 只打开 `docs/api-contract.zh-cn.md`。通用响应封装的 `model.name` 是 `mobilenetv2`。 | `cowork/ledger/round-002/01-modules/docs.md`；`docs/api-contract.zh-cn.md`；`docs/openapi.json` | 无 |
| P2-19 | 同一次成功推理超过 4096 次之后，gateway `/metrics` 不再写明时间主要花在哪一段。 | 用会在响应里写很小 `server_time_ms` 的上游连续成功推理，使 gateway 五段 p50 之和大于该 `server_time_ms` p50 的一半，再读 gateway `/metrics`。 | `k_stage_sample_cap` 为 4096。每次成功都累加五段直方图，样本留在最多 4096 条的 deque 里。`mortred_gateway_stage_dominant` 只在 `stage_observed_ <= 4096` 且五段 p50 之和大于 `server_time_ms` p50 的一半时写出。第 4097 次成功之后直方图计数继续增加，这一行不再出现。 | 连续 4097 次成功推理后，带 metrics token 读一次 gateway `/metrics`。正文仍有五段 `mortred_gateway_stage_duration_ms`，并且有 `mortred_gateway_stage_dominant{model="gateway",stage="..."} 1`，`stage` 为这五段里 p50 最大的名字。 | `cowork/ledger/round-002/01-modules/src-control.md`；`src/control/gateway/gateway_app.h`；`src/control/gateway/gateway_app.cpp` | 无 |
| P2-20 | `gpu-pr-gate` 成功时，`inference-paths` 打印 `TRT_PROOF=yes` 并以 0 退出。这次作业实际跑的四个 golden 都是 `type=onnx`。 | 看一次 `gpu-pr-gate` 为 success 的 `inference-paths` 日志，并对照同一次 gtest 加载的配置。 | `MORTRED_GPU_SMOKE_FILTER` 是 `model_golden.yolov8_detection`、`model_golden.yolov8_mixed_size_batch_matches_single_runs`、`model_golden.mobilenetv2_classification`、`model_golden.libface_detection`。它们加载 `conf/ci/yolov8_onnx_hosted.toml`、`conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml`、`conf/ci/libface_onnx_hosted.toml`，`backend.type` 都是 `onnx`。`gpu == success` 时 `ci_eval_inference_paths.py` 打印 `TRT_PROOF=yes` 并返回 0。产品 `type=tensorrt` 用例不在该过滤器里。`convert_trt_engines.sh --only yolov8` 在引擎已存在且未加 `--force` 时打印 skip 并继续。工作流注释写 “Eight cases”，过滤器是这四个名字。GPU 作业被跳过时仍是 `NOT_A_TRT_PROOF`、退出码 2，不在本条。 | 只跑这四个 ONNX 用例，且 yolov8 引擎步骤被 skip 或只写出引擎文件时，`inference-paths` 日志里没有 `TRT_PROOF=yes`。日志出现 `TRT_PROOF=yes` 时，同一次 `gpu-pr-gate` 的 gtest XML 里至少有一个产品 `type=tensorrt` 用例，且 `skipped=0`。 | `cowork/ledger/round-002/01-modules/ci.md`；`.github/workflows/ci.yml`；`scripts/ci_eval_inference_paths.py`；`test/model_golden_test.cc`；`conf/ci/yolov8_onnx_hosted.toml`；`conf/ci/libface_onnx_hosted.toml`；`conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml`；`scripts/convert_trt_engines.sh` | 无 |

## 附录
| id | 陈述 | 相对上轮 |
| --- | --- | --- |
| P0-1 | 驻留集中的 `libface` 对一张合法 JPEG 完不成成功推理。产品 toml、TensorRT profile 与引擎剖面的高宽不一致。当前三处已是高 480、宽 640。 | 重复描述 |
| P1-1 | 不创建 `/tmp/mortred_stage_timing` 时，gateway `/metrics` 读不到阶段耗时（PF4）。当前成功转发路径会记五段，不看这个文件。 | 重复描述 |
| P1-2 | 按文档给已有家族加一个模型时，英文 `docs/how_to_add_new_server.md` 与 `docs/model-developer-guide.md` 的步骤和 `catalog()`、`--list` 不一致（EX1）。这两份英文文件已使用 `densenet`，且不要求单独的服务器二进制。中文示例见 P2-16，不在本条关闭范围内。 | 重复描述 |
| P2-1 | 覆盖集 `max_batch_size=1` 的路径不记录 batch 观测（PF3）。当前每张执行完的图记 1。 | 重复描述 |
| P2-2 | 打开限流后，鉴权失败仍消耗来源 IP 的配额（ST2）。当前 401 会退回这一次准入。 | 重复描述 |
| P2-3 | `fork` 之后、`exec` 之前的库函数跟踪里出现 `prctl`、`execvp` 或 `setenv`（ST5）。 | 重复描述 |
| P2-4 | 没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。本轮没有把实跑失败写成已经发生的故障。 | 重复描述 |
| P2-5 | 推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。 | 重复描述 |
| P2-6 | 停止信号没有收成只发给 supervisor 主进程（OP5）。 | 重复描述 |
| P2-7 | 解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。当前进会话前返回 `MODEL_INPUT_SIZE_REJECTED`。 | 重复描述 |
| P2-8 | `conf/onnx_sources.json` 与覆盖集产品 toml 的后端和配置路径不一致。交换目录不是调用路径。 | 范围外 |
| P2-9 | `golden_baseline.json` 的 `case_count` 为 30、`golden_count` 为 26，且缺少 `dinov2_vits14` 与 `metric3d_512`。当前名单是 49 与 62。 | 重复描述 |
| P2-10 | 中文 README 第一小时命令块没有 `mortredctl next`。当前第一小时已列出该命令。 | 重复描述 |
| P2-11 | 覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。当前清单在 `docs/perf/coverage_set_http_rps.md`，`-c` 等于已部署 `worker_nums` 的两倍。 | 重复描述 |

## 未覆盖模块
无
