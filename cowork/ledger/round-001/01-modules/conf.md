# 模块评分

- id: conf
- 权重: 5
- 分数: 6.0
- 读过的路径: `cowork/ledger/rubric.md`；`conf/onnx_sources.json`（文首 policy，以及 MOBILENETV2、YOLOV8、LIBFACE、DBNET、PPHUMAN_SEG、REAL_ESRGAN、SUPERPOINT、DINOV2/vits14、METRIC3D/512x1088）；覆盖集九个产品 toml 与对应 `conf/server` toml，外加 `conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml`、`conf/server/classification/mobilenetv2/mobilenetv2_server_cpu_config.toml`；`conf/packs/` 下 `demo.toml`、`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml`；`conf/mortred.toml`；`conf/product_static_bs1.json` 中这九个 id；`conf/trt_profiles/set_a_libface.json`。对照过 `docs/benchmark-h3/libface.txt`、`docs/benchmark-h3/yolov8s.txt`、`docs/benchmark-set-a.md`，以及 `src/models/object_detection/detection_params.h`、`src/models/object_detection/libface_detector.inl`。

## 已具备

覆盖集九个产品 toml 都是 `type = "tensorrt"`，权重路径都是 `*.engine.static_bs1.*`，与 `conf/product_static_bs1.json` 的 `engine`、`model_config`、`server_config`、端口一致。对应九份 `conf/server` 都是 `worker_nums=1`、`max_batch_size=1`。这满足「结论不写进 `conf/server`」和 static batch=1。cpu 侧另有 `mobilenetv2_cpu.toml`（`type = "onnx"`）和 `mobilenetv2_server_cpu_config.toml`（`profile="cpu"`），产品文件没有被改成用 ONNX 直接服务。EX3 里「产品 toml 被改成 `type=onnx`」这一条，就这九个文件而言已经满足。

`conf/packs/yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml` 写明种子占用 `gpu_mem_mib = 1800`、`gpu_mem_source = "seed"`，并写了复制后再 `mortredctl calibrate --pack <copy> --write-pack`。`conf/mortred.toml` 把 gateway 放在 `127.0.0.1:8080`、supervisor 放在 `127.0.0.1:8787`。YOLOV8 与 METRIC3D/512x1088 在 `conf/onnx_sources.json` 里的 `product_type` 和 `runtime_path` 与产品 engine 一致。

## 问题

1. 级别 P1。完成标准 PF1、TP1、PF5。出处：`conf/packs/demo.toml`、`conf/packs/yolov8.toml`、`conf/packs/metric3d.toml`、`conf/packs/gpu_yolo_metric3d.toml`；对照 `docs/benchmark-set-a.md` 表 3、表 4 与文首「产品 toml」行，以及 `docs/benchmark-h3/yolov8s.txt`。现在的行为：四份 pack 都没有 `decode_auto`，`worker_nums` 都是 1。`[pack.*]` 只覆盖 `mobilenetv2`、`yolov8s`、`metric3d_512`。`demo.toml` 的 `mobilenetv2` 指向 `mobilenetv2_cpu.toml`（`type = "onnx"`），不是 TensorRT 产品 toml。`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan` 没有 pack 节。`docs/benchmark-set-a.md` 写明标定时长 15 秒、表 3 不是发布 HTTP 值，表 4 里 `yolov8s` 的 w* 为 8、并发为 32，且标定使用的 `conf/bench/set_a/` 在仓库里不存在。`docs/benchmark-h3/yolov8s.txt` 只有 trtexec Throughput 400.677 qps，没有 HTTP 成功 RPS 与它的比值。改完后一次外部验收即可确认的变化：每个覆盖集 id 在某份 pack 的 `[pack.<id>]` 里有 `worker_nums`，以及 `decode_auto` 为 `cpu` 或 `gpu`；对应 `conf/server` 仍是 `worker_nums=1`；仓库里有一份报告同时写出硬件、日期、这两项、带 `--raw` 的压测命令、成功 RPS、p50、p99、错误率，以及 `docs/benchmark-h3/<id>.txt` 的 Throughput 和二者比值；并发数等于该 pack 的 `worker_nums`，时长不少于 30 秒。

2. 级别 P1。出处：`conf/onnx_sources.json` 与九个产品 toml。现在的行为：交换目录和进程实际加载的产品 toml 不是同一套后端。`MOBILENETV2`、`DBNET`、`PPHUMAN_SEG`、`SUPERPOINT`、`DINOV2`/`vits14` 的 `product_type` 是 `mnn`，`runtime_path` 是 `.mnn`；`PPHUMAN_SEG` 的备注写着 “Product toml stays type=mnn”。`REAL_ESRGAN` 的 `product_type` 是 `mnn`，`runtime_path` 是 `.model`。这六个产品 toml 的 `type` 都是 `tensorrt`，`model_file_path` 是 static batch=1 engine。`LIBFACE` 的 `product_type` 是 `onnx`，`config` 指向 `conf/model/object_detection/libfacedetection/640x480_config.toml`，该路径不存在；产品文件是 `libface.toml`，`type = "tensorrt"`，engine 为 `face_detection_yunet_2026may.engine.static_bs1.fp16`。改完后一次外部验收即可确认的变化：这九个 id 在 `conf/onnx_sources.json` 里的 `product_type`、`runtime_path`、`config` 与对应产品 toml 的 `type`、`model_file_path` 和实际路径一致，且 `config` 指向的文件能打开。

3. 级别 P1。出处：`conf/model/object_detection/libfacedetection/libface.toml`、`conf/trt_profiles/set_a_libface.json`、`docs/benchmark-h3/libface.txt`。解析约定见 `src/models/object_detection/detection_params.h`：`model_input_image_size` 是 `[height, width]`。现在的行为：toml 为 `[640, 480]`，按该约定是高 640、宽 480，profile 的 opt 是 NCHW `[1, 3, 640, 480]`。同一条 engine `face_detection_yunet_2026may.engine.static_bs1.fp16` 在 `docs/benchmark-h3/libface.txt` 里的命令是 `--shapes=input:1x3x480x640`。三处的高和宽不一致。改完后一次外部验收即可确认的变化：这三处的高和宽相同。

## 范围外

覆盖集以外的 catalog id，包括 `yolov8n`、`yolov8l`、`yolov8x`、`metric3d_1088`，以及 `conf/onnx_sources.json` 里的 diffusion、SAM、CLIP、LightGlue、RTDETR、`msocrnet`。`*.dyn.onnx` 与 TensorRT 动态 shape、动态 batch。`conf/mortred.toml` 不存放推理、管理、metrics token，这三枚 token 由环境变量进入进程。Docker 镜像与默认运行用户。

## 依据

PF1、TP1、PF5 未满足，因为仓库里没有按「本次性能约束」写完的报告，而且 `conf/packs` 没有可供该报告引用的 `decode_auto`。这次评分没有跑压测，这几行按缺少报告记为未满足，不记阻塞。`docs/benchmark-h3/<id>.txt` 有 trtexec Throughput，该数字不是 HTTP 吞吐。

EX3 的产品 toml 条款已满足，见「已具备」。cpu profile 上 `type=tensorrt` 是否会进入 ready，这次没有运行，记为未验证判断。

AV1：`mobilenetv2_cpu.toml` 是 `type=onnx`。缺少 HTTP RPS 不使该行未满足。这次没有单独拉起并做一次推理，能否成功记为未验证判断。

PF2、PF3、PF4、ST1–ST6、TP2、TP3、OP1–OP5、UX1–UX4、EX1、EX2 的可观察行为在服务进程、脚本或文档里，不在本次作为 conf 问题的文件行为里。`conf/mortred.toml` 的 `[gateway.rate_limit] enabled = false` 是否等于队列打满后没有 429，未读到对应拒绝路径，记为未验证判断。`metric3d_512.toml` 的 `canonical_size = [512, 960]` 与 engine 名 `512x1088` 是否同一套预处理，未读到使用处，记为未验证判断。`dbnet.toml` 里的 `model_class_nums=80` 是否被 OCR 检测读到，同样记为未验证判断。

`scripts/check_legacy_yolov8s_engine_align.py` 只把 YOLOV8 以外的 HRNET、DEPTH_ANYTHING、METRIC3D 纳入 `runtime_path` 对照，不检查上面七个不一致的覆盖集条目。

## 未覆盖

无
