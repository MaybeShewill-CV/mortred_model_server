# 模块评分

- id: conf
- 权重: 5
- 分数: 9.0
- 读过的路径: `cowork/ledger/rubric.md`；`conf/onnx_sources.json`（文首 policy，以及 MOBILENETV2、YOLOV8、LIBFACE、DBNET、PPHUMAN_SEG、REAL_ESRGAN、SUPERPOINT、DINOV2/vits14、METRIC3D/512x1088）；覆盖集九个产品 toml 与对应九份 `conf/server` toml；`conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml`；`conf/server/classification/mobilenetv2/mobilenetv2_server_cpu_config.toml`；`conf/packs/` 下 `demo.toml`、`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml`；`conf/product_static_bs1.json` 中这九个 id；`conf/trt_profiles/set_a_libface.json`；`conf/mortred.toml` 的 supervisor、gateway、rate_limit；`docs/benchmark-h3/libface.txt` 的 shapes 与 Throughput；`docs/perf/coverage_set_http_rps.md` 九行；`src/models/object_detection/detection_params.h` 的 `[height, width]` 解析。

## 已具备

覆盖集九个产品 toml 都是 `type = "tensorrt"`，`model_file_path` 都是 `*.engine.static_bs1.*`，与 `conf/product_static_bs1.json` 的 `engine`、`model_config`、`server_config`、端口一致。对应九份 `conf/server` 都是 `worker_nums=1`、`max_batch_size=1`，并指向上述产品 toml。这些文件里没有 `decode_auto`。cpu 侧另有 `mobilenetv2_cpu.toml`（`type = "onnx"`）和 `mobilenetv2_server_cpu_config.toml`（`profile="cpu"`）。九个产品文件没有被改成用 ONNX 直接服务。

`libface.toml` 的 `model_input_image_size = [480, 640]`。`detection_params.h` 把该数组定为 `[height, width]`，即高 480、宽 640。`conf/trt_profiles/set_a_libface.json` 的 min/opt/max 都是 NCHW `[1, 3, 480, 640]`。`docs/benchmark-h3/libface.txt` 的命令是 `--shapes=input:1x3x480x640`，Throughput 为 1668.97 qps。这三处的高和宽相同。

`conf/packs` 四份都是种子：`worker_nums = 1`，没有 `decode_auto`。`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml` 写明 `gpu_mem_source = "seed"`，并要求复制后再 `mortredctl calibrate --pack <copy> --write-pack`。`demo.toml` 的 `mobilenetv2` 指向 `mobilenetv2_cpu.toml`。`conf/mortred.toml` 把 gateway 放在 `127.0.0.1:8080`、supervisor 放在 `127.0.0.1:8787`，`autostart_default = false`。

`docs/perf/coverage_set_http_rps.md` 九行都写 `conf_server_worker_nums=1`，与当前九份 server 文件一致。标定结论在 `/tmp` pack：`worker_nums` 与 `decode_auto` 分别是 mobilenetv2 4/cpu、yolov8s 4/cpu、libface 4/gpu、pphuman_mobile 2/gpu、dbnet 2/gpu、dinov2_vits14 2/cpu、superpoint 1/gpu、realesrgan 4/cpu、metric3d_512 1/gpu。每行 `http_infer_rps.py --raw`，`-c` 是该 `worker_nums` 的两倍（8、8、8、4、4、4、2、8、2），`wall_s` 不少于 60，`error_rate=0.0`。

## 问题

无

## 范围外

P0-1 重复描述：产品 toml、TensorRT profile 与引擎剖面三处高宽不一致。当前三处已是高 480、宽 640，不回问题。

P2-8 重复描述：`conf/onnx_sources.json` 与产品 toml 不一致。当前交换目录里 MOBILENETV2、DBNET、PPHUMAN_SEG、SUPERPOINT、DINOV2/vits14 的 `product_type` 仍是 `mnn`，REAL_ESRGAN 的 `product_type` 仍是 `mnn` 且 `runtime_path` 是 `.model`，LIBFACE 的 `product_type` 仍是 `onnx`，`config` 仍指向不存在的 `conf/model/object_detection/libfacedetection/640x480_config.toml`。交换目录不是调用路径，不重新入列。

P2-11 重复描述：覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单。当前清单在 `docs/perf/coverage_set_http_rps.md`，`-c` 等于已部署 `worker_nums` 的两倍，不回问题。

覆盖集以外的 catalog id，包括 `yolov8n`、`yolov8l`、`yolov8x`、`metric3d_1088`，以及 diffusion、SAM、CLIP、LightGlue、RTDETR、`msocrnet`。`*.dyn.onnx` 与 TensorRT 动态 shape、动态 batch。Docker 镜像与默认运行用户。`conf/mortred.toml` 不存放推理、管理、metrics token。

## 依据

EX3：九个产品 toml 的 `type` 都是 `tensorrt`，不是 `type=onnx` 直接服务。LIBFACE（YuNet）当前也是 `type=tensorrt`，不落入「已经是 ONNX」的豁免缺口。cpu profile 的配置指向 `mobilenetv2_cpu.toml` 的 `type=onnx`，没有把 `type=tensorrt` 配进 cpu server。cpu profile 上 `type=tensorrt` 是否会初始化成功或进入 ready，这次没有运行，记为未验证判断，不写成故障。

`conf/server` 的 `worker_nums=1`：九份覆盖集 server toml 都是 1，报告里的 `conf_server_worker_nums=1` 与文件一致。标定出的 `worker_nums` 和 `decode_auto` 写在 `/tmp` pack，没有写进 `conf/server`。仓库里的 `conf/packs` 保持种子，符合「结论不写进 `conf/server`」。

AV1：九个产品 toml 都不是 `type=mnn` 或 `type=onnx`，这一行的触发条件在产品文件上不成立。`mobilenetv2_cpu.toml` 是 `type=onnx`。缺少 HTTP RPS 不使该行未满足。这次没有单独拉起并做一次推理，能否成功记为未验证判断，不写入问题。

PF1、TP1 里属于本模块的部分已由上述 server 文件和这份报告对上。报告中的排队计数、推理计数、gateway 阶段和 `server_time_ms` 在文档行里，不另开 conf 问题。`canonical_size` 只出现在 metric3d 的 toml，`src` 里没有这个字段名，不记成故障。

## 未覆盖

无
