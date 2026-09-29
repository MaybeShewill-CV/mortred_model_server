# 模块评分

- id: conf
- 权重: 10
- 分数: 8.0
- 读过的路径: `cowork/ledger/rubric.md`; `cowork/ledger/round-001/01-modules/conf.md`; `conf/onnx_sources.json`; `conf/model/object_detection/yolov8/yolov8s.toml`; `conf/server/object_detection/yolov8/yolov8s_server_config.toml`; `conf/packs/`（`demo.toml`、`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml`）；`conf/product_static_bs1.json`（yolov8s）；`conf/trt_engines.json`（yolov8s）；`conf/trt_engines.set_a.json`（yolov8s）

## 优势
- 产品交付主路径上的 yolov8s engine 真源已对齐：`onnx_sources.json` YOLOV8 `runtime_path`、`yolov8s.toml` `model_file_path`、`product_static_bs1.json`、`trt_engines.set_a.json` 同为 `yolov8s.engine.static_bs1.fp16`，上轮 `fp16in` 分裂已消除。
- 单品链路仍完整：`yolov8s.toml`（tensorrt）与 `yolov8s_server_config.toml`（port/URI/`model_config_file_path`）成对，可按 catalog id 起检测服务。
- 代表性 GPU pack 具备可 gate 的 seed 占用：`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml` 均写 `gpu_mem_mib=1800`、`gpu_mem_source=seed`，并注明须本机 `calibrate --write-pack`；多 id pack 给出 copy→prepare→calibrate→doctor 流程注释。

## 劣势
- 遗留转换清单仍用旧 engine 名：`trt_engines.json` yolov8s 仍是 `yolov8s.engine` + 旧 `yolov8s.onnx`，与 set_a/产品 toml 并存；若按该文件 prepare，仍可能建错文件。
- pack 仍偏样板：仅 4 份，GPU 侧是 seed 占用而非本机校准结果；`demo.toml` 仍只有 `worker_nums`，多客户同机并发仍要机外拷贝与重校准。
- conf 树命名分散（`*_config.toml` / 短名、两套 trt 清单）未消；`onnx_sources` 仍标 ATTENTIVE_GAN toml 尺寸 stale（`[240,320]` vs session 240×360），削弱「改 conf 即真相」。

## 依据
- 分数 8.0：相对上轮 6.8，主产品 engine 路径共识与 pack seed occupancy 明显改善；遗留 `trt_engines.json`、pack 覆盖与命名/stale 仍挡「独立交付且出错必能定位」。
- 优势出处：`conf/onnx_sources.json` YOLOV8；`conf/model/.../yolov8s.toml`；`conf/server/.../yolov8s_server_config.toml`；`conf/product_static_bs1.json`；`conf/trt_engines.set_a.json`；`conf/packs/{yolov8,metric3d,gpu_yolo_metric3d}.toml`。
- 劣势出处：`conf/trt_engines.json` yolov8s `engine`/`onnx`；`conf/packs/` 四文件及 seed 注释；`conf/onnx_sources.json` ATTENTIVE_GAN stale 说明。

## 未读
无
