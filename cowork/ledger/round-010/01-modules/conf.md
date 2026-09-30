# 模块评分

- id: conf
- 权重: 10
- 分数: 7.9
- 读过的路径: `cowork/ledger/rubric.md`；`cowork/ledger/round-010/00-rule-change.md`；`conf/onnx_sources.json`；`conf/model/object_detection/yolov8/yolov8s.toml`；`conf/server/object_detection/yolov8/yolov8s_server_config.toml`；`conf/packs/`（`demo.toml`、`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml`）；核对用：`conf/trt_engines.json`；`conf/trt_engines.set_a.json`；`conf/product_static_bs1.json`；`conf/model/mono_depth_estimation/depth_anything/depth_vits14.toml`；`conf/model/mono_depth_estimation/metric3d/metric3d_512.toml`；`conf/model/scene_segmentation/hrnet/hrnet.toml`；`conf/model/enhancement/attentive_gan_derain/attentive_gan.toml`

## 优势
- 产品 GPU 真源已对齐：`product_static_bs1.json` 39 条 engine 路径与对应 model toml 一致；`trt_engines.set_a.json` 同写 `*.engine.static_bs1.fp16`（含 yolov8s / depth_* / metric3d_* / hrnet）。
- yolov8s 主路径闭环：`yolov8s.toml`（tensorrt + `yolov8s.engine.static_bs1.fp16`）与 `yolov8s_server_config.toml`（port/URI/`model_config_file_path`）成对；`trt_engines.json` 中 yolov8s 的 engine 已同步为同名。
- pack 具备可 gate 的 seed 占用与换机注释：`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml` 写 `gpu_mem_mib=1800`、`gpu_mem_source=seed`，并标明须本机 `calibrate --write-pack`。

## 劣势
- 遗留 `conf/trt_engines.json` 仍用旧 engine 名，与产品路径不一致：depth_anything_*、metric3d_* 仍为 `*.engine`，hrnet 仍为 `hrnetw48_ccd_fp32.engine`；产品 toml / `product_static_bs1` / set_a 已是 `*.engine.static_bs1.fp16`。若按遗留清单 convert，会建到错误文件。
- `conf/onnx_sources.json` 与产品 toml 漂移：大量条目仍标 `product_type=mnn` 且 `runtime_path` 指向 `.mnn`/`.model`/旧 `.engine`，而对应 `conf/model/**/*.toml` 已是 `type=tensorrt` + `*.engine.static_bs1.*`（如 mobilenetv2、yolov5l、depth_vits14、metric3d_512、hrnet）；出错时 conf 内多真源，难定位。
- `attentive_gan.toml` 仍写 `model_input_image_size = [240, 320]`，`onnx_sources.json` ATTENTIVE_GAN 已注明 session 实为 240×360、toml 尺寸 stale。
- pack 覆盖仍偏样板：仅 4 份；GPU 侧是 seed 占用而非本机校准结果；`demo.toml` 只有 `worker_nums`，多客户同机仍要机外拷贝与重校准。

## 不计毛病
- 跨机拷贝 TensorRT engine（须本机本卡 prepare）。
- 把产品 toml 改成 `type=onnx`（产品后端约定保持 mnn/tensorrt；既有 onnx 例外除外）。

## 依据
- 分数 7.9：产品表 + model toml + set_a 已能支撑主交付链，且 yolov8s 遗留名已修；但遗留 `trt_engines.json`、`onnx_sources` 与产品 toml 不一致、ATTENTIVE 尺寸 stale、pack 偏薄，仍挡「独立交付且出错必能从 conf 定位」。
- 优势出处：`conf/product_static_bs1.json`；`conf/trt_engines.set_a.json`；`conf/model/.../yolov8s.toml`；`conf/server/.../yolov8s_server_config.toml`；`conf/trt_engines.json` yolov8s；`conf/packs/{yolov8,metric3d,gpu_yolo_metric3d}.toml`。
- 劣势出处：`conf/trt_engines.json` depth/metric3d/hrnet 条目 vs 对应 model toml / set_a；`conf/onnx_sources.json` `product_type`/`runtime_path` vs `conf/model/**/*.toml`；`conf/model/.../attentive_gan.toml` + onnx_sources ATTENTIVE_GAN notes；`conf/packs/` 四文件。

## 未读
无
