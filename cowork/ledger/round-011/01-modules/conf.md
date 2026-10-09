# 模块评分

- id: conf
- 权重: 10
- 分数: 8.1
- 读过的路径: `cowork/ledger/rubric.md`；`conf/onnx_sources.json`；`conf/model/object_detection/yolov8/yolov8s.toml`；`conf/server/object_detection/yolov8/yolov8s_server_config.toml`；`conf/packs/`（`demo.toml`、`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml`）；核对用：范围内 39 个 id 的 `conf/model/**` 产品 toml 与对应 `conf/server/**`、`conf/product_static_bs1.json`、`scripts/product_static_bs1.py`、`scripts/mortredctl_prepare.sh`、`scripts/prepare_pack.sh`、`scripts/mortredctl_doctor.sh`、`scripts/pack_trt.py`；对照 `src/factory` 各 `catalog()` / `face_catalog()`

## 优势
- 范围内 39 个 catalog id 各有一份产品 toml，`type = "tensorrt"`，`model_file_path` 都指向 `*.engine.static_bs1.fp16` 或 `fp32`。`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x` 以及 `libface`、`centerface` 各用各的文件，没有互相代替。
- 这些路径与 `conf/product_static_bs1.json` 的 `engine` 一致（toml 多一层 `../`）。对应 server toml 的 `model=` 与 `model_config_file_path` 指回同一份产品 toml。H2 若加载产品 toml，读到的就是这份 engine 路径。

## 劣势
- H4 实际消费的是 pack，不是这份产品 engine 清单。`mortredctl prepare` / `doctor` 默认 pack 是 `conf/packs/demo.toml`，其中 `mobilenetv2` 的 `model_config` 指向 `mobilenetv2_cpu.toml`（`type = "mnn"`），没有 TensorRT engine。入库 GPU pack 只选中 `yolov8s` 与 `metric3d_512`。其余 type=tensorrt id（含 `yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`）没有一份现成 pack 把产品 toml 里的 engine 交给 prepare。不设 `MORTRED_PACK` 时，删掉例如 `weights/object_detection/yolov8/yolov8n.engine.static_bs1.fp16` 后执行 `mortredctl doctor --strict` 仍退出 0。改完后，同一命令对该 id 缺这份 engine 时退出非 0，输出点名该路径，并给出能生成它的下一行 `mortredctl prepare` 命令。

## 不计毛病
- engine 须在本机本卡生成，不把「把 engine 文件拷到另一台机器」写成缺口。
- 范围内产品后端保持 `tensorrt`。SAM、diffusion、CLIP、LightGlue、OCR 识别器不在本次核对范围。

## 依据
- 分数 8.1：39 个 id 的产品 toml、server 指针和 `product_static_bs1` engine 已经对齐，主配置能交付；默认 prepare/doctor 不读取这些路径，缺 engine 仍报成功，所以到不了独立换机可定位。
- 优势出处：各产品 toml 的 `type` / `model_file_path`；`scripts/product_static_bs1.py` 的 `model_rows()`；`conf/product_static_bs1.json`；各 `*_server_config.toml` 的 `model` 与 `model_config_file_path`。
- 劣势出处：`conf/packs/demo.toml`、`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml`；`scripts/mortredctl_prepare.sh` 与 `scripts/mortredctl_doctor.sh` 的默认 pack；`scripts/prepare_pack.sh` 在 pack 无 TensorRT backend 时退出 0；`scripts/pack_trt.py` 只解析 pack 里的 id。
- 已关闭的 P1-3、P1-4、D-2 原句不送回。

## 未读
无
