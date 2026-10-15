# 模块评分

- id: conf
- 权重: 5
- 分数: 9.0
- 读过的路径: `conf/onnx_sources.json`、`conf/model/classification/mobilenetv2/mobilenetv2.toml`、`conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml`、覆盖集其余八份产品 model/server toml、`conf/packs/`（`demo.toml`、`yolov8.toml`、`metric3d.toml`、`gpu_yolo_metric3d.toml`）、`conf/ci/` 下 hosted/golden toml、`conf/weights_manifest.json`、`scripts/product_static_bs1.py` 中禁止再发布产品域 `*_cpu.toml` 的门闩

## 已具备

- 覆盖集九份产品 toml 均为 `type=tensorrt`、`device=gpu`，engine 名带 `static_bs1`。
- 九份覆盖集 `conf/server` 的 `worker_nums=1`，`model=` 与 catalog id 一致。
- 提交 `ea22867d` 后仓库**不再**发布默认 cpu 产品：`conf/model/**/mobilenetv2_cpu.toml` 与 `conf/server/**/*_cpu_config.toml` 已删除；`scripts/product_static_bs1.py` 若它们再现则记 error。cpu 用法改由 pack + `docs/deployment.md` §10.7 组装；CI 仍用 `conf/ci/*_cpu_golden.toml` / `mobilenetv2_onnx_hosted.toml`。
- `conf/packs/demo.toml` 种子为 TensorRT 产品（`[pack.mobilenetv2]` 指向 `mobilenetv2.toml`，`gpu_mem_source = "seed"`）。注释写明 cpu profile 下该 id 不在目录、需按文档自建 cpu pack。

## 问题

无

## 范围外

- P2-8 交换目录与产品路径不一致仍在范围外。
- `canonical_size` 仅出现在 metric3d toml，不记故障。
- cpu profile 上 `type=tensorrt` 是否进 ready：未在本机跑，未验证判断。

## 依据

分数 9.0：覆盖集产品与 server 契约、demo pack 的 TRT 种子、以及「不发布默认 cpu 产品」的门闩与文档方向一致。EX3/AV1 在产品文件上的触发条件对得上。本模块无未关闭问题。文档 §5.2/§14.1 仍写「两个 `*_cpu` 条目」的过时句，出处在 `docs/`，记入 docs 模块，不记入 conf。

## 未覆盖

无
