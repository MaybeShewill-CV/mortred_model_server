# 模块评分

- id: docs
- 权重: 5
- 分数: 7.8
- 读过的路径: `docs/api-contract.md`、`docs/unsupported-boundaries.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`docs/api-contract.zh-cn.md`、`docs/tutorials_of_model_servers.md`、`docs/tutorials_of_model_servers.zh-cn.md`、`docs/how_to_add_new_server.md`、`docs/how_to_add_new_server.zh-cn.md`、`README.md`、`README.zh-cn.md`、`docs/openapi.json`（对照响应示例）、`docs/perf/coverage_set_http_rps.md`

## 已具备

- 第 2 轮已关闭的覆盖集教程 `--model mobilenetv2` / `superpoint`、worker_nums 与 pack 注入表述、中英文 how-to 的 `densenet`、接口契约 `model.name` 与 openapi 对齐，当前文件仍保持。
- `docs/deployment.md` §10.7 与中文对应节写明：不发布默认 cpu 产品；cpu 用 pack + 自写 mnn/onnx toml；`conf/packs/demo.toml` 为 TRT 种子。
- `docs/unsupported-boundaries.md` 写明不期望默认 cpu 产品随包，或 `type=tensorrt` 在 cpu 上跑通。
- 第一小时路径仍能从 `mortredctl next` 走到 `doctor`（UX2 文档侧主干仍在）。

## 问题

1. 英文与中文模型服务教程里，非覆盖集三条启动命令的 `--model` 仍是大写：`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN`。catalog / server toml 对应 id 是 `yolov5l`、`bisenetv2`、`attentive_gan`（`src/factory/obj_detection_task.h`、`scene_segmentation_task.h`、`enhancement_task.h`；`src/apps/model_server_main.cpp` 写明 `--model` 大小写敏感）。按这三条命令会得到 `unknown --model`。第 2 轮关闭范围只覆盖了覆盖集的 `mobilenetv2` / `superpoint`，这三条仍可观察。改完后一次外部验收：只打开两份教程，这三条 `--model` 分别等于 `yolov5l`、`bisenetv2`、`attentive_gan`。

2. `docs/deployment.md` §5.2 仍写 cpu catalog「恰好两个 `*_cpu` 条目（mobilenetv2 / resnet）」；§14.1 仍写「cpu 需要 `*_cpu` 配置」。中文 `docs/deployment.zh-cn.md` 对应句同样过时。提交 `ea22867d` 后产品域 `*_cpu.toml` / `*_cpu_config.toml` 已删除，§10.7 与 `conf/packs/demo.toml` 写明默认 cpu compose 只起 supervisor+gateway。按 §5 用默认 demo pack 做 cpu 验收时，读者会以为空 catalog 是故障。改完后一次外部验收：§5.2 / §14.1（及中文对应）与 §10.7 一致，不再要求随包 `*_cpu` 产品条目。

## 范围外

- 已关闭的 P2-12（仅覆盖集两条）、P2-14、P2-16、P2-17、P2-10、P1-2 等重复描述不重新入列。
- `docs/ui-redesign/`、monitoring、about_model 长文等未逐份读完的可选文档；必读三份已读，不触发「未覆盖路径封顶 8.0」。

## 依据

分数 7.8：第一小时、边界、OpenAPI 示例与九行 HTTP 报告仍能支撑 GPU 上线；覆盖集教程与契约示例已对齐。仍有教程三条错误 `--model`，以及 deployment 在 `ea22867d` 后未改干净的 cpu catalog 预期，所以到不了「按文档无歧义地走完 cpu 验收或教程里每一条启动命令」。高于第 2 轮 7.4，是因为覆盖集相关 P2 已关闭。

UX1：中英文 README 的 cpu/gpu 构建命令与 `full` / `full-cpu` 一致。UX4：openapi 示例与契约对齐的覆盖集部分仍在。EX1 点名英文指南不要求每模型单独二进制。

## 未覆盖

无
