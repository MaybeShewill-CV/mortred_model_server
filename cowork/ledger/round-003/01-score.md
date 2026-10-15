# 项目评分

- 轮次: 3
- 分数原始分: 9.0
- 分数加权分: 9.0
- 上限规则: 无
- 上一轮分数加权分: 8.9 → 9.0

## 模块
| id | 权重 | 分数 | 出处 |
| --- | --- | --- | --- |
| src-models | 15 | 9.0 | `cowork/ledger/round-003/01-modules/src-models.md` |
| src-server-apps | 20 | 9.4 | `cowork/ledger/round-003/01-modules/src-server-apps.md` |
| src-control | 15 | 9.3 | `cowork/ledger/round-003/01-modules/src-control.md` |
| src-factory-common | 10 | 9.2 | `cowork/ledger/round-003/01-modules/src-factory-common.md` |
| conf | 5 | 9.0 | `cowork/ledger/round-003/01-modules/conf.md` |
| ops | 15 | 8.8 | `cowork/ledger/round-003/01-modules/ops.md` |
| test | 10 | 8.8 | `cowork/ledger/round-003/01-modules/test.md` |
| docs | 5 | 7.8 | `cowork/ledger/round-003/01-modules/docs.md` |
| ci | 5 | 9.0 | `cowork/ledger/round-003/01-modules/ci.md` |

## P0
无

## P1
无

## P2
| id | 陈述 | 验收 | 现在的行为 | 一次外部验收即可确认的变化 | 出处 | 完成标准 |
| --- | --- | --- | --- | --- | --- | --- |
| P2-22 | 英文与中文模型服务教程里，非覆盖集三条启动命令的 `--model` 仍是 `YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN`。按教程执行会得到 `unknown --model`。 | 只改 `docs/tutorials_of_model_servers.md` 与 `docs/tutorials_of_model_servers.zh-cn.md`，把这三条改成 catalog id。 | 两份教程仍写 `--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN`。`src/apps/model_server_main.cpp` 写明 `--model` 大小写敏感且必须等于 catalog `model_section`。对应 id 是 `yolov5l`、`bisenetv2`、`attentive_gan`。第 2 轮关闭的 P2-12 只覆盖覆盖集的 `mobilenetv2` / `superpoint`。 | 只打开这两份教程：三条 `--model` 分别是 `yolov5l`、`bisenetv2`、`attentive_gan`。 | `cowork/ledger/round-003/01-modules/docs.md`、`docs/tutorials_of_model_servers.md`、`docs/tutorials_of_model_servers.zh-cn.md`、`src/apps/model_server_main.cpp`、`src/factory/obj_detection_task.h`、`src/factory/scene_segmentation_task.h`、`src/factory/enhancement_task.h` | 无 |
| P2-23 | `docs/deployment.md` 与中文版在 cpu 验收预期上仍要求随包两个 `*_cpu` 产品条目，与提交 `ea22867d` 后「不发布默认 cpu 产品」及 §10.7 矛盾。 | 只改 `docs/deployment.md` 与 `docs/deployment.zh-cn.md` 中 §5.2 预期与 §14.1 排障句，使其与 §10.7 一致。 | §5.2 写 cpu catalog「两个 `*_cpu` 条目 - mobilenetv2 / resnet」；§14.1 写「cpu 需要 `*_cpu` 配置」。中文对应句同样。仓库已无产品域 `*_cpu.toml` / `*_cpu_config.toml`；默认 `conf/packs/demo.toml` 为 TRT 种子，cpu profile 下该 id 不在目录。 | 只读这两份部署文档：§5.2 / §14.1（及中文）不再要求随包 `*_cpu` 产品；与 §10.7「自建 cpu pack」一致。 | `cowork/ledger/round-003/01-modules/docs.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`conf/packs/demo.toml`、`scripts/product_static_bs1.py` | 无 |

## 附录
| id | 陈述 | 处理说明 |
| --- | --- | --- |
| P0-1 | 驻留集中的 `libface` 与一份合法 JPEG 不可成功；产品 toml 与 TensorRT profile 高宽不一致。 | 重复描述 |
| P1-1 | 不依赖 `/tmp/mortred_stage_timing` 时，gateway `/metrics` 读不到五段耗时（PF4）。 | 重复描述 |
| P1-2 | 中文添加中型目录模型时，英文 how-to / model-developer-guide 与 `catalog()` 不一致（EX1）。 | 重复描述 |
| P2-1 … P2-7、P2-9 … P2-12、P2-14、P2-16、P2-17、P2-19、P2-20、P2-21 | 第 2 轮及更早已关闭的各项。 | 重复描述；当前文件仍满足关闭时的验收，不重开 |
| P2-8 | `conf/onnx_sources.json` 与覆盖集产品 toml 后端/路径不一致。 | 范围外 |
| P2-13、P2-15、P2-18 | 第 2 轮锁定附录并入项。 | 重复描述 |

## 未覆盖模块
无
