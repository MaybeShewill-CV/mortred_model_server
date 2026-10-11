# 销账

- 轮次: 12
- 已关闭: P1-12
- 挂起: P0-2（工作树缺 39 个产品 onnx，且工作树看不见 trtexec。未跑验收 6、7、8、12）。P1-8（msocrnet 在约 19GiB 的 WSL 上被 OOM 杀掉，没有黄金；工作树没有 full 构建的 model_golden_test。未跑验收 3 的 msocrnet 与 19 条同进程过滤，以及验收 11 的 full 列表）
- 非挂起劣势: 相对本轮起点减少

## 删除的原句
| id | 模块 | 原句 |
| --- | --- | --- |
| P1-12 | test | H2：除 `yolov8s` 以外、产品 toml 为 `type=tensorrt` 的 id 没有同规则 TRT 比较 |

## 留下的劣势
| id | 模块 | 原句 | 状态 |
| --- | --- | --- | --- |
| P0-2 | conf、ops | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:` | 挂起 |
| P1-8 | test | H2：点名的 id 没有 CPU 数值用例 | 挂起 |
| D-8 | ops | S3：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因 | 未做 |
| D-9 | src-control | S4：推理请求在鉴权失败时仍消耗限流配额 | 未做 |
| D-10 | src-control | S6：`nvidia-smi` 读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率 | 未做 |
| D-11 | docs | `deployment.md` 把 cpu 目录写成 `resnet50`，并把 `msocrnet` 写成 Bench-only | 未做 |
| D-12 | src-control | S2：一次拉起后，检查失败并点名 fork 与 exec 之间的调用 | 未做 |

## 附录
同义句：除 `yolov8s` 以外、产品 toml 为 `type=tensorrt` 的 id 没有同规则 TRT 比较。关闭范围只含这条。merge commit `0b2a75980f481814a63d66e02ac18628f3f2a03b`，问题分支提交 `611ecea4`。QA 结论为 pass。diff 是 `test/model_golden_test.cc`、`test/model_golden_registry.h`，以及 18 个 `test/golden/trt_<id>.json`。

`yolov8s` 的已关闭 IoU 门不写入这条的完成。P1-5 只含 `yolov8s`。P1-11 只含 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 CPU 全分辨率掩码重合。

P1-8 仍挂起，所以 H2 整行仍是未满足。P0-2 仍挂起。已关闭原句不重开：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、D-1 至 D-6。`centerface` 的 engine 输入名 `input.1` 不重开。

方案 v2 把这 9 条的失败原文当成期望，QA 已通过，不写成新劣势：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`yolov7`、`libface`、`enlightengan`、`dinov2_vitb14`。验收 5 至 13 退出码都是 1，输出含对应失败原文，不含 `NUMERICAL_PASSED`，不含 `device=cpu`。

`TrtProductGolden.msocrnet` 本机未执行。二进制含 `conf/model/scene_segmentation/msocrnet/msocrnet.toml`，验收 1 与验收 3 都没有该过滤的 `NUMERICAL_PASSED`。这不是挂起，也不新开。

## 新准入
无
