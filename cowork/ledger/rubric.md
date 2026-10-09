# 落地评分口径

自 2026-10-09 起，目标有两条，两条都要达到：

1. 个人项目天花板：分类、目标检测、人脸检测、分割、OCR、图像增强、特征点检测、feature embedding、matting、深度估计里，HTTP catalog 的每一个 id 都达到下面的最高标准。YOLOv8 含 `yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`，不能用其中一个代替其余。对照是 llama.cpp、KuiperInfer，以及同量级的个人 model server。不对照 Triton 的功能清单。
2. 中小企业生产落地：上述每一个 id 都能部署、能换机、能定位错误，并且不会把失败当成成功。

第 10 轮的 `done` 只表示旧出口「有效劣势无且挂起为空」已经满足。它不表示本文件的双目标已经达到。口径变更后的下一轮从第 1 步重打分。已关闭 id 的原句和关闭范围不送回队列；同义句放附录。新的可观察行为可以新开 id。

综合分按本文件计算，只作记录，不决定出口，也不决定劣势能不能写「无」。

评分人是已经在中小企业 Linux 服务器上使用 Mortred 的 CTO。分数表示：这个模块让小团队拿去交付客户项目时有多能用。0 是无法使用，10 是小团队可以独立交付并在出错时定位。

## 模块

| id | 路径 | 权重 |
| --- | --- | --- |
| src-models | `src/models` | 15 |
| src-server-apps | `src/server`、`src/apps` | 15 |
| src-control | `src/control` | 10 |
| src-factory-common | `src/factory`、`src/common` | 5 |
| conf | `conf/` | 10 |
| ops | `scripts/`、`deploy/`、`mortredctl` | 15 |
| test | `test/` | 15 |
| docs | `docs/`、`README.md`、`README.zh-cn.md` | 10 |
| ci | `.github/` | 5 |

九个模块都要有独立记录。没读过的模块不能打分，也不能从分母里去掉。

模块分取 0.0 到 10.0，保留一位小数。每条优势、劣势和分数都要有文件、命令或测试出处。没有出处的分数无效，综合分不得计算。模块没有未关闭的准入项时，劣势写「无」，分数照实记录。

加权综合分 = 各模块（分数 × 权重）之和 / 100，保留一位小数。

## 准入

一条劣势在进入队列前要同时有：出处文件、现在的行为、改完后一次黑盒就能判定的变化。做不到的句子在冻结前拆成有限清单，或移入不计毛病。已关闭问题的同义改写写入附录，不留在劣势里，也不进入队列。

## 问题

P0：现有接法让一次客户交付发不出去，或出了错无法定位。

P1：能交付，但每上一个模型、换一台机器都要重复一套易错手工步骤。

通过准入的每一条劣势都进入实现队列。达不到 P0 或 P1 的，列入「劣势」，编号 D-1、D-2。同一类痛点并成一条。

队列跨轮保留。已签字且未关闭、未挂起的 id 原样进入下一轮，不再表决。只对新 id 表决。

## 销账

QA 通过并合入的 id，对应的劣势原句从有效劣势集合删除。调度者核对。原句还在有效劣势里的记录退回。同义句放附录。销账不要求把模块分改到 9.0。

## 挂起

验收因环境跑不了而 `blocked` 时，该条挂起：写明缺的环境、恢复条件和未跑的验收。挂起项留在队列里，不挡住其余项。挂起不计入「劣势集合没有减少」。环境恢复后只重跑挂起项，不重新打分，也不重开方案。

## 双目标验收

每一轮打分都要逐条复核。未满足的行必须进入 P0、P1 或劣势，并写出出处、现在的行为和一次黑盒可判定的变化。已满足的行写出证据，不新开 id。挂起只用于环境跑不了的验收。调度者在核算文件里逐行写「已满足 / 未满足 / 挂起」。有一行未满足或挂起时，不得把出口写成 `done`。模块记录把未满足行写成「无」，该记录退回。

同一类痛点并成一条。已关闭的关闭范围不扩大，也不把「其余都做了」写成这一条的完成。不得用「边界写明不做」把下表里的 id 移出最高标准。

### 范围

最高标准施加于这些 HTTP catalog 家族的每一个 id。id 以打分时的 `catalog()` 为准；下表是 2026-10-09 的名单，新增 id 自动进入，删除 id 不再要求。

| 任务 | catalog | id |
| --- | --- | --- |
| 分类 | `classification` | `mobilenetv2`、`resnet`、`densenet` |
| 目标检测 | `object_detection` | `yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x` |
| 人脸检测 | `face_catalog()` | `libface`、`centerface` |
| 分割 | `scene_segmentation` | `bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet` |
| OCR | `ocr` | `dbnet` |
| 图像增强 | `enhancement` | `attentive_gan`、`enlightengan`、`realesrgan` |
| 特征点检测 | `feature_point` | `superpoint` |
| feature embedding | `feature_embedding` | `dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14` |
| matting | `matting` | `modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2` |
| 深度估计 | `mono_depth_estimation` | `depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` |

`yolov8s` 已关闭的同类 IoU 门只覆盖 `yolov8s`。`yolov8n`、`yolov8l`、`yolov8x` 各自单独证明。

### 最高标准

上表每一个 id 都要同时满足 H1 至 H4。一个 id 做完不算该任务完成。

| 编号 | 未满足时的可观察标准 |
| --- | --- |
| H1 | 该 id 没有一条与 `docs/openapi.json` 一致的请求和响应示例 |
| H2 | 该 id 没有与 CPU golden 同级的数值比较。目标检测和人脸检测的框要求同类且 IoU ≥ 0.5。分类要求 top-1 一致。分割和 matting 要求掩码重合达到测试里写明的阈值。OCR 要求检测框或多边形重合达到测试里写明的阈值。增强、特征点、embedding、深度使用该测试写明的距离或误差阈值。产品 toml 为 `type=tensorrt` 的 id 还要有一条规则相同的 TRT 比较。缺权重时，设置了要求权重则失败；选中用例全部因缺权重跳过则进程退出 77 并打印 `NUMERICAL_NOT_RUN`。静默变绿算未满足 |
| H3 | 该 id 没有一行同时写出硬件、复现命令、指标和日期的基准 |
| H4 | 换到另一台机器时，不能用现成命令为该 id 生成 engine；或 `mortredctl doctor --strict` 在该 id 缺 engine 时仍退出 0，或失败输出里没有下一行命令 |

### 全服务

这些行不按 id 豁免。

| 编号 | 未满足时的可观察标准 |
| --- | --- |
| C1 | 中文和英文 README 的 CPU、GPU 源码构建命令不一致，或与 `CMakePresets.json` 的 `full` / `full-cpu` 不一致 |
| C4 | CI 不再检查 OpenAPI 与仓库一致，或 GPU 作业被跳过时仍会被写成 TensorRT 证明 |
| S1 | 已服务路径上，契约失败仍报成空输出。已关闭范围不重开；新入口上的新证据才开新 id |
| S2 | supervisor 在 `fork` 之后、`exec` 之前调用 `setenv` 或其他非 async-signal-safe 函数 |
| S3 | 默认发布镜像以 root 运行，且边界文档没有写明必须 root 的原因 |
| S4 | 推理请求在鉴权失败时仍消耗限流配额，或未鉴权即可调用推理路由。健康检查豁免不在此列 |
| S6 | `nvidia-smi` 失败或读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率，且日志或状态没有标明 offline |

## 不计毛病

下列内容不写成模块劣势，不进入实现队列，可以记在附录。双目标表里的行不在此列。

- `docs/unsupported-boundaries.md` 列出的不支持项。HTTP OCR 只有 `dbnet` 检测、没有识别；识别器不因「OCR 要达到最高标准」而进入队列。`dbnet` 本身仍要满足 H1 至 H4。
- SAM、diffusion、CLIP、LightGlue。它们不在最高标准的范围表里。
- 一个 catalog id 一个 OS 进程。超时 drain 之后析构成员，再由外层 SIGKILL。
- 训练平台、标注平台、多云平台。
- 为对齐企业推理平台而新增的 gRPC、ensemble、模型热加载、模型版本库、官方客户端 SDK。
- 没有真实交付损失就推翻 HTTP 契约或现有后端。
- 一次黑盒判定不了、又拆不成有限清单的句子。其中已由签字方案收窄并关闭的，只保留方案里的关闭范围。

## 上限

最终列表里还有未关闭且未挂起的 P0 时，综合分不得超过 7.9。还有未关闭且未挂起的 P1、没有这种 P0 时，综合分不得超过 8.9。挂起项不触发上限。先按权重算出原始分，再套上限。这些上限不代替出口。

## 出口

`done`：双目标验收每一行都是「已满足」，九份模块的有效劣势都是「无」，且挂起集为空。三件同时成立才算完成。只清空旧队列不算完成。

`awaiting-env`：非挂起劣势都是「无」，双目标里没有「未满足」，但仍有「挂起」。恢复环境后只重跑挂起项。

停止：本轮有非挂起项合入或移入不计毛病，但销账后非挂起劣势集合一条都没减少，且没有任何双目标行从「未满足」变成「已满足」。

综合分只作记录。有效劣势是尚未关闭、且没有移入附录或不计毛病的条目。挂起项仍算未关闭，只是不参与本轮的减少判定。
