# 交付评分标准

自 2026-10-12 起，目标只有一条：把 Mortred 做成中型企业能放上一台 Linux GPU 服务器的 CV 推理服务。业务后端通过 gateway 调用。一个小组负责运维。评价只看六件事：单机性能、稳定性、吞吐、易用、运维、扩展。

对照的是同量级的单机推理服务。不对照 Triton 的功能清单，也不对照多机调度。

git 历史里的旧轮次整段忽略：不检回工作区，不读取，不继承队列、已关闭 id、分数或阻塞项。那些记录里的 `done` 不表示本文件的完成标准已经达到。新的执行从第 1 轮第 1 步开始，记录直接写到 `cowork/ledger/round-001/`。完成标准是否满足只看本文件和这一轮新产生的证据。

加权分按本文件计算，只作记录，不决定终态，也不决定问题能不能写「无」。

评分人是已经在中型企业的一台 Linux GPU 服务器上使用 Mortred 的 CTO。分数表示：这个模块能不能放进这台生产机器。0 是不能放。10 是一个小组可以在这一台上线，讲清一次失败的时间构成，给出 HTTP 吞吐，并按文档加上一个模型。

## 基准包

中型企业在这一台上同时用到的是一组模型，不是目录里的每一个尺寸。

覆盖集固定为九个任务各一个 HTTP id：

| 任务 | id |
| --- | --- |
| 分类 | `mobilenetv2` |
| 检测 | `yolov8s` |
| 人脸 | `libface` |
| 分割 | `pphuman_mobile` |
| OCR 检测 | `dbnet` |
| embedding | `dinov2_vits14` |
| 特征点 | `superpoint` |
| 增强 | `realesrgan` |
| 单目深度 | `metric3d_512` |

驻留集是覆盖集里这台卡经 `mortredctl doctor --strict` 允许同时拉起的子集，至少包含 `yolov8s` 和 `libface`。九个 id 都要能单独 prepare、单独拉起、单独完成一次成功推理。不要求九个同时驻留。显存装不下九个时，把同时驻留收成 doctor 能通过的子集；这不免除「每个 id 单独能推理」。

本机 WSL 的默认发行版 Ubuntu 提供 GPU 和本程序 runtime。性能、吞吐、engine、占用都先在这个 WSL 里取证，仓库路径是 `/mnt/g/Codex/mortred_model_server`。Windows 本机 shell 跑不了这些命令，不算没有 GPU。只有 WSL 里确认缺少 GPU 或 runtime 时，这几行才标阻塞，不标已满足。cpu profile 仍须能构建，并能用 MNN 或 ORT 单独跑通 `mobilenetv2`。`type=tensorrt` 在 cpu profile 上必须初始化失败并给出明确错误，进程不得进入 ready。

## 本次性能约束

服务性能只优化 `type=tensorrt`，并且只优化 static shape、batch=1 的 engine。MNN 与 ONNX Runtime 只要求服务可用：能初始化，能完成一次成功推理，失败不报成成功。不为二者做吞吐或延迟优化。

模型进程里每个 worker 持有各自的 TensorRT execution context。这个结构保持不变。多个 worker 共享一个 context，或在 server 里增加跨 worker 的 context 池，不进入实现队列。

覆盖集中 static batch=1 的 trtexec 结论以 `docs/benchmark-h3/<id>.txt` 为准。文件中有硬件、日期、命令和 Throughput。该 Throughput 是引擎参照，不是 HTTP 吞吐。HTTP 压测报告要引用同 id 文件里的 Throughput，并写出 HTTP 成功 RPS 与它的比值。

某个 TensorRT id 的 HTTP RPS 按这个顺序取得，缺一步的数字不算数：

1. 对该 id 做 calibrate，选出本机的 `worker_nums` 和 JPEG 解码路径。命令是 `mortredctl calibrate --pack <pack> --write-pack`。worker 数扫描和 JPEG 解码投票的请求都走 `--raw`（图片原始字节）。用 base64 JSON 跑出的结论不算数。结论写在该 pack 的 `[pack.<id>]`：`worker_nums`，以及 `decode_auto` 为 `cpu` 或 `gpu`。`conf/server` 保持 `worker_nums=1`，不把结论写进那里。
2. 按这份 pack 部署并拉起该 id。进程里的 worker 数等于 `worker_nums`，`MORTRED_DECODE_AUTO` 等于 `decode_auto`。
3. 用 `scripts/server/http_infer_rps.py` 压测。必须带 `--raw`。并发数等于已部署的 `worker_nums` 的两倍，测量时长不少于 30 秒。数据是图片原始字节，不是 base64 JSON。

压测期间 `/metrics` 的 `mortred_jpeg_decode_total` 要落在 `decode_auto` 选出的那条路径上。`decode_auto=gpu` 时 `jpeggpu` 计数要增加；`decode_auto=cpu` 时走 CPU 解码。实际路径和结论不一致的报告不算数。

## 模块

| id | 路径 | 权重 |
| --- | --- | --- |
| src-models | `src/models` | 15 |
| src-server-apps | `src/server`、`src/apps` | 20 |
| src-control | `src/control` | 15 |
| src-factory-common | `src/factory`、`src/common` | 10 |
| conf | `conf/` | 5 |
| ops | `scripts/`、`deploy/`、`mortredctl` | 15 |
| test | `test/` | 10 |
| docs | `docs/`、`README.md`、`README.zh-cn.md` | 5 |
| ci | `.github/` | 5 |

九个模块都要有独立记录。没读过的模块不能评分，也不能从分母里去掉。

模块分取 0.0 到 10.0，保留一位小数。已具备、问题和分数都要有文件、命令或测试出处。没有出处的分数无效，原始分不得计算。模块没有未关闭的入列项时，问题写「无」，分数照实记录。

原始分 = 各模块（分数 × 权重）之和 / 100，保留一位小数。记录用的加权分是原始分套上限之后的结果。

## 入列条件

一条问题在进入队列前要同时有：出处文件、现在的行为、改完后一次外部验收即可确认的变化。做不到的句子在锁定前拆成有限清单，或移入范围外。已关闭问题的重复描述写入附录，不留在 P2 里，也不进入队列。

## 问题

P0：这台机器上的业务调用发不出去；失败被报成成功；一个模型故障或一次过载让整机入口不可用；过载或崩溃之后进程不再按已写明的策略恢复。

P1：能够服务，但换机、加一个已有家族的模型、看清一次请求的时间构成、或处理一次单模型故障，都要重复一套文档里没有的手工步骤。

满足入列条件的每一条问题都进入实现队列。达不到 P0 或 P1 的，列入 P2，编号 P2-1、P2-2。同一类问题并成一条。

队列跨轮保留。已确认且未关闭、未阻塞的 id 原样进入下一轮，不再评审。只对新 id 评审。

## 关闭

QA 通过并合入的 id，对应的问题陈述从开放项删除。编排方核对。问题陈述还在开放项里的记录退回。重复描述放附录。关闭不要求把模块分改到 9.0。

## 阻塞

验收因环境跑不了而 `blocked` 时，该条标为阻塞：写明缺的环境、恢复条件和未跑的验收。阻塞项留在队列里，不挡住其余项。阻塞不计入「开放项没有减少」。环境恢复后只重跑阻塞项，不重新评分，也不重开方案。需要 GPU 或 runtime 的验收先在本机 WSL Ubuntu 里跑。只有 WSL 里确实跑不了时才阻塞，并且只阻塞性能、吞吐、engine 和占用行。

## 完成标准

每一轮评分都要逐条复核。未满足的行必须进入 P0、P1 或 P2，并写出出处、现在的行为和一次外部验收即可确认的变化。已满足的行写出证据，不新开 id。阻塞只用于环境跑不了的验收。编排方在评分复核里逐行写「已满足 / 未满足 / 阻塞」。有一行未满足或阻塞时，不得把终态写成 `done`。模块记录把未满足行写成「无」，该记录退回。

同一类问题并成一条。已关闭的关闭范围不扩大，也不把「其余都做了」写成这一条的完成。不得用「边界写明不做」把下表的行移出完成标准。

压测使用 `scripts/server/http_infer_rps.py --raw`，并发数等于该 id 已部署的 `worker_nums` 的两倍，请求的是 static shape、batch=1 的 TensorRT engine。不带 `--raw`、或并发数不等于已部署 `worker_nums` 的两倍的结果不算数。trtexec 的 Throughput 只作 `docs/benchmark-h3/<id>.txt` 中的引擎参照，不算 HTTP 吞吐。GPU 作业被跳过时，不得把该次结果写成已满足。

### 性能

| 编号 | 未满足时的可观察标准 |
| --- | --- |
| PF1 | 覆盖集中任一 `type=tensorrt` 的 id，没有一份按「本次性能约束」完成 calibrate、按结论部署、再用 `http_infer_rps.py --raw` 压测的报告。报告须同时写明硬件、日期、pack 里的 `worker_nums` 与 `decode_auto`、压测命令、成功 RPS、p50、p99、错误率，以及 `docs/benchmark-h3/<id>.txt` 里的 Throughput 和二者比值。命令不是 `--raw`，或并发数不等于已部署的 `worker_nums` 的两倍，算未满足 |
| PF2 | 同一次压测读不到模型进程 `/metrics` 里分开的排队等待和推理耗时，或排队等待被算进推理耗时 |
| PF3 | 对覆盖集中任一 TensorRT id 一次提交 N 张合法图片（N>1）时，响应不是 N 条逐项结果，或 batch 观测不是 1。单张失败拖垮同批其他成功项，或失败被报成空成功，同样算未满足。本次不要求动态 batch |
| PF4 | 同一次压测读不到 gateway 阶段耗时，也读不到模型 `server_time_ms`。gateway 阶段 p50 大于模型 `server_time_ms` p50 的一半时，报告没有写明时间主要花在哪一段 |
| PF5 | 同一硬件、同一命令的复测中，成功 RPS 低于已记录值的 85%，或 p99 高于已记录值的 150%，且变更说明没有记载原因。尚无已记录值时，以 PF1 的报告不存在为未满足 |
| AV1 | 产品 toml 为 `type=mnn` 或 `type=onnx` 的已服务 id 不能单独拉起，一次合法推理没有成功结果，或失败被报成成功。缺少 HTTP RPS 或 trtexec 对照，不使本行未满足 |

### 稳定

| 编号 | 未满足时的可观察标准 |
| --- | --- |
| ST1 | 未知字段、空图或错误 token 仍得到成功状态，或 `results[].data` 被填成空成功。健康检查除外 |
| ST2 | 鉴权失败仍消耗限流配额，或未鉴权可以调用推理路由。健康检查豁免不在此列 |
| ST3 | 队列打满后，新请求在超时前收不到 429 和 `Retry-After`；或拒绝结束后，后续合法请求不能再成功，且进程没有按重启策略回到 ready |
| ST4 | 停掉驻留集里的一个模型后，gateway 对该 id 不是 503，或驻留集里另一个 id 的合法请求不再成功 |
| ST5 | supervisor 在 `fork` 之后、`exec` 之前调用 `setenv` 或其他非 async-signal-safe 函数 |
| ST6 | `nvidia-smi` 失败或读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率，且日志或状态没有标明 offline |

### 吞吐

| 编号 | 未满足时的可观察标准 |
| --- | --- |
| TP1 | 覆盖集中任一 `type=tensorrt` 的 id，没有一份 static batch=1 的 HTTP 压测：先 calibrate，再按该结论部署，然后 `http_infer_rps.py --raw`，并发数等于已部署的 `worker_nums` 的两倍，持续不少于 30 秒。容量内错误率高于 1%。429 没有单独计数，却被算进这 1%。未 calibrate 就固定并发、不用 `--raw`、并发数不等于已部署 `worker_nums` 的两倍、把 engine 改成动态 batch，或让多个 worker 共享一个 TensorRT context，都不算满足。缺 GPU 却写成已达标，同样算未满足 |
| TP2 | 没有一次压测在 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求，且两者都有成功响应。先停掉其中一个再压另一个，不算满足 |
| TP3 | 驻留集缺 engine、缺占用戳，或占用门闩被关闭时，`mortredctl doctor --strict` 仍退出 0。失败输出里没有下一条命令，同样算未满足 |

### 易用

| 编号 | 未满足时的可观察标准 |
| --- | --- |
| UX1 | 中文和英文 README 的 CPU、GPU 源码构建命令不一致，或与 `CMakePresets.json` 的 `full` / `full-cpu` 不一致 |
| UX2 | README 或 `docs/deployment.md` 给出的首小时路径不能从 `mortredctl next` 走到 `doctor`，或文档中的命令在仓库里不存在 |
| UX3 | CI 不再检查 OpenAPI 与仓库一致，或 GPU 作业被跳过时仍被写成 TensorRT 证明 |
| UX4 | 覆盖集任一 id 没有与 `docs/openapi.json` 一致的请求和响应示例。已删除字段（包括 `img_data`）没有返回 422，或错误没有指出该用哪个字段 |

### 运维

| 编号 | 未满足时的可观察标准 |
| --- | --- |
| OP1 | 换到另一台 GPU 机器时，不能用现成命令为覆盖集的每个 id 生成 engine |
| OP2 | `mortredctl status` 或 supervisor 状态接口不能区分驻留集每个 id 的运行状态，或看不到最近一次退出信息 |
| OP3 | 不能只重启驻留集中的一个 id。重启该 id 必须连带重启 gateway 或其他 id，算未满足 |
| OP4 | 推理 token、管理 token、metrics token 缺失或彼此相同的情况下，gateway 仍然启动 |
| OP5 | 收到停止信号后，模型进程没有先退出，或 gateway 退出后仍留下驻留集的模型进程 |

### 扩展

| 编号 | 未满足时的可观察标准 |
| --- | --- |
| EX1 | `docs/model-developer-guide.md` 或 `docs/how_to_add_new_server.md` 仍要求为每个模型单独做一个服务器二进制，或步骤与当前 `catalog()`、`mortred-model-server.out --list` 不一致 |
| EX2 | 在已有家族的 `catalog()` 增加一个 id 后，`--list` 不出现该 id，或 gateway 需要改路由代码才能转发到它 |
| EX3 | 覆盖集的产品 toml 被改成 `type=onnx` 直接服务。已经是 ONNX 的 id 和 LIBFACE（YuNet）不在此列。cpu profile 上 `type=tensorrt` 初始化成功或进入 ready，同样算未满足 |

## 范围外

下列内容不写成模块问题，不进入实现队列，可以记在附录。完成标准表里的行不在此列。

- 覆盖集以外的 catalog id，包括同一家族的其他尺寸。`yolov8n`、`yolov8l`、`yolov8x`、`metric3d_1088`、`attentive_gan`、`enlightengan` 以及 Depth Anything 的 vit 尺寸，不因覆盖集已有该任务的代表 id 而进入完成标准，也不再要求每个 id 单独做齐旧的 H1 至 H4。
- 旧目标里的个人项目完成线、目录级最高标准，以及为对齐该线而做的逐 id 基准。已关闭的关闭范围不扩大。
- 多机、多节点调度、Kubernetes、服务网格。
- 为对齐 Triton 或多云推理平台而新增的 gRPC、ensemble、模型热加载、模型版本库、官方客户端 SDK。
- 训练平台、标注平台、数据闭环、业务策略。
- Docker 镜像与默认运行用户。本机没有 Docker 测试环境。不开发、不测试、不合入镜像用户改动。
- `msocrnet`。本机内存不允许这么大的模型参与测试，也不把它放进覆盖集。
- SAM、diffusion、CLIP、LightGlue、RTDETR、MOT。它们不是这台机器的驻留或覆盖任务。
- HTTP OCR 的识别器、识别文本字段。`dbnet` 只做检测，检测本身仍在覆盖集里。
- 一个 catalog id 一个 OS 进程。超时 drain 之后析构成员，再由外层 SIGKILL。
- MNN 与 ONNX Runtime 的吞吐、延迟优化。二者只保持服务可用。
- TensorRT 的动态 shape、动态 batch，以及 batch 大于 1 的 engine。本次性能只覆盖 static shape、batch=1。
- 改动「每个 worker 持有各自独立的 TensorRT execution context」。包括多 worker 共享一个 context，以及 server 内的跨 worker context 池。
- 没有交付损失就推翻现有 HTTP 契约或 MNN、ONNX Runtime、TensorRT 后端。
- Windows、macOS，以及其他 CPU 架构。
- `docs/unsupported-boundaries.md` 里的其余不支持项。
- 一次外部验收确认不了、又拆不成有限清单的句子。其中已由确认方案收窄并关闭的，只保留方案里的关闭范围。

## 上限

最终列表里还有未关闭且未阻塞的 P0 时，加权分不得超过 7.9。还有未关闭且未阻塞的 P1、没有这种 P0 时，加权分不得超过 8.9。阻塞项不触发上限。先按权重算出原始分，再套上限。这些上限不代替终态。

## 终态

`done`：完成标准每一行都是「已满足」，九份模块的开放项都是「无」，且阻塞项为空。三件同时成立才算完成。只清空旧队列不算完成。

`awaiting-env`：未阻塞的开放项都是「无」，完成标准里没有「未满足」，但仍有阻塞项。恢复环境后只重跑阻塞项。

停止：本轮有未阻塞项合入或移入范围外，但关闭后未阻塞的开放项一条都没减少，且没有任何完成标准行从「未满足」变成「已满足」。

加权分只作记录。开放项是尚未关闭、且没有移入附录或范围外的条目。阻塞项仍算未关闭，只是不参与本轮的减少判定。
