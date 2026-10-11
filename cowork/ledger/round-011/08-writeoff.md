# 销账

- 轮次: 11
- 已关闭: P0-3、P1-7、P1-9、P1-10、P1-11
- 挂起: P0-2（工作树缺 39 个产品 onnx，且工作树看不见 trtexec。未跑验收 6、7、8、12）。P1-8（msocrnet 在约 19GiB 的 WSL 上被 OOM 杀掉，没有黄金；工作树没有 full 构建的 model_golden_test。未跑验收 3 的 msocrnet 与 19 条同进程过滤，以及验收 11 的 full 列表）
- 非挂起劣势: 相对本轮起点减少

第 1 步的模块记录和 `02-p0-p1-final.md` 保留作本轮起点，不改写。有效劣势以本文件为准。本次销 P1-11，不重新打分。角色原文在 `issues/p1-11/07-cto.md`。已关闭的 fingerprint 原句不留在留下的劣势里。H2 整行仍未满足。

## 删除的原句
| id | 模块 | 原句 |
| --- | --- | --- |
| P0-3 | src-server-apps | S1：`yolov7` 与 `yolov7x` 在输出张量非空、但既没有 `output`/`518`/`532`，也凑不齐 3 个 `[1,3,H,W,C]` 头时，仍报成空输出。 |
| P1-7 | src-server-apps、docs | H1：范围内每个 HTTP catalog id 都没有一条与 `docs/openapi.json` 一致的请求和响应示例。docs 原句是以下 32 个 id 没有这样的示例。 |
| P1-9 | docs | H3：范围内 39 个 id 都没有一行同时写出硬件、复现命令、指标和日期 |
| P1-10 | docs | C1：中文和英文 README 的 GPU 源码构建命令不一致 |
| P1-11 | test | H2：`bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 只有 fingerprint |

## 留下的劣势
| id | 模块 | 原句 | 状态 |
| --- | --- | --- | --- |
| P0-2 | conf、ops | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:` | 挂起 |
| P1-8 | test | H2：点名的 id 没有 CPU 数值用例 | 挂起 |
| P1-12 | test | H2：除 `yolov8s` 以外、产品 toml 为 `type=tensorrt` 的 id 没有同规则 TRT 比较 | 未做 |
| D-8 | ops | S3：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因 | 未做 |
| D-9 | src-control | S4：推理请求在鉴权失败时仍消耗限流配额 | 未做 |
| D-10 | src-control | S6：`nvidia-smi` 读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率 | 未做 |
| D-11 | docs | `deployment.md` 把 cpu 目录写成 `resnet50`，并把 `msocrnet` 写成 Bench-only | 未做 |
| D-12 | src-control | S2：一次拉起后，检查失败并点名 fork 与 exec 之间的调用 | 未做 |

## 附录
同义句：范围内 id 没有与 `docs/openapi.json` 一致的请求和响应示例。`docs/api-contract.md` 的 7 个人读片段不是 OpenAPI 示例。

同义句：范围内 id 没有一行同时写出硬件、复现命令、指标和日期。`docs/benchmark-set-a.md` 的硬件、日期、指标和命令不在同一行。表 2 各行有指标和 cmd 路径，没有硬件、日期和命令正文。表 4 有 HTTP 指标，没有硬件、命令和日期。

同义句：`yolov7` 与 `yolov7x` 在输出张量非空、头不匹配时仍报成空输出。`yolov7`/`yolov7x` 的非空头不匹配会被看成空输出。已关闭的 SAM 原句不在此条，不重开。

同义句：中文和英文 README 的 GPU 源码构建命令不一致。`README.zh-cn.md` 有 `sudo ./scripts/install_deps.sh --nvidia`，英文源码构建第一个 bash 围栏的 GPU 依赖行只有 `./scripts/install_deps.sh --all` 与 `./scripts/install_deps.sh --check`。CPU 档两边都有，不另开。

同义句：`bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 只有 fingerprint，没有掩码重合阈值。关闭范围只含这四个 id 的 CPU 全分辨率掩码重合。其余 id 的 fingerprint 比较和 TRT 比较不写入本条的关闭。

QA 残余风险不进入队列：类型不符只做了白盒。第一次 `--check` 因未改的 `mobilenetv2_classification.json` 在工作区被写成 CRLF 而退出码 1。该文件的 git 对象没有被这次提交改动；工作区为 LF 时 `--check` 退出码 0。这两句没有新的、一次黑盒可判定的变化。

QA 残余风险不进入队列：检查脚本没有接进 CI，也没有在副本上删掉 `sudo` 行来看退出码 1。方案写明不把脚本接进 CI，验收 5 只要求交付树上 `python scripts/check_readme_gpu_lines.py` 退出码 0。这两句没有新的、一次黑盒可判定的变化。

QA 残余风险不进入队列：验收 7 用有限全零头证明框数为 0 时顶层 `status` 为 0，且 `data` 为 JSON 数组；方案已要求这个结果。三个名字对齐之后的 dtype、通道、buffer 和非有限值失败，方案写明保持契约失败。这两句没有新的一次黑盒可判定的变化。

## 新准入
无
