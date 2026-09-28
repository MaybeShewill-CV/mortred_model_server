# P0 与 P1

- 轮次: 1
- 版本: v2
- 综合分上限: P0 上限 7.9

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | GPU/TRT 产品真源分裂，且 init/doctor 默认不 fail-closed，换机准备易建错/找错 engine，却仍显示「可交付」 | 一两台 GPU 机上客户检测等 TRT 包：`mortredctl init` → `prepare` → `doctor` → supervisor 起 catalog | 同一 id 的 engine 路径在 toml/`product_static_bs1`/`trt_engines.set_a`（`*.static_bs1.fp16`）、`onnx_sources.json`（`fp16in`）、遗留 `trt_engines.json`（`yolov8s.engine`）之间不一致；init 权重失败仍 `init done`，GPU 不强制本机 prepare；doctor 缺 occupancy/engine 默认 WARN，仅 `--strict` 才失败 | 单一真源路径可被 prepare/doctor/起服一致解析；未就绪时默认命令非 0 退出并指出缺哪份 engine/stamp，不再出现「验收绿但 spawn/ready 挂」 | `conf.md` 劣势（engine 真源分裂）；`ops.md` 劣势（init 软失败、doctor fail-open、prepare 入口过薄）；现场经验：小团队易以为 init 完就能上客户 TRT 包 |
| P0-2 | 网关 scrape token 配空会永久拒启，autostart 失败几乎只打 stderr 且仍继续起 model，整棵入口起不来或无路由时当场难定位 | 小团队按 `mortred.toml` 起 supervisor+gateway+多模型，首次上机或换 token | token 为空则 gateway 永久 spawn 失败；autostart/deadline 失败主要 `fprintf(stderr)`，网关失败后仍拉 model；无 `uri` 的 child 不做 readiness，可能长期 starting | 配错 token/网关未就绪时，supervisor 状态/API/`mortredctl doctor --live` 给出明确永久失败原因与「勿继续起 model」终态，不再只靠翻 stderr | `src-control.md` 劣势（token 永久失败、autostart 可观测性、无 uri 不做 probe） |

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | OCR 对外交付口径不清：仓库只有检测、没有识别，客户易当整包 OCR 接单 | 业务侧按 Zoo/docs 接 OCR HTTP，交付「文字识别」类客户单 | HTTP/Zoo 侧 OCR 实质只有 `db_text_detector`（检测）；`src/models/io/ocr.h` 仅 `text_region`（框/多边形/分），无识别文本字段、无 recognizer；`unsupported-boundaries` 未写死「OCR 无识别」 | 必读 docs（含 `unsupported-boundaries`）与 README Model Zoo 一致、可黑盒核对：OCR HTTP 仅检测、无识别；不再出现「文档写 OCR、实际无文本输出」的歧义接单 | `src-models.md` 劣势（OCR 缺识别）；`docs/unsupported-boundaries.md`（未列 OCR 识别边界）；PM/技术负责人：P1-1 拆开后仅保留可黑盒 OCR 条 |
| P1-2 | 多客户共卡的 pack occupancy/校准未进可用样本，每换机要手工补 | 一两块卡同时伺候多个 catalog，依赖 `gpu_mem_mib` 与 calibrate | 仓库仅 `demo`/`yolov8`/`metric3d` 且多只有 `worker_nums=1`，无校准写出的 `gpu_mem_mib`；occupancy 靠声明+瞬时 nvidia-smi；文档步骤分散 | 代表性 pack 带可复用 occupancy；换机 `calibrate`→写回→doctor/spawn 一条链可复跑 | `conf.md` 劣势（pack 过薄）；`src-control.md` 劣势（occupancy 声明）；`docs.md` 劣势（GPU pack 步骤分散） |
| P1-3 | 主机 systemd/edge 安装仍注释驱动，换机重复易错 | 不用 compose、走本机 install 的客户机 | `deploy/*` 以注释列出 useradd、`cmake --install`、拷 env、enable 单元等，未收进单一命令 | 一条 `mortredctl`/脚本完成主机安装与单元 enable，换机可复述成功 | `ops.md` 劣势（主机安装手工） |
| P1-4 | 文档里网关路径与任务响应形态分裂，每次接客户 SDK 易接错 | 业务侧按 docs 写客户端对接 gateway | `deployment.md` 示意 `/mortred_ai_server_v1/...`，`api-contract.md` 主推 `/v1/models/{id}/…`；多任务 `results[].data` 推到 OpenAPI/序列化头 | 单一推荐路径与任务 payload 示例可在必读 docs 内核对，联调不再靠猜入口 | `docs.md` 劣势（路径口径不一致、payload 示例不足） |

## 附录
| 级别 | 摘要 | 依据 |
| --- | --- | --- |
| P2 | 同进程 `WORKFLOW_library_init` 只认首次线程池；挂死时 drain 无界，靠外层 SIGKILL | `src-server-apps.md` |
| P2 | Catalog/`make_model` 失败多返回 `nullptr`+LOG，未进统一 `StatusCode`；信封 decode 对半残顶层偏松 | `src-factory-common.md` |
| P2 | Golden/多数 output-contract 默认依赖 full+权重，TRT 无同级数值回归；旧 `contract_baseline` 易混淆 | `test.md` |
| P2 | 无自托管 GPU runner 时 `gpu-pr-gate` skip，合入前拦不住真实 TRT 回归；部分门禁偏手动/schedule | `ci.md` |
| P2 | `deployment.md` FAQ 游离 `</details>` 结构瑕疵 | `docs.md` |

## 未采纳
| id | 来源表决 | 原因 |
| --- | --- | --- |
| P1-1（只改 TOML 上任意新导出图/常见变体） | PM、技术负责人：P1-1 拆开 | 「只改 TOML 上任意新模型」无具体客户单、黑盒无法判定「常见/任意」；不可验收，不保留成条 |
| P1-1（MOT HTTP 交付） | PM、技术负责人：P1-1 拆开 | `docs/unsupported-boundaries.md` / README 已写无 MOT HTTP；勿编进本仓库需求 |
| P1-1（diffusion / 多 session「成熟度」当交付项、任意新架构零 C++） | 技术负责人：P1-1 拆开剔超边界 | 与单图 YOLO 不同质接法属成熟度描述；「任意新架构零 C++」单轮无交付闭环，且易推翻现有模型接入方式；不进 P0/P1 队列 |

## 签字
| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| cv-cto | 同意 | v2 | `cowork/ledger/round-001/02-discussion/v2-sign-cto.md` |
| cv-pm | 同意 | v2 | `cowork/ledger/round-001/02-discussion/v2-sign-pm.md` |
| cv-tech-lead | 同意 | v2 | `cowork/ledger/round-001/02-discussion/v2-sign-tech-lead.md` |
