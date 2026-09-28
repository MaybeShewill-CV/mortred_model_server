# 项目评分

- 轮次: 1
- 原始综合分: 7.6
- 加权综合分: 7.6
- 上限规则: P0 上限 7.9
- 相对上轮综合分: 无

## 模块
| id | 权重 | 分数 | 记录 |
| --- | --- | --- | --- |
| src-models | 15 | 7.8 | `cowork/ledger/round-001/01-modules/src-models.md` |
| src-server-apps | 15 | 7.7 | `cowork/ledger/round-001/01-modules/src-server-apps.md` |
| src-control | 10 | 7.7 | `cowork/ledger/round-001/01-modules/src-control.md` |
| src-factory-common | 5 | 7.6 | `cowork/ledger/round-001/01-modules/src-factory-common.md` |
| conf | 10 | 6.8 | `cowork/ledger/round-001/01-modules/conf.md` |
| ops | 15 | 7.3 | `cowork/ledger/round-001/01-modules/ops.md` |
| test | 15 | 7.8 | `cowork/ledger/round-001/01-modules/test.md` |
| docs | 10 | 7.8 | `cowork/ledger/round-001/01-modules/docs.md` |
| ci | 5 | 7.2 | `cowork/ledger/round-001/01-modules/ci.md` |

核算：7.8×15 + 7.7×15 + 7.7×10 + 7.6×5 + 6.8×10 + 7.3×15 + 7.8×15 + 7.8×10 + 7.2×5 = 756.0 / 100 = 7.6；有 P0，上限 7.9，加权综合分仍为 7.6。

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-1 | GPU/TRT 产品真源分裂，且 init/doctor 默认不 fail-closed，换机准备易建错/找错 engine，却仍显示「可交付」 | 一两台 GPU 机上客户检测等 TRT 包：`mortredctl init` → `prepare` → `doctor` → supervisor 起 catalog | 同一 id 的 engine 路径在 toml/`product_static_bs1`/`trt_engines.set_a`（`*.static_bs1.fp16`）、`onnx_sources.json`（`fp16in`）、遗留 `trt_engines.json`（`yolov8s.engine`）之间不一致；init 权重失败仍 `init done`，GPU 不强制本机 prepare；doctor 缺 occupancy/engine 默认 WARN，仅 `--strict` 才失败 | 单一真源路径可被 prepare/doctor/起服一致解析；未就绪时默认命令非 0 退出并指出缺哪份 engine/stamp，不再出现「验收绿但 spawn/ready 挂」 | `conf.md` 劣势（engine 真源分裂）；`ops.md` 劣势（init 软失败、doctor fail-open、prepare 入口过薄）；现场经验：小团队易以为 init 完就能上客户 TRT 包 | 无 |
| P0-2 | 网关 scrape token 配空会永久拒启，autostart 失败几乎只打 stderr 且仍继续起 model，整棵入口起不来或无路由时当场难定位 | 小团队按 `mortred.toml` 起 supervisor+gateway+多模型，首次上机或换 token | token 为空则 gateway 永久 spawn 失败；autostart/deadline 失败主要 `fprintf(stderr)`，网关失败后仍拉 model；无 `uri` 的 child 不做 readiness，可能长期 starting | 配错 token/网关未就绪时，supervisor 状态/API/`mortredctl doctor --live` 给出明确永久失败原因与「勿继续起 model」终态，不再只靠翻 stderr | `src-control.md` 劣势（token 永久失败、autostart 可观测性、无 uri 不做 probe） | 无 |

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-1 | 上新客户图/非同质任务仍要写 C++，不能只改 TOML | 每个新导出图，或 OCR 整包、MOT/diffusion 等与单图 YOLO 不同质的交付 | 实现以头+`.inl` 为主；OCR 仅 `db_text_detector`，`io/ocr.h` 无识别文本；MOT/diffusion/多 session 接法不一 | 常见客户变体有可配置的后处理/注册路径；OCR 整包或明确边界文档可交付，少一次「为这一单新开类」 | `src-models.md` 劣势（OCR 缺识别、新架构写 C++、成熟度不均） | 无 |
| P1-2 | 多客户共卡的 pack occupancy/校准未进可用样本，每换机要手工补 | 一两块卡同时伺候多个 catalog，依赖 `gpu_mem_mib` 与 calibrate | 仓库仅 `demo`/`yolov8`/`metric3d` 且多只有 `worker_nums=1`，无校准写出的 `gpu_mem_mib`；occupancy 靠声明+瞬时 nvidia-smi；文档步骤分散 | 代表性 pack 带可复用 occupancy；换机 `calibrate`→写回→doctor/spawn 一条链可复跑 | `conf.md` 劣势（pack 过薄）；`src-control.md` 劣势（occupancy 声明）；`docs.md` 劣势（GPU pack 步骤分散） | 无 |
| P1-3 | 主机 systemd/edge 安装仍注释驱动，换机重复易错 | 不用 compose、走本机 install 的客户机 | `deploy/*` 以注释列出 useradd、`cmake --install`、拷 env、enable 单元等，未收进单一命令 | 一条 `mortredctl`/脚本完成主机安装与单元 enable，换机可复述成功 | `ops.md` 劣势（主机安装手工） | 无 |
| P1-4 | 文档里网关路径与任务响应形态分裂，每次接客户 SDK 易接错 | 业务侧按 docs 写客户端对接 gateway | `deployment.md` 示意 `/mortred_ai_server_v1/...`，`api-contract.md` 主推 `/v1/models/{id}/…`；多任务 `results[].data` 推到 OpenAPI/序列化头 | 单一推荐路径与任务 payload 示例可在必读 docs 内核对，联调不再靠猜入口 | `docs.md` 劣势（路径口径不一致、payload 示例不足） | 无 |

## 附录
| 级别 | 摘要 | 依据 |
| --- | --- | --- |
| P2 | 同进程 `WORKFLOW_library_init` 只认首次线程池；挂死时 drain 无界，靠外层 SIGKILL | `src-server-apps.md` |
| P2 | Catalog/`make_model` 失败多返回 `nullptr`+LOG，未进统一 `StatusCode`；信封 decode 对半残顶层偏松 | `src-factory-common.md` |
| P2 | Golden/多数 output-contract 默认依赖 full+权重，TRT 无同级数值回归；旧 `contract_baseline` 易混淆 | `test.md` |
| P2 | 无自托管 GPU runner 时 `gpu-pr-gate` skip，合入前拦不住真实 TRT 回归；部分门禁偏手动/schedule | `ci.md` |
| P2 | `deployment.md` FAQ 游离 `</details>` 结构瑕疵 | `docs.md` |

## 未读模块
无
