# 项目评分

- 轮次: 1
- 自算原始分: 8.0
- 自算加权分: 8.0
- 上限规则: P1 上限 8.9
- 相对上轮综合分: 高于 7.6

## 模块
| id | 权重 | 分数 | 记录 |
| --- | --- | --- | --- |
| src-models | 15 | 7.8 | `cowork/ledger/round-001/07-modules/src-models.md` |
| src-server-apps | 15 | 7.7 | `cowork/ledger/round-001/07-modules/src-server-apps.md` |
| src-control | 10 | 8.3 | `cowork/ledger/round-001/07-modules/src-control.md` |
| src-factory-common | 5 | 7.6 | `cowork/ledger/round-001/07-modules/src-factory-common.md` |
| conf | 10 | 8.0 | `cowork/ledger/round-001/07-modules/conf.md` |
| ops | 15 | 8.5 | `cowork/ledger/round-001/07-modules/ops.md` |
| test | 15 | 8.1 | `cowork/ledger/round-001/07-modules/test.md` |
| docs | 10 | 8.5 | `cowork/ledger/round-001/07-modules/docs.md` |
| ci | 5 | 7.2 | `cowork/ledger/round-001/07-modules/ci.md` |

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-1 | 网关门闩合入后，控制面仍有三类易卡交付的残余：空 scrape token 永久拒启、模型波/deadline 失败几乎只打 stderr、无 uri 的 child 不做 readiness | 小团队按 pack 起 supervisor+gateway+多模型；token 配空、某模型 spawn 失败、或无 `uri` 的 child | gateway metrics token 为空仍永久失败 spawn；模型波 autostart/models deadline 主要 `fprintf(stderr)`，未并入与 gateway 同级的结构化 gate；无 `uri` 的 child 可能长期停在 starting 直到 deadline | 配错 token / 模型波失败 / 无 uri child 时，`/api/v1/status` 或 `mortredctl doctor --live` 给出与 gateway `autostart` 同级的明确终态与原因 | `07-modules/src-control.md` 劣势 | 新证据 |
| P1-2 | `mortredctl prepare` 运维入口过薄，换机准备失败时步骤与退出语义全在下游脚本 | 客户 GPU 机：`init` → `prepare` → `doctor` → 起服；prepare 中途失败需定位缺 engine/转换步骤 | `scripts/mortredctl_prepare.sh` 仅 `exec prepare_pack.sh`，失败语义与步骤不在 mortredctl 入口层暴露 | `mortredctl prepare` 失败时入口层非 0 退出并点名缺哪一步/哪份产物 | `07-modules/ops.md` 劣势 | 新证据 |
| P1-3 | 人读契约仍缺抠图 / 增强 / 深度等交付族的完整 `results[].data` 样例 | 业务侧按 `docs/api-contract.md` 写客户端对接 matting / enhance / depth | 分类 / 检测 / 分割 / OCR 检测四类样例已齐；抠图、增强、深度等族的人读样例仍缺 | 必读 docs 内可直接核对上述交付族的 `results[].data` 样例 | `07-modules/docs.md` 劣势 | 新证据 |

## 附录
已关闭六条不重复开单。遗留 `trt_engines.json`、seed 之后的本机校准、install-host 不 start、OCR 代码仍无识别，记为旧问题改写。WORKFLOW 首次锁定、catalog nullptr、golden 默认 SKIP、无 GPU CI、FAQ 游离 `</details>` 记为 P2。

## 未读模块
无
