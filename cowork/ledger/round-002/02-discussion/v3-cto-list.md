# 列表修订

- 轮次: 2
- 版本: v3
- 针对: v2

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | 模型波/deadline 失败几乎只打 stderr，未进 status 级门闩 | 小团队运维按 pack 起 supervisor+gateway 时，多模型 pack 中某模型 spawn 失败或 models deadline 超时 | 模型波 autostart/models deadline 主要 `fprintf(stderr)`，未并入与 gateway 同级的结构化 gate | 人为触发模型波失败后，一次 `mortredctl doctor --live` 或读 `/api/v1/status` 即可看到明确终态与原因 | `07-modules/src-control.md` 劣势 |
| P1-2 | 无 uri 的 child 不做 readiness，可能长期停在 starting | 小团队运维按 pack 起服时，pack 中存在无 `uri` 的 child | 无 `uri` 的 child 可能长期停在 starting 直到 deadline | 起含无 uri child 的 pack 后，一次 `mortredctl doctor --live` 或读 `/api/v1/status` 即可看到该 child 的明确终态与原因 | `07-modules/src-control.md` 劣势 |
| P1-3 | `mortredctl prepare` 运维入口过薄，换机准备失败时步骤与退出语义全在下游脚本 | 小团队运维在客户 GPU 机走 `init` → `prepare` → `doctor` → 起服；prepare 中途失败需定位缺 engine/转换步骤 | `scripts/mortredctl_prepare.sh` 仅 `exec prepare_pack.sh`，失败语义与步骤不在 mortredctl 入口层暴露 | `mortredctl prepare` 失败时入口层非 0 退出并点名缺哪一步/哪份产物 | `07-modules/ops.md` 劣势 |
| P1-4 | 人读契约仍缺抠图 / 增强 / 深度等交付族的完整 `results[].data` 样例 | 业务侧对接同学按 `docs/api-contract.md` 写客户端对接 matting / enhance / depth | 分类 / 检测 / 分割 / OCR 检测四类样例已齐；抠图、增强、深度等族的人读样例仍缺 | 必读 docs 内可直接核对上述交付族的 `results[].data` 样例 | `07-modules/docs.md` 劣势 |

## 附录
v2 P1-1（空 scrape token 时 gateway 永久拒启 → doctor/status 可查终态）与已合入关闭的第 1 轮 P0-2 验收重叠，本版删除，不开「空 token 仍启动 gateway」；相对上轮：旧问题改写。已关闭六条不重复开单。遗留 `trt_engines.json`、seed 之后的本机校准、install-host 不 start、OCR 代码仍无识别，记为旧问题改写。WORKFLOW 首次锁定、catalog nullptr、golden 默认 SKIP、无 GPU CI、FAQ 游离 `</details>` 记为 P2。

## 未采纳
无
