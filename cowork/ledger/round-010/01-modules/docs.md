# 模块评分

- id: docs
- 权重: 10
- 分数: 9.1
- 读过的路径: `cowork/ledger/rubric.md`、`cowork/ledger/round-010/00-rule-change.md`、`docs/api-contract.md`、`docs/unsupported-boundaries.md`、`docs/deployment.md`、`README.md`、`README.zh-cn.md`

## 优势
- `docs/api-contract.md` 写清 gateway `:8080` / supervisor `:8787` 拓扑、`POST /v1/models/{id}/infer` 与 async 前缀、鉴权与状态映射，并给出分类 / 检测 / 分割 / OCR 检测 / 抠图 / 增强 / 深度的 `results[].data` 样例。
- `docs/deployment.md` 覆盖三入口、profile、权重、pack prepare/calibrate、鉴权与 TLS、监控、排障与验收门，小团队可按文档完成首小时上线。
- `docs/unsupported-boundaries.md` 单独列出 OCR 仅检测、RTDETR 脚手架、CUDA 12 / TRT 10、engine 本机重建等边界，避免假预期。
- `README.md` / `README.zh-cn.md` 快速开始与 Model Zoo 已对齐 catalog id 与 gateway 路径，并回链部署与边界文档。

## 劣势
- `README.zh-cn.md`「路径 B：full build」仍写 `./scripts/setup_full_deps.sh` 与 `mkdir build && cmake ..`，与同文件前文及 `docs/deployment.md` / `deployment.zh-cn.md` 的 `install_deps.sh` + `cmake --preset full(-cpu)` 不一致，中文读者按该段源码构建会走偏。
- `docs/deployment.md` §15 FAQ 在 `worker_nums` 条目之后残留无 `<summary>` 的离线拷贝 weights 段落，并多出一个 `</details>`，FAQ 结构损坏（完整离线流程仍在 §9.4）。

## 不计毛病
- HTTP OCR 仅 `dbnet` 文字检测、无识别（`docs/unsupported-boundaries.md`、`docs/api-contract.md` OCR 样例说明）。
- RTDETR 脚手架未进 catalog、无 MOT、bench-only 非 SME HTTP 路径（`docs/unsupported-boundaries.md`、`README.md` Model Zoo）。
- `README.md` Benchmark 仍标注历史环境 CUDA 11.5；产品线以 CUDA 12.x / TRT 10.x 为准（`docs/unsupported-boundaries.md`、Quick Start）。
- SuperPoint / DINOv2 / SAM / diffusion 等人读契约未单独给 `results[].data` 块；完整 schema 指向 `docs/openapi.json`（`docs/api-contract.md`）。

## 依据
- 必读三份与两份 README 交叉核对：网关路径与交付任务样例已在契约中；运维闭环在 deployment；边界页不计入劣势。
- 仍可核验的短板仅见 `README.zh-cn.md` 路径 B 与 `docs/deployment.md` §15 末尾结构损坏两处。

## 未读
无
