# 技术方案

- 问题: P1-1
- 版本: v1
- 对应列表: `/mnt/g/Codex/mortred_model_server/cowork/ledger/round-001/02-p0-p1-final.md`

## 边界

- 只收口 OCR 对外交付口径。必读 docs（含 `docs/unsupported-boundaries.md`）与 README Model Zoo 一致、可黑盒核对：OCR HTTP 仅文字检测（`dbnet`），无识别、无识别文本字段、无 recognizer。
- 只改文档。不新增识别模型，不改 `src/`，不做 P1-2、P1-3、P1-4。
- 不把 `msocrnet` 写成 OCR 识别。

## 做法

### 推荐

在 `docs/unsupported-boundaries.md` 与中文摘要、README Model Zoo（中英）、`docs/deployment.md` 与中文「可用模型」一句写明：OCR HTTP 只有 `dbnet` 文字检测，不支持文字识别。

### 否决

1. 新增识别模型。超出本条。
2. 只改 README、不写 unsupported-boundaries。冻结要求两边一致。

## 改动面

- `docs/unsupported-boundaries.md` 与 `docs/unsupported-boundaries.zh-cn.md`
- `README.md` 与 `README.zh-cn.md` 的 Model Zoo
- `docs/deployment.md` 与 `docs/deployment.zh-cn.md` 的模型能力一句

## 明确不改

`src/`、产品 toml、OpenAPI 路径、pack occupancy、主机安装、`msocrnet` 分类。

## 契约

文档必须能判定 OCR HTTP 仅检测、无识别。运行时契约不改。

## 性能与资源

无进程、显存、engine 变化。

## 验收标准

1. **AC1**：`docs/unsupported-boundaries.md` 写明 HTTP OCR 仅文字检测（`dbnet`），且不支持识别。
2. **AC2**：`docs/unsupported-boundaries.zh-cn.md` 与 AC1 同义。
3. **AC3**：`README.md` Model Zoo 的 OCR 行标明 text detection，并呼应无识别。
4. **AC4**：`README.zh-cn.md` 与 AC3 同义。
5. **AC5**：两份 deployment 的模型列表把 OCR 写成文字检测，不写成可交付文字识别。
6. **AC6**：上述六页不得把 HTTP OCR 写成已支持 text recognition / 文字识别 / recognizer。

## 测试义务

QA 只读文档判定 AC1–AC6。无代码白盒。

## 自测命令

| 目的 | 命令 | 期望退出码 |
| --- | --- | --- |
| 边界页含检测与 dbnet | `rg -n -i 'ocr' docs/unsupported-boundaries.md` | `0` |
| 边界页写明无识别 | `rg -n -i 'recogniz|无识别|不支持' docs/unsupported-boundaries.md` | `0` |
| 中文边界 | `rg -n 'OCR|识别|检测' docs/unsupported-boundaries.zh-cn.md` | `0` |
| 英文 Zoo | `rg -n -i 'text detection|OCR' README.md` | `0` |
| 中文 Zoo | `rg -n '文字检测|OCR' README.zh-cn.md` | `0` |

## RD 不得自行决定

不新增识别模型。不省略 unsupported-boundaries。不扩到其它 P1。不改 HTTP 运行时契约。
