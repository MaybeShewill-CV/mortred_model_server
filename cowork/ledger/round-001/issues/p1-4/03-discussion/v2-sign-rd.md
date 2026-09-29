# 第3步签字

- 问题: P1-4
- 方案版本: v2
- 结论: 不同意
- 条件: A4 自测脚本在 `objs = list(walk(parsed))` 后，`walk` 不产出 list，故 `det_items` 的 `isinstance(o, list)` 分支为死代码；检测/OCR 项只能从带 `"data"` 列表的 dict 收集。按正文「四段独立 JSON（检测数组 / OCR 检测数组）」写顶层 JSON 数组时，`has_det`/`ocr_blocks` 会空失败。须二选一：示例把检测与 OCR 包成含 `"data": [...]` 的对象（或可 walk 到的等价信封），或改脚本使 `loads_all` 的顶层 JSON 数组也进入 `det_items`。
- 理由: v2 已按文件分别判定 A1/A2/A5；A2 核对 `:8080` 行；A1 删除「或等价」且边标注卡死字面 `POST /v1/models/{id}/infer`；A4 用无 `polygon` 防 OCR 冒充检测、OCR 只查键名无 `text`。推荐路径与 OpenAPI 四类字段文档侧可落地。卡在 A4 命令与「顶层独立数组」正文不一致，按现脚本无法同时满足。
