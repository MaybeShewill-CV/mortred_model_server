# 第3步签字

- 问题: P1-4
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: v3 已按用户选定保留检测/OCR 顶层 JSON 数组，并否决「包成 data 对象」。A4 自测在 `objs = list(walk(parsed))` 之后执行了顶层 list 收集再 `items.extend(det_items(objs))`，顶层检测/OCR 数组可进入 `has_det`/`ocr_blocks`，消除 v2 卡点；其余仍为四份必读文档收口、不改运行时，可落地。
