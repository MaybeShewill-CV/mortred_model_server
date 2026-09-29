# 模块评分

- id: src-models
- 权重: 15
- 分数: 7.8
- 读过的路径: 上轮 `01-modules/src-models.md`；`git diff --name-only 7e501bed HEAD -- src/models` 为空

## 优势
- 统一生命周期与三后端可交付；出错可定位；主力族覆盖检测/分割/抠图/深度/增强/分类/OCR 检测。

## 劣势
- OCR 交付不完整：仅检测，无识别文本字段。
- 接新架构仍是写 C++，不能只改 TOML。
- MOT / diffusion 成熟度不均。

## 依据
- 分数 7.8：沿用上轮。本轮产品 diff 未改 `src/models`。有未读，上限 8.0，上轮分已不高于 8.0。

## 未读
本轮未重读，沿用上轮记录。
