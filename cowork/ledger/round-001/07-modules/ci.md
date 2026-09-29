# 模块评分

- id: ci
- 权重: 5
- 分数: 7.2
- 读过的路径: 上轮 `.github/workflows/ci.yml`；本轮 diff 未触及 `.github`

## 优势
- PR 有 tests、quality-gates、sanitizers；cpu-profile 有 hosted golden。

## 劣势
- 默认 PR 下 GPU/TRT 非 fail-closed。
- clang / clang-tidy 偏可选。
- container-boot 有路径过滤，非容器 diff 可能漏掉 compose 活路。

## 依据
- 分数 7.2：沿用上轮。本轮 diff 未改 `.github`。

## 未读
本轮未重读，沿用上轮记录。
