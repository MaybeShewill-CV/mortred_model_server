# 技术方案
- 问题: P0-1
- 版本: v1
## 目标
CI 将 `check_version_single_source.py` 作为独立 fail-closed 步骤运行，不再作为 `check_consistency.py` 的位置参数。
## 边界
- 不改 packing/Docker。
- 不改 check_consistency 行为（仅去掉错误调用）。
## 实现要点
1. `.github/workflows/ci.yml` Repository consistency checks：先 `python3 scripts/check_version_single_source.py`，再 `python3 scripts/check_consistency.py`（无多余位置参数）。
2. 版本检查失败必须使该 step 失败（set -e已有）。
## 确认
| 角色 | 结论 | 方案版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意 | v1 | Round 7 队列锁定 |
