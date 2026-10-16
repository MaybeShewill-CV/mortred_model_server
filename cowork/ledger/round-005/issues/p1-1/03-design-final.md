# 技术方案

- 问题: P1-1
- 版本: v1

## 目标
FixedWindow vs GCRA dual semantics

## 边界
Operator-visible RateLimit headers + layer tag; keep dual algorithms

## 实现要点
见编排方按用户供给队列执行的步骤 4/5。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | 用户指令 Round 5 队列锁定 |
