# 技术方案

- 问题: P1-2
- 版本: v1

## 目标
gpu_sampler nvidia-smi waitpid harden

## 边界
Reaper/ECHILD / fail-closed like occupancy_gate

## 实现要点
见编排方按用户供给队列执行的步骤 4/5。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | 用户指令 Round 5 队列锁定 |
