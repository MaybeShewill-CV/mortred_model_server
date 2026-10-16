# 技术方案
- 问题: P1-1
- 版本: v1
## 目标
真 NVML GpuProbe（dlopen libnvidia-ml.so.1；失败诚实 fail-closed）+ 将 shared_gpu_probe 接入 sampler / occupancy / supervisor；去掉可去的重复 smi fork。
## 边界
- 不改 packing/Docker。
- 无 GPU / 无 NVML 时 fail-closed 并允许 chain 落到 SmiGpuProbe。
## 实现要点
1. NvmlGpuProbe：dlopen + nvmlInit_v2 / DeviceGet*；失败 available=false backend=nvml。
2. shared_gpu_probe 可测试替换；sampler loop 优先 shared probe，去掉自有 fork 主路径。
3. supervisor occupancy live 快照经 shared probe → LiveGpuSnapshot。
4. 保留 SmiGpuProbe/query_nvidia_smi 作为 chain fallback。
## 确认
| 角色 | 结论 | 方案版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意 | v1 | Round 7 队列锁定 |
