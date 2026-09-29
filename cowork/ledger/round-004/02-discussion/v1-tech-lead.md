# 第2步表决

- 轮次: 4
- 针对版本: v1
- 结论: 待修订

## 逐条
无

## 不同意的条目
建议新开至多 3 条 P1：

1. 同步超时或 `DEADLINE_EXCEEDED_PARTIAL` 时，分段耗时能从响应或既有 metrics 读到，不再依赖 `touch /tmp/mortred_stage_timing`。
2. 深度族补不依赖权重的假 tensor 后处理契约测试，失败能定位到后处理或 shape。
3. `scripts/test_p0_1_engine_fail_closed.sh` 进入默认 `ctest`；无 GPU 时 SKIP 的退出码约定不变。
