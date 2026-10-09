# 挂起

- 问题: P1-5
- 方案版本: v2
- 分支: cowork/r10-p1-5-trt-golden
- 工作树: /tmp/mortred-cowork-r10-p1-5
- 问题分支提交: 30ffb4ae8244cb562e681bf8842574d3237f7b99
- QA 结论: blocked
- 缺的环境: WSL2 上 `nvidia-smi` 能看到 RTX 2070 SUPER，没有 `/dev/nvidia*`，`cudaSetDevice` 失败
- 恢复条件: `/dev/nvidia*` 可用，并且 `scripts/run_tests.sh build/full -R '^Yolov8TrtDetection$'` 退出码 0，结果是 Passed
- 未跑的验收: 2、3
- 重跑: 调度者重跑同一命令，退出码 8，与 `05-qa.md` 一致。不合入

环境恢复后只重跑第 5 步。方案 v2 和问题分支保持不变。
