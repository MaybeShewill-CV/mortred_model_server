# 挂起

- 问题: P1-8
- 方案版本: v2
- 分支: cowork/r11-p1-8-cpu-golden
- 工作树: G:\Codex\mortred-cowork-r11-p1-8
- 问题分支提交: aedd09c2dc3c1ffdab5d2fca3850075eca687e15
- QA 结论: blocked
- 缺的环境: 用户于 2026-10-11 决定不再在这台 WSL 上测试 `msocrnet`，因为机器内存不够。本机物理内存约 23.9GB，WSL 可见约 19GiB。上次推理退出码 137，`test/golden/msocrnet.json` 没有生成。`./build/full/test/model_golden_test` 已经建成，full 列表已通过。
- 恢复条件: 在能容下这次推理内存的机器上，由 CPU onnx 生成并提交 `test/golden/msocrnet.json`，再单进程跑 `model_golden.msocrnet`，退出码 0 且标准输出含 `NUMERICAL_PASSED`。不要在这台 WSL 上重跑。full 列表已经通过，不需要为验收 11 再编译。
- 未跑的验收: 3 的 msocrnet，以及 19 条放进同一进程的过滤。
- 重跑: 未合入。调度者重跑 `model_golden.depth_vitl14` 与 `model_golden.metric3d_512`，输出都有一行 `NUMERICAL_PASSED`，且各有 `[  PASSED  ] 1 test.`。2026-10-11 在同一工作树、可见内存约 19GiB、交换区 8GiB 上再跑 `MORTRED_UPDATE_GOLDEN=1 ./_bin/model_golden_test --gtest_filter=model_golden.msocrnet`。ORT 已打印 `x:f32[1,3,1024,2048]`。进程驻留约 18.5GiB、交换约 4.6GiB 后被杀掉，退出码 137。`test/golden/msocrnet.json` 没有生成。同日 `cmake --preset full` 建成 `./build/full/test/model_golden_test`，`--gtest_list_tests` 退出码 0，标准错误为空。套件只有 `model_golden.` 与 `Yolov8TrtDetection.`。这 19 个名字都在 `model_golden.` 下。同日把 WSL 交换区临时改成 32GB 后又按用户决定停掉这次重跑，交换区已改回 8GB。验收 3 的 msocrnet 与 19 条同进程过滤仍未跑，且不再在这台 WSL 上重跑。
