# 第3步签字

- 问题: P1-8
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v2 只覆盖冻结的 19 个 catalog id，每个 id 一条 CPU 用例。检测沿用现有框比较，类别不同时失败输出含 `class_id`，IoU 低于 0.5 时含 `low IoU`；分割和抠图的阈值是 0.90，embedding 余弦是 0.999，深度相对误差是 0.02。上一版审查已进入验收 6、7、9、11：类别和 IoU 两次都做并分别改回，分割、抠图、深度写明了黄金路径、UTF-8 JSON 和同尺寸约束，验收 11 只看 `--gtest_list_tests`。缺权重仍是退出码 77 与 `NUMERICAL_NOT_RUN`，`MORTRED_CI_REQUIRE_WEIGHTS=1` 时退出码 1。自测命令可在仓库根目录执行并写明期望退出码。`yolov8s` 的已关闭范围、fingerprint 四条和产品 TensorRT 比较留在原条。
