# 第3步确认

- 问题: P2-12
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: v1 覆盖锁定条目的那一次打开：两份教程里分类命令的 `--model` 写成 `mobilenetv2`，特征点命令的 `--model` 写成 `superpoint`。这两处与 `classification_task.h`、`feature_point_task.h` 的 id 和 family，以及两份 server toml 的 `model=` 和表名一致；`--model` 仍按现有说明大小写敏感，且必须等于 catalog 的 `model_section`。改动只在这两份教程的四个实参。catalog、产品 toml、server toml、一个 catalog id 一个进程、每个 worker 自己的 TensorRT execution context、engine 和官方压测规则都保持原样。覆盖集外的三条启动命令保持现在的大写实参，保持即通过。`worker_nums` 那一句仍归 P2-14，`densenet` 仍归 P2-16，两份教程不出现 `densenet`。四条验收都是打开文件就能判定。自测在 `/mnt/g/Codex/mortred_model_server` 只读文件，通过时退出码为 0。PM 与 RD 已对同一版 v1 同意，条件都是无。
