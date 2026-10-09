# 第3步签字

- 问题: D-6
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: v1 只改 `README.zh-cn.md` 路径 B 切片：可执行行改为 `install_deps.sh` 与 `cmake --preset full` / `full-cpu`，去掉 `setup_full_deps.sh` 与手写 `cmake ..`，gpu/cpu 仍两档且要求只执行一档，与冻结句「路径 B 与 deployment/preset 一致」相符。验收 1–9 均可黑盒判定，自测命令覆盖这九条，通过退出码 0 且最后一行 `ok`，失败退出码 1。删整节、改回旧入口均已否决。不改路径 A、tests-only、`README.md`、`docs/deployment.zh-cn.md`、`CMakePresets.json` 与推理代码，不重开 D-5。
