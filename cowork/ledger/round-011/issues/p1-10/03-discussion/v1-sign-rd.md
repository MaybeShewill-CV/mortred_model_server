# 第3步签字

- 问题: P1-10
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 英文 `### Building from source` 的第一个 bash 围栏现在有 `./scripts/install_deps.sh --all`、`./scripts/install_deps.sh --check` 和 `cmake --preset full && cmake --build --preset full`，没有 `sudo ./scripts/install_deps.sh --nvidia`。中文路径 B 在去掉 ` #` 行尾注释后，这四条已经逐字都在。只在英文 `--all` 的下一行补上这一条，两边的 GPU 可执行行集合都等于边界列出的四条；中文顺序不同，按方案不参与判定。区间结束于围栏外的下一条一级标题，两边都是 `# Benchmark`。围栏里的 `# dependencies`、`# Ubuntu 22.04` 是注释，不是标题。这样路径 B、两条 `--cpu` 行、`./scripts/verify_deployment.sh --basic` 和两条完整 cmake 行都留在区间内；路径 A、`tests-only` 对不上那四条，`# Deployment` / `# 部署说明` 的一键安装围栏在区间之外。`cmake --preset full-cpu && cmake --build --preset full-cpu` 用整行相等判断，不会因为带有 `cmake --preset full` 前缀进入 GPU 集合；这两行已经在两边，方案也不改它们。`CMakePresets.json` 的 `configurePresets` 与 `buildPresets` 里，`name` 已经分别有 `full` 和 `full-cpu`。`install_deps.sh` 里非 root 的 `--all` 在 nvidia 未装时以退出码 3 结束，并写出下一步就是 `sudo ./scripts/install_deps.sh --nvidia`，英文注释可以按这个事实来写。检查脚本只读两份 README 和 `CMakePresets.json`，不改依赖脚本、不跑构建，也不动推理热路径。
