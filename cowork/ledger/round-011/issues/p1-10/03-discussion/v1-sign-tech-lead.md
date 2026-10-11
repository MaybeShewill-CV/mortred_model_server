# 第3步签字

- 问题: P1-10
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: v1 只在英文 `### Building from source` 的第一个 bash 围栏补上 `sudo ./scripts/install_deps.sh --nvidia`。中文路径 B 去掉行尾注释后已经有 `--all`、这条 sudo 命令、`--check` 和 `cmake --preset full && cmake --build --preset full`。两边的 `full-cpu` 整行都还在，且不进入这四条；`CMakePresets.json` 的 configurePresets 与 buildPresets 里已有 `full` 和 `full-cpu`。非 root 的 `--all` 在 `install_deps.sh` 里以退出码 3 结束并指向这条 sudo 命令，所以不删中文行，也不改脚本。验收 1 到 4 按区间和逐字集合可以黑盒判定。自测命令 `python scripts/check_readme_gpu_lines.py` 通过时期望退出码 0，不一致时期望退出码 1，且不调用安装或 cmake。CPU 行、部署节、gateway、supervisor 和 engine 都不改。
