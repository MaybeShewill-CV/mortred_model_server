# 第3步签字

- 问题: P1-10
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v2 已把 v1 不同意的条件写进边界和验收 1。区间从 `### Building from source` / `### 从源码构建` 的下一行起，结束于围栏外、去掉前导空白后以 `# ` 开头的下一条标题；围栏内的 `# dependencies`、`# Ubuntu 22.04`、`# 1. 安装 vcpkg` 不结束区间。按此读仓库，两边的下一条都是 Benchmark 标题，英文第一个 bash 围栏和中文路径 B 的 bash 围栏都在区间内。英文该围栏现有 `./scripts/install_deps.sh --all`、`./scripts/install_deps.sh --check` 和 `cmake --preset full && cmake --build --preset full`，没有 `sudo ./scripts/install_deps.sh --nvidia`；中文路径 B 去掉行尾注释后这四条已经逐字都在。只在英文 `--all` 的下一行补上这一条后，两边抽出的集合都等于这四条。行序不参与判定。不在这四条里的行不进入集合，多一行不改变验收 1 的通过与否。两条 `cmake --preset full` 与 `full-cpu` 的整行都还在，`full-cpu` 不进入这四条。`CMakePresets.json` 的 configurePresets 与 buildPresets 已有 `full` 和 `full-cpu`。非 root 的 `--all` 在 `install_deps.sh` 里以退出码 3 结束并指向这条 sudo 命令，所以不删中文行，也不改脚本。自测 `python scripts/check_readme_gpu_lines.py` 通过时期望退出码 0，不一致时期望退出码 1，且不调用安装或 cmake。CPU 行、中文 README、部署节、gateway、supervisor 和 engine 都不改。
