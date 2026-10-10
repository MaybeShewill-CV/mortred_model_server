# 第3步签字

- 问题: P1-10
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的条件已经写进边界和验收 1，「多一条别的 GPU 命令都不通过」已删掉。按 README 做 GPU 源码构建的人，现在能看到英文 `### Building from source` 第一个 bash 围栏没有 `sudo ./scripts/install_deps.sh --nvidia`，中文 `#### 路径 B：full build` 的 bash 围栏有；两边的 `cmake --preset full` 与 `full-cpu` 已在，`CMakePresets.json` 的 configure / build preset 也有这两个名字。区间从这两个标题的下一行起，结束于围栏外、去掉前导空白后以 `# ` 开头的下一条标题；围栏内的 `# dependencies`、`# Ubuntu 22.04`、`# 1. 安装 vcpkg` 以及其他 `# ` 行不结束区间，`#### 路径 A` 与 `#### 路径 B` 也不结束。按这个规则，两份都收到 `# \`Benchmark\`` 之前，四条命令和两条 preset 都落在区间内。验收 1 只比抽出的集合：英文只取第一个 bash 围栏，中文只取路径 B 的 bash 围栏，两边都逐字等于 `./scripts/install_deps.sh --all`、`sudo ./scripts/install_deps.sh --nvidia`、`./scripts/install_deps.sh --check`、`cmake --preset full && cmake --build --preset full`。少一条或 nvidia 行没有 `sudo` 不通过；行序不同仍通过；不在这四条里的行不进入集合，多一行也不改变通过与否。`cmake --preset full-cpu` 整行留在两个区间里，不进这个集合。验收 3、4 和 `python scripts/check_readme_gpu_lines.py` 的退出码可以单独看。中文路径 B 的这四条和已有 preset 名是复用，英文只补那一行。本方案不改 `install_deps.sh`、产品 toml、契约和 `mortredctl doctor`，也不突破 `docs/unsupported-boundaries.md`。
