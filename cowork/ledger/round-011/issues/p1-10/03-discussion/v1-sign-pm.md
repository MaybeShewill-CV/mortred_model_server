# 第3步签字

- 问题: P1-10
- 方案版本: v1
- 结论: 不同意
- 条件: 源码构建区间从 `### Building from source` / `### 从源码构建` 的下一行起，结束于代码围栏之外、去掉前导空白后以 `# ` 开头的下一条标题。围栏内的 `# dependencies`、`# Ubuntu 22.04`、`# 1. 安装 vcpkg` 不结束区间。按此，两份 README 都收到各自的 `# Benchmark` 之前：英文集合计入 `### Building from source` 第一个 bash 围栏，中文集合计入路径 B 的 bash 围栏。验收 1 只比较抽出的集合：两边都逐字等于 `./scripts/install_deps.sh --all`、`sudo ./scripts/install_deps.sh --nvidia`、`./scripts/install_deps.sh --check`、`cmake --preset full && cmake --build --preset full`。少一条，或 nvidia 那一行没有 `sudo`，不通过。行序不同仍通过。不在这四条里的行不进入集合，多一行也不改变通过与否。删掉「多一条别的 GPU 命令都不通过」。
- 理由: 按 README 做 GPU 源码构建的人，现在能看到英文源码构建围栏没有 `sudo ./scripts/install_deps.sh --nvidia`，中文路径 B 有；两边的 `cmake --preset full` 与 `full-cpu` 已在，`CMakePresets.json` 的 configure / build preset 也有这两个名字。验收 3 和自测退出码可以单独看。验收 1、2、4 依赖区间收到「下一条以 `# ` 开头的标题」。英文围栏里的 `# dependencies`、中文路径 A 里的 `# Ubuntu 22.04` 都是以 `# ` 开头的行，而且都在四条 GPU 命令之前。收到这里时两边集合都是空的，补上 nvidia 那一行后仍不等于四条，cmake 行和 `--cpu` 行也落在区间外。把结束理解成围栏外的一级标题时，才收到 `# Benchmark`，改完后集合能对上。同一次对照有两种结果。抽取规则又规定只有逐字等于四条的行进入集合，「多一条别的 GPU 命令都不通过」没有封闭清单；围栏里再多一行其他命令时，集合比较通过，这句话不通过。
