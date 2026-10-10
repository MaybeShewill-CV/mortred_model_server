# 技术方案

- 问题: P1-10
- 版本: v1
- 对应列表: v3

## 边界

轮次 11。上一版审查意见无。本方案只覆盖冻结句：中文和英文 README 的 GPU 源码构建命令不一致。改完后，两份 README 源码构建代码块的 GPU 可执行行集合相同，并且仍包含 `cmake --preset full` 与 `cmake --preset full-cpu`。

现在 `README.zh-cn.md` 路径 B 的代码块里有 `sudo ./scripts/install_deps.sh --nvidia`。`README.md` 的 `### Building from source` 代码块里，GPU 依赖行是 `./scripts/install_deps.sh --all` 与 `./scripts/install_deps.sh --check`。两边的 `cmake --preset full` 与 `cmake --preset full-cpu` 已与 `CMakePresets.json` 的 configure / build preset 同名。附录已把「中文路径 B 改用 `install_deps.sh` 和这两条 preset」写成旧问题改写，本方案不重开。

源码构建区间：`README.md` 从 `### Building from source` 的下一行起，到下一条以 `# ` 开头的标题为止；`README.zh-cn.md` 从 `### 从源码构建` 的下一行起，到下一条以 `# ` 开头的标题为止。GPU 可执行行只从这两个区间的 `bash` 围栏里抽取。行尾 `\r` 去掉，从 ` #` 起的行尾注释去掉，再去掉首尾空白。空行和以 `#` 开头的行丢掉。与下面四条逐字相同的行进入集合，重复只计一次：

- `./scripts/install_deps.sh --all`
- `sudo ./scripts/install_deps.sh --nvidia`
- `./scripts/install_deps.sh --check`
- `cmake --preset full && cmake --build --preset full`

`cmake --preset full-cpu && cmake --build --preset full-cpu` 含有前缀 `cmake --preset full`，它不进入这个集合，但两个区间都必须保留这一整行。顺序不参与判定。`--cpu`、`--headers`、`cmake --preset tests-only`、`./scripts/verify_deployment.sh --basic`、路径 A 的 `apt-get` / `cmake -B build`，以及 `# Deployment` / `# 部署说明` 里的一键安装围栏，都不是这个集合的成员。

不覆盖 P0-2、P1-8、P1-11、P1-12、D-8 至 D-12。不重开已关闭原句。不改 CPU 构建行，不改部署文档，不改推理热路径。

## 做法

### 推荐

只在 `README.md` 的 `### Building from source` 第一个 bash 围栏里，紧接 `./scripts/install_deps.sh --all` 那一行之后，增加一行 `sudo ./scripts/install_deps.sh --nvidia`。注释用英文，可写明这一步需要 root，以及非 root 的 `--all` 会以退出码 3 结束、要等这一行才装上 CUDA / TensorRT / cuDNN。命令本身不加别的参数。

中文路径 B 里这四条已经齐，所以不改 `README.zh-cn.md`。加上这一行之后，两边的 GPU 可执行行集合都是上面四条。英文围栏里原有的 `--cpu` 行、`cmake --preset full-cpu` 行和 `./scripts/verify_deployment.sh --basic` 留在原处。

另加一个只读脚本 `scripts/check_readme_gpu_lines.py`，按「验收标准」抽取并给出退出码。它不调用 `install_deps.sh`、`cmake` 或 Docker。

代价：英文源码构建块多一行需要 root 的命令。已经用 root 跑过 `--all` 的机器会再进入一次 `--nvidia`；`install_deps.sh` 的 stamp 还在且文件还在时会跳过复制，这一次仍要 root。非 root 的 `--all` 仍以退出码 3 结束。本方案不把围栏包进 `set -e`，也不改脚本的退出码。检查脚本只读两个 README 和 `CMakePresets.json`。

### 否决

从 `README.zh-cn.md` 删掉 `sudo ./scripts/install_deps.sh --nvidia`，让两边只剩 `--all` 与 `--check`。代价是少一行。非 root 执行 `--all` 时，脚本在非 root 依赖装完后以退出码 3 结束，并写出下一步就是 `sudo ./scripts/install_deps.sh --nvidia`，然后才是 `--check`。删掉这一行后，按任一份 README 的源码构建代码块都停在未装上的 CUDA / TensorRT / cuDNN，`--check` 仍会报这些项缺失。

把两边都改成只跑 `sudo ./scripts/install_deps.sh --all`，或改 `install_deps.sh` 让非 root 的 `--all` 直接装 nvidia，并重排中文代码块、改 CPU 行、路径 A 或部署节的一键安装围栏。代价是把现在不需要 root 的 MNN、WORKFLOW、fmt 编译并进 root 步骤，或去改冻结句没有点名的文件。`--nvidia` 才是脚本里要求 root 的那一步。preset 名已经和 `CMakePresets.json` 一致。这些改动不产生「两边 GPU 可执行行集合相同」这一条可观察变化所需要的那一处文档对齐。

## 改动面

- `README.md`：在 `### Building from source` 的第一个 bash 围栏中，`./scripts/install_deps.sh --all` 的下一行增加 `sudo ./scripts/install_deps.sh --nvidia`。
- `scripts/check_readme_gpu_lines.py`：只读检查。实现验收标准里的区间、注释剥离、四条命令集合、两条 preset 命令、`CMakePresets.json` 的 preset 名，以及英文区间里 `./scripts/verify_deployment.sh --basic` 与两条 `--cpu` 命令仍在。失败时退出码 1，标准错误至少一行点名不符的集合或命令。通过时退出码 0。

## 明确不改

- `README.zh-cn.md` 全文，含路径 B 的现有顺序、路径 A、`cmake --preset tests-only` 围栏，以及「gpu 档使用 --all、--nvidia、--check」这段散文。
- 两个 README 里所有含 `--cpu` 的行，含 `./scripts/install_deps.sh --cpu --all` 与 `./scripts/install_deps.sh --cpu --check`。
- 两条已有的 cmake 行：`cmake --preset full && cmake --build --preset full` 与 `cmake --preset full-cpu && cmake --build --preset full-cpu`。不拆开，不改 preset 名。
- 英文围栏里的 `./scripts/verify_deployment.sh --basic`，以及英文 tests-only 围栏的 `./scripts/install_deps.sh --headers` 与 `cmake --preset tests-only`。
- `# Deployment` 与 `# 部署说明` 的一键安装围栏、Docker 示例、TensorRT 引擎重建段落。两边引擎段落里已有的 `sudo ./scripts/install_deps.sh --nvidia` 散文保持原样。
- `scripts/install_deps.sh`、`CMakePresets.json`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`docs/unsupported-boundaries.md`。
- 产品 toml、HTTP 契约、tensor 契约、gateway（:8080）、supervisor（:8787）、engine 生成、`mortredctl prepare`。cpu profile 继续不链 TensorRT。gpu profile 继续是 CUDA 12.x + TensorRT 10.x。

## 契约

HTTP 契约和 tensor 契约不变。产品 toml 的 `type` 不变。本条变的是源码构建围栏里的 GPU 命令集合：两边都包含且只包含这四条 GPU 可执行行：

- `./scripts/install_deps.sh --all`
- `sudo ./scripts/install_deps.sh --nvidia`
- `./scripts/install_deps.sh --check`
- `cmake --preset full && cmake --build --preset full`

`cmake --preset full-cpu && cmake --build --preset full-cpu` 仍留在两个源码构建区间里，不记入 GPU 集合。

## 性能与资源

不拉起 gateway（:8080）和 supervisor（:8787），不生成 engine，不占显存。检查脚本只读文件。本方案不执行 `install_deps.sh`。文档中的 `--nvidia` 仍是现有脚本里安装 CUDA / TensorRT / cuDNN 的 root 步骤；engine 仍在本机本卡用 `mortredctl prepare` 生成。

## 验收标准

1. 按「边界」的区间和注释剥离规则，两份 README 源码构建区间抽出的 GPU 可执行行集合都等于边界列出的四条，逐字相同。少一条、多一条别的 GPU 命令，或把 `sudo` 去掉，都不通过。两边行序不同仍通过。
2. 两个区间都能找到去掉注释后完全等于 `cmake --preset full && cmake --build --preset full` 的行，以及完全等于 `cmake --preset full-cpu && cmake --build --preset full-cpu` 的行。后一行不计入验收 1 的集合。
3. `CMakePresets.json` 的 `configurePresets` 与 `buildPresets` 里都有 `name` 为 `full` 和 `full-cpu` 的项。本条不修改该文件。
4. 英文源码构建区间仍有去掉注释后等于 `./scripts/verify_deployment.sh --basic` 的行。两个区间仍有去掉注释后等于 `./scripts/install_deps.sh --cpu --all` 与 `./scripts/install_deps.sh --cpu --check` 的行。
5. 下面的自测命令退出码为 0。

## 测试义务

QA 按边界的规则读两个源码构建区间的 bash 围栏，核对验收 1 到 4。重点看英文新增行是 `sudo ./scripts/install_deps.sh --nvidia`，中文路径 B 原有的这一行还在，两条 `cmake --preset full` 与 `full-cpu` 都还在。`--headers`、`tests-only`、路径 A 和一键安装围栏不进 GPU 集合。

白盒看 diff：`README.zh-cn.md`、`scripts/install_deps.sh`、`CMakePresets.json`、`docs/deployment.md`、`docs/deployment.zh-cn.md` 无改动。`README.md` 只在 `### Building from source` 的第一个 bash 围栏增加那一行，含 `--cpu` 的行、两条 cmake 行和 `verify_deployment.sh --basic` 的原文还在。`scripts/check_readme_gpu_lines.py` 不执行安装、配置或构建。交付树上的自测命令退出码为 0。不要求装 CUDA，不要求跑推理。

## 自测命令

在仓库根目录执行。本机入口是 `python`。

`python scripts/check_readme_gpu_lines.py`

验收 1 到 4 都成立时，期望退出码 0。GPU 集合与四条不一致、任一条 preset 命令缺失、`CMakePresets.json` 的 configure 或 build preset 缺少 `full` 或 `full-cpu`、英文区间缺少 `./scripts/verify_deployment.sh --basic`，或任一条 `--cpu` 命令缺失时，期望退出码 1。脚本把 `\r\n` 当成普通换行处理。它不调用 `install_deps.sh` 和 `cmake`。

## RD 不得自行决定

- 不得删中文的 `sudo ./scripts/install_deps.sh --nvidia` 来凑集合，不得去掉 `sudo`，不得把 `--all`、`--nvidia`、`--check` 合成一行，不得加 `--cpu` 或其他参数。
- 不得改 `scripts/install_deps.sh` 的退出码 3，不得给源码构建围栏加 `set -e` 或 `|| true`。
- 不得改写、拆开或移动两条 `cmake --preset full` / `full-cpu` 命令，不得改 `CMakePresets.json`。
- 不得改任何 `--cpu` 行、路径 A、tests-only 围栏、`verify_deployment.sh --basic`、一键安装围栏、Docker 示例或引擎重建段落。
- 不得改产品 toml、gateway、supervisor、推理热路径、engine 生成或 `docs/deployment.md` 来凑这一条。
- 新行的注释若写版本，只写邻近 `--all` 行已有的 CUDA 12 / TensorRT 10.x。不得写成 CUDA 11 或 TensorRT 8。
