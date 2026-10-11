# 第3步签字

- 问题: P1-10
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 英文 `### Building from source` 的第一个 bash 围栏在去掉行尾 ` #` 注释后已有 `./scripts/install_deps.sh --all`、`./scripts/install_deps.sh --check` 和 `cmake --preset full && cmake --build --preset full`，只缺 `sudo ./scripts/install_deps.sh --nvidia`。中文 `#### 路径 B：full build` 下那一个 bash 围栏已逐字含这四条，所以不改 `README.zh-cn.md`。区间结束规则按围栏外、去掉前导空白后以 `# ` 开头的下一行截断；两边实际都停在 `` # `Benchmark` `` 之前。围栏内的 `# dependencies`、`# Ubuntu 22.04`、`# 1. 安装 vcpkg` 以及 `#### 路径 A` / `#### 路径 B` 都不会截断区间。`cmake --preset full-cpu`、两条 `--cpu`、`verify_deployment.sh --basic`、tests-only、路径 A 和部署节一键安装都不进入这四条集合。`CMakePresets.json` 的 configurePresets 与 buildPresets 已有 `full` 和 `full-cpu`。`install_deps.sh` 对非 root 的 `--all` 已以退出码 3 结束并写出下一步 `sudo ./scripts/install_deps.sh --nvidia`，本方案不改该脚本。检查脚本只读两个 README 和 `CMakePresets.json`，不调用安装或构建。推理热路径不动：这一条只对齐文档里抽出的命令集合，现有预处理和后端已经够用。
