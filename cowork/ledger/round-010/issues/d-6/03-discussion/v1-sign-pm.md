# 第3步签字

- 问题: D-6
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收第 1–9 条已编号，只读 `README.zh-cn.md` 路径 B 切片、`docs/deployment.zh-cn.md` §7.1/§7.2 与 `CMakePresets.json` 的 `full` / `full-cpu` 就能判定。第 1–2 条核对全文与切片里旧的 `setup_full_deps`、手写 `cmake ..` 和 `-D` 入口已消失。第 3–4 条核对切片内只有一个 bash 代码块，可执行行是 §7.1 四条、`install_deps.sh --cpu --check`、§7.2 两条 `cmake --preset`，且代码块外有「只执行其中一档」等五句。第 5–6 条核对路径 A、导语和 tests-only 段仍在，改动只落在路径 B。第 7–9 条核对现行 preset、英文 README 的 full 入口和 `scripts/setup_full_deps.sh` 仍在。这是复用现有 `install_deps.sh` 与 preset，不改 HTTP 契约、模型目录或 conf，也不突破不支持边界。
