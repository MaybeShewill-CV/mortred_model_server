# 第3步签字

- 问题: D-6
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 9 都有编号，靠全文或切片检索、代码块行比对、`git show` / `git diff --name-only`、preset 字段和 `scripts/setup_full_deps.sh` 是否存在即可判定。禁用串现在只在路径 B 切片里，按正文删掉后第 1 条可通过；切片前的导语、路径 A、两处 `cmake -B build -DMORTRED_BUILD_FULL=OFF`，以及紧随切片的 tests-only 代码块与 `HEAD` 逐字相符。§7.1 的四条可执行行和 §7.2 的前两条可以原样抄入，第 5 行按注释约束补上后，切片内只有一个 bash 代码块、恰好 7 行，五句都在代码块外，且不含第 2 条的禁用子串。`CMakePresets.json` 的 `full` / `full-cpu`、`README.md` 的六条命令和 `scripts/setup_full_deps.sh` 已满足第 7 至 9 条。仓库根目录的 bash 能执行这支 `python3` 自测；按契约改完路径 B 后退出码为 0，最后一行是 `ok`。
