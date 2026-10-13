# 第3步确认

- 问题: P2-10
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 4 已编号，只打开 `README.zh-cn.md` 从 `### 第一小时：mortredctl` 到 `### 从源码构建` 之前，并看产品 diff，就能判定。标准 1：这一节的 bash 命令块有一行，去掉行尾 `#` 注释和首尾空白后等于 `mortredctl next`。标准 2：同一节里，这一行之后仍能读到 `mortredctl doctor`；块里现有的 `mortredctl doctor` 与 `mortredctl doctor --strict` 都算。标准 3：这两条不打开 `README.md` 或 `docs/deployment.md`，也不执行 `mortredctl`。标准 4：产品 diff 只有 `README.zh-cn.md`，`README.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`scripts/mortredctl_next.sh` 与改动前相同。现在该中文命令块从 `mortredctl init` 列到 `doctor`，全文没有 `mortredctl next`；补上后，只看这一节就能从 next 走到 doctor。UX2 已由英文 README 的首小时说明与 First-hour 命令块、以及 `docs/deployment.md` 的首小时主航道满足，它们已从 `mortredctl next` 走到 `doctor --strict`。本条不把 UX2 写成未满足，也不改这两份文件。`scripts/mortredctl_next.sh` 最后仍打印 `mortredctl doctor --strict`。本条不改脚本、不启动进程、不生成 engine。
