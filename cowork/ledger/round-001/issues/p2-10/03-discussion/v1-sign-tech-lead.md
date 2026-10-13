# 第3步确认

- 问题: P2-10
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 4 已编号，且都能在外部判定。标准 1：`README.zh-cn.md` 从 `### 第一小时：mortredctl` 到 `### 从源码构建` 之前的 bash 块，有一行去掉行尾 `#` 注释和首尾空白后等于 `mortredctl next`。标准 2：同一节里这一行之后仍能读到 `mortredctl doctor`，块内现有的 `mortredctl doctor` 与 `mortredctl doctor --strict` 都算。标准 3：这两条只打开这一节即可判定，不打开 `README.md` 或 `docs/deployment.md`，也不执行 `mortredctl`。标准 4：产品 diff 只有 `README.zh-cn.md`，`README.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`scripts/mortredctl_next.sh` 与改动前相同。对照现状，该中文块从 `mortredctl init` 列到 `status | catalog`，全文没有 `mortredctl next`；在已有 `mortredctl doctor` 之前加这一行，就满足锁定句「命令块出现 `mortredctl next`，并且同一节能走到 `mortredctl doctor`」。UX2 已由 `README.md` 的首小时说明与 First-hour 命令块、以及 `docs/deployment.md` 的首小时主航道满足，二者都从 `mortredctl next` 走到 `doctor --strict`；`scripts/mortredctl_next.sh` 最后一条仍是 `mortredctl doctor --strict`。本条不改这三处，不改 `docs/deployment.zh-cn.md`，不启动进程，不生成 engine，不改 TensorRT execution context，也不扩到 P2-11。两条自测都在 `/mnt/g/Codex/mortred_model_server`，期望退出码都是 0。上一版审查意见无。
