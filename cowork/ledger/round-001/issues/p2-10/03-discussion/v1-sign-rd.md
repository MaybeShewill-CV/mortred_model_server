# 第3步确认

- 问题: P2-10
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `README.zh-cn.md` 从 `### 第一小时：mortredctl` 到 `### 从源码构建` 的 bash 块现从 `mortredctl init` 列到 `mortredctl status | catalog`，块内已有 `mortredctl doctor` 与 `mortredctl doctor --strict`，该文件没有 `mortredctl next`。在已有 `mortredctl doctor` 之前只加一行 `mortredctl next` 后，方案第一段自测按原文执行退出码为 0。第二段 `git diff --exit-code` 所查的 `README.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`scripts/mortredctl_next.sh` 当前无差异，退出码为 0；本条不改这四份文件。英文 README 与 `docs/deployment.md` 已从 `mortredctl next` 走到 `doctor --strict`，不必改自测脚本或英文文档。
