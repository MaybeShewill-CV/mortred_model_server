# 合入

- 问题: P1-6
- 分支: cowork/r10-p1-6-method-label
- worktree: /mnt/g/Codex/mortred_wt_r10c
- 问题分支提交: a5d08259ede3abf8a9b2e8048e9d2e12f8adf98f（分支 tip 另有编排方为修正基线而加的合并提交 `3f1a8a34`）
- QA 结论: pass
- 重跑: 编排方在 worktree 重跑 QA 命令，退出码与报告一致；受测两文件与 `a5d08259` diff 为空
- 命令: `git merge --no-ff cowork/r10-p1-6-method-label -m "Merge branch cowork/r10-p1-6-method-label (Round 10 P1-6)"`
- merge commit: 7453b1e282f05ba9fe36910650eb7a8c46d5097c

## 编排方重跑记录

| 命令 | 退出码 | 输出 |
| --- | --- | --- |
| `./_bin/prometheus_metrics_test` | 0 | `[  PASSED  ] 11 tests.` |
| `./_bin/server_e2e_contract_test` | 0 | `[  PASSED  ] 45 tests.` |
| `git diff --stat a5d08259 HEAD -- <两文件>` | 0 | 空（受测文件 == 提交内容） |
| `git show --name-only --format= a5d08259` | 0 | 恰 `src/server/prometheus_metrics.h`、`test/prometheus_metrics_test.cc` |
| `grep -n normalize_http_method src/server/prometheus_metrics.h` | 0 | `:46` 定义、`:82` counter、`:93` histogram |

## 合入内容

| 文件 | 变化 |
| --- | --- |
| `src/server/prometheus_metrics.h` | 新增 `normalize_http_method`：先拦 `'\|'`/C0 控制字符/`0x7f` → `other`，再 trim + 大写、长度 > 16 → `other`，命中 7 个已知方法则原样返回，其余 → `other`；`inc_http_requests` 与 `observe_http_duration_ms` 两处入口在拼 `method + "\|" + status` 前统一调用 |
| `test/prometheus_metrics_test.cc` | 新增 4 条用例（未知方法折叠、大小写归一、`'\|'` 不注入标签、直方图同归一化）；既有 `method="POST"` 字面断言 0 删除行 |

合计 160 insertions / 2 deletions。`method` 取值集合从"任意请求方法"收敛为 `{GET,POST,PUT,PATCH,DELETE,HEAD,OPTIONS,other}`，并顺带消除用 `A|B` 伪造 `method|status` 标签对的注入面。

## 基线修正说明

该分支原切自 `08d7ba55`（P1-7 合入之前），因此其 worktree 内 `server_e2e_contract_test` 仍有 P1-7 已修复的那两条陈旧用例失败（43/45）。编排方在 QA 前把含 P1-7 的基线并入该分支（`3f1a8a34`），使其 A4 能在正确基线上判定为 45/45；RD 已用 `git stash` + 规范化 diff = 0 行自证那两条失败与本项无关。

## 清理

合入记录写完后删除该 worktree（先移除 `3rd_party`/`weights` 目录联接，再 `git worktree remove --force`，避免递归删除波及主仓库依赖树）。
