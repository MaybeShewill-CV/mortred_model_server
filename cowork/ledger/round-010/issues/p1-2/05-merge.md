# 合入

- 问题: P1-2
- 分支: cowork/r10-p1-2-cred-scrub
- worktree: /mnt/g/Codex/mortred_wt_r10f
- 问题分支提交: 36ac1d8f0452ffefaf7c12f935f16cfd7e350b5b
- QA 结论: pass
- 重跑: 编排方在 worktree 重跑 QA 的确定性命令与 git 检查，**并用 QA 自己的 live 脚本重跑了一遍真盘**；结果与报告逐项一致
- 命令: `git merge --no-ff cowork/r10-p1-2-cred-scrub -m "Merge branch cowork/r10-p1-2-cred-scrub (Round 10 P1-2)"`
- merge commit: 19eb0cb2816b8fcb6ead1bc9440b07108d952c62

## 编排方重跑记录

| 命令 | 退出码 | 输出 |
| --- | --- | --- |
| `git show --stat --format= 36ac1d8f` | 0 | 3 files changed, 145 insertions(+), 7 deletions(-) |
| `git status --porcelain -- <三文件>` | 0 | 空（工作区 == 提交） |
| `./_bin/supervisor_unittest` | 0 | `[  PASSED  ] 24 tests.` |
| `./_bin/supervisor_malformed_e2e_test` | 0 | `[  PASSED  ] 1 test.` |
| `bash logs/r10_p12_live3.sh`（QA 的真盘脚本，自带端口预检与清理） | 0 | `== A4 mortredctl exit -> 0`、`PASS A4_mortredctl_exit = 0`、supervisor `listening on http://127.0.0.1:8787 (managed servers: 1, auth enabled, expose=loopback)`、gateway `listening on http://127.0.0.1:8080 (routes: 1) [static-token auth, metrics scrape token, expose=loopback]`、`== leftovers: none`、`== SUMMARY FAILS=0` |

合入前 dry-run（`merge-tree --write-tree`）输出 tree `02b86493`，无冲突；合入后基线工作树干净。

## 合入内容

| 文件 | 变化 |
| --- | --- |
| `src/control/supervisor/supervisor_app.cpp` | 删除全局 `::setenv("MORTRED_API_TOKEN", auth_token_, 1)`（冗余：同一值经 `set_gateway_trust` 交给 `_gateway_admin_token`，gateway 分支显式注入）；保留 CORS 的 setenv 并重写注释 |
| `src/control/supervisor.cpp` | 新增 `drop_env_entry()`（按 `key + "="` 精确匹配，避免误删 `MORTRED_API_TOKEN_FILE` 这类前缀相同的键）；`spawn_locked` 在 environ 拷贝之后、`fork()` **之前**删除四个凭据键，随后既有分支注入按角色恢复 |
| `test/supervisor_unittest.cc` | 新增 `SupervisorTest.managed_children_do_not_inherit_management_credentials`：把四键以 `poison-*` 注入 supervisor 自身环境，随后读模型子进程与 gateway 子进程的 `/proc/<pid>/environ` 断言（模型侧四键 0、`MORTRED_AUTH_TOKEN` 1、无 `poison-`；gateway 侧四键各 1、无 `poison-`） |

合计 145 insertions / 7 deletions。净效果：操作者 shell / systemd `EnvironmentFile` 里的管理、scrape、infer 凭据不再随整份 environ 进入任何被管子进程；模型进程不再持有管理 bearer，只保留按设计注入的内部令牌（`MORTRED_AUTH_TOKEN`）与 loopback host。

## 独立证据强度说明

本项的 QA 是本轮最强的一份：A2/A3/A4 **不是单测推断而是真盘**（真 supervisor、真 gateway、真 mortredctl），并且白盒四项各带反证——例如为验证 `drop_env_entry` 的精确匹配，真盘种下 `MORTRED_API_TOKEN_FILE` 与 `MORTRED_METRICS_TOKEN_EXTRA` 并确认二者存活；旁路检查枚举了全 `src/` 的 token 类 `getenv`，确认不存在 `*_TOKEN_FILE`/`*_TOKEN_PATH` 间接通道。编排方随后用同一脚本重跑，得到相同的 `SUMMARY FAILS=0` 与 `leftovers: none`。

## 清理

合入记录写完后删除该 worktree（先移除 `3rd_party`/`weights` 目录联接，再 `git worktree remove --force`），其中未跟踪的 `.qa_p12_*` 夹具目录随目录一并消失。
