# 合入

- 问题: P1-3
- 分支: cowork/r10-p1-3-process-group
- worktree: /mnt/g/Codex/mortred_wt_r10g
- 问题分支提交: 9478e8f006139ed07cc25d875fb9853bb7e429a8
- QA 结论: **blocked**（仅因 A2/A3 未以 HTTP 活体形式运行；A1/A4/A4b/A5 与全部白盒项 pass）。**编排方裁决据 A1 的进程级证据与"未改动的既有面"两点合入**，保留意见见 `05-qa.md` 末节
- 重跑: 编排方在 worktree 用带断言的重跑脚本执行 QA 可运行的命令，**`SUMMARY FAILS=0`**
- 命令: `git merge --no-ff cowork/r10-p1-3-process-group -m "Merge branch cowork/r10-p1-3-process-group (Round 10 P1-3)"`
- merge commit: a980ef03037ba924d68a15aaf07fac4d23f7dc29

## 编排方重跑记录（脚本自带断言，任一项不符即非零退出）

| 检查 | 退出码 / 输出 |
| --- | --- |
| 提交文件面 | `files=4 forbidden=0`（`forbidden` 同时排除了 `deploy/`，即单元文件不在提交内） |
| 四文件是否被 QA 改动 | 空（工作区 == 提交） |
| 单元文件（A4b） | `28e365eb 2026-10-13` 未变；`KillMode=process` 恰 1 行 |
| `python3 scripts/stop_order_record.py --check`（A4） | **exit 0**（4 行「为通过」+ 3 行自带负例自检） |
| `./_bin/supervisor_unittest`（A1） | **exit 0，`[  PASSED  ] 26 tests.`**；`stop_signals_the_whole_process_group` OK（716 ms）、`stop_reaps_grandchild_that_ignores_sigint` OK（5723 ms）——与 QA 的 754/5716 ms 一致 |
| 白盒：投递点 | 仅 3 处 `kill*`：`:45 killpg(pgid,0)`、`:86 killpg(pgid,sig)`、`:90` 带 `pid>0` 守卫的回退 |
| 残留进程 | `pgrep -a fake_model_server` 无输出 |
| 汇总 | `SUMMARY FAILS=0` |

合入前 dry-run（`merge-tree --write-tree`）无冲突；合入后基线工作树干净，且 `process_group_gone` 在基线上命中 **8** 处（合入前为 **0**，与我第 64 轮的完整性检查一致——该检查具备判别力）。

## 合入内容

| 文件 | 变化 |
| --- | --- |
| `src/control/supervisor.cpp` | 新增 `process_group_gone(pgid)`、`signal_process_group(pgid, pid, sig)`（唯一的 `pgid<=0` 回退）、`wait_process_group_gone`（有界 20ms 轮询、不持锁）、`reap_group_with_grace`（grace 按**整组**计算、升级后确认整组消失、未消失则写日志）；`fork` 后父侧 `setpgid(pid,pid)`、fd 关闭循环后子侧 `setpgid(0,0)`；`Child::pgid` 记录；**7 处旧单 pid 投递点全部改整组投递** |
| `src/control/supervisor.h` | 声明新原语 + 类注释（进程组 / Ctrl-C 语义）、`Child::pgid`（默认 `-1`，使回退分支真实可达） |
| `test/fake_model_server.cc` | `--spawn-grandchild` / `--grandchild-ignores-sigint` 及配置键；孙进程挂在领队组内且**从不** `setpgid`，打印 `fake grandchild pid=… pgid=… ignore_sigint=…` |
| `test/supervisor_unittest.cc` | `start_fake_with_grandchild`（**先自证拓扑**：领队自成一组的、且不等于测试进程的组、孙进程在领队组内）+ 正例（`stop_ms < 4000` 作为"整组 SIGINT 一次到位"的行为判别）+ 反例（忽略 SIGINT → 走 SIGKILL 升级） |

合计 315 insertions / 35 deletions。语义变化（方案已预期并在本记录与 `05-qa.md` 写明）：子进程不再属于 supervisor 的前台进程组，终端 Ctrl-C 不再直达子进程；models 先 / gateway 后的顺序不变。

## 清理

合入记录写完后删除该 worktree（先移除 `3rd_party`/`weights` 目录联接，再 `git worktree remove --force`，避免递归删除波及主仓库依赖树）。
