# 合入

- 问题: P1-7
- 分支: cowork/r10-p1-7-stale-tests
- worktree: /mnt/g/Codex/mortred_wt_r10
- 问题分支提交: 6104d57ae980be3211b69d513900eea64d92c0f8
- QA 结论: pass
- 重跑: 编排方在 worktree 重跑 QA 三条命令，退出码与报告一致；四文件工作区与提交一致（`git status --porcelain -- <四文件>` 为空）
- 命令: `git merge --no-ff cowork/r10-p1-7-stale-tests -m "Merge branch cowork/r10-p1-7-stale-tests (Round 10 P1-7)"`
- merge commit: d4419a944beb267e44adeeaf70c06e1c18215d95

## 编排方重跑记录

| 命令 | 退出码 | 输出 |
| --- | --- | --- |
| `./_bin/request_envelope_unittest` | 0 | `[  PASSED  ] 17 tests.` |
| `./_bin/server_e2e_contract_test` | 0 | `[  PASSED  ] 45 tests.` |
| `./_bin/gateway_e2e_test` | 0 | `[  PASSED  ] 49 tests.` |
| `grep -n "rate_limit_qps" src/server/server_config_schema.h` | 0 | `81:` 注释行 + `90:` 键表行 |

重跑与 QA 报告逐项一致（命令、退出码、通过数），worktree 内四个受测文件与提交 `6104d57a` 完全一致。合入后基线工作树干净。

## 合入内容

| 文件 | 变化 |
| --- | --- |
| `test/request_envelope_unittest.cc` | 期望改为解码后字节；`{"images":["ok",5]}` 改为 2 条违规的 (pointer, message) 集合断言 |
| `test/server_e2e_contract_test.cc` | 两条陈旧用例改名并对齐产品行为（`rate_limit_qps` 对模型侧准入无效；`/healthz`+`/ready` 两种配置下免鉴权 200） |
| `test/gateway_e2e_test.cc` | 三个非 loopback 正例加 `MORTRED_RATE_LIMIT=enforce`；三个负例加 `MORTRED_RATE_LIMIT_ALLOW_DISABLE=1` 以抵达各自守卫 |
| `src/server/server_config_schema.h` | 仅注释：`rate_limit_qps` 不再参与模型侧准入，gateway GCRA 为唯一限流权威 |

合计 83 insertions / 24 deletions，产品逻辑零改动。

## 清理

合入记录写完后删除该 worktree 与其残留 scratch（`_p17_*.sh`、`_p17_logs/`、`build*`）。删除前先移除 `3rd_party`/`weights` 两个目录联接，避免递归删除波及主仓库依赖树。
