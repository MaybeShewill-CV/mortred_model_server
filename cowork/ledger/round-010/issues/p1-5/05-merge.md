# 合入

- 问题: P1-5
- 分支: cowork/r10-p1-5-timeout-clamp
- worktree: /mnt/g/Codex/mortred_wt_r10e
- 问题分支提交: d86e2216d0512944d61ac06ec4382a43fa7b2335
- QA 结论: pass（A2 判据为编排方修订版）
- 重跑: 编排方在 worktree 用带断言的重跑脚本执行 QA 的确定性命令与静态检查，**`SUMMARY FAILS=0`**
- 命令: `git merge --no-ff cowork/r10-p1-5-timeout-clamp -m "Merge branch cowork/r10-p1-5-timeout-clamp (Round 10 P1-5)"`
- merge commit: 7558abdf977619b292149c195f9313fa0a6c43b6

## 编排方重跑记录（脚本自带断言，任一项不符即非零退出）

| 检查 | 退出码 / 输出 |
| --- | --- |
| 提交文件面 | `files=10 forbidden=0`（无 `cowork/ledger/`、构建产物、`mortred_version.h`） |
| `base_server_impl.h` 未改 | `diff_lines=0` |
| `./_bin/server_runtime_config_unittest` | exit 0，`[  PASSED  ] 6 tests.` |
| `./_bin/server_e2e_contract_test` | exit 0，`[  PASSED  ] 47 tests.`；其中 `model_timeout_returns_504` OK（1001 ms）、`model_timeout_zero_is_clamped_to_a_bounded_budget` OK（2003 ms） |
| `grep -rn 'model_run_timeout=0' conf/server \| wc -l` | `0` |
| 显式 `model_run_timeout=300000` 的配置份数 | `5` |
| `grep -c '无限' docs/about_model_server_configuration.zh-cn.md` | `0` |
| 汇总 | `SUMMARY FAILS=0` |

## 合入内容

| 文件 | 变化 |
| --- | --- |
| `src/server/server_runtime_config.h` | 新增 `k_model_run_timeout_clamp_ms = 300000`（注释说明为何**不**用 500ms 默认值）；`<= 0` 分支由"仅告警"改为"告警（含生效值）+ 赋值夹值" |
| 五份 `conf/server/**` | `model_run_timeout=0` → `300000`（各 1 行：`diffusion/{cls_cond_ddim,ddim,ddpm,ldm}`、`segment_anything/sam_amg`） |
| 两份 `docs/about_model_server_configuration*` | 删除"0/-1 表示无限大"，改为"必须 > 0；`<=0` 启动告警并夹到 300000ms" |
| `test/server_runtime_config_unittest.cc` | 三态用例（`0`→300000、`-1`→300000、缺键→500）并钉住常量与 `ewma_seed_ms` 连带值 |
| `test/server_e2e_contract_test.cc` | 新增 `model_timeout_zero_is_clamped_to_a_bounded_budget` 等用例 + 白盒访问器；既有 45 例一字未改（含 P1-7 的对齐结果） |

合计 147 insertions / 10 deletions。改后 `model_run_timeout` 生效值恒 > 0：每个请求都有 deadline 与请求定时器，`_m_waiting_jobs` 必然回落；`base_server_impl.h` 未修改（其 `<=0` 早退分支 `:571-573` 按设计死代码化）。

## 判据修订说明（重要）

方案初版的 A2 判据"`model_run_timeout=0` + `fake_delay_ms=1000` → **504**"在 300000ms 夹值下**算不出来**（1 秒推理早于 5 分钟预算 299 秒结束）。实现者用同机同构建的探针给出三组证据（改前 `applied=0`→200；改后 `applied=300000`→200；对照 `=100`+1s→504），未偷改夹值、未拒绝启动，而是把用例改为钉住夹值实际产生的契约。编排方裁决**接受该替代**（错在方案），并把 A2 改写为四条（正常启动 / 白盒生效值 300000 / 1s 推理仍 200 / 槽位可复用），其中"1s 推理仍 200"正是防"夹到 500ms"回归的护栏。504 路径本身由既有的 `model_timeout_returns_504` 与"定时器必然建立"共同锁定。

## 清理

该 worktree 已于合入后删除（先 `rmdir` 掉 `3rd_party`/`weights` 目录联接再 `git worktree remove --force`）；主仓库依赖树删除前后均为 2515 / 179 文件，未受影响。
