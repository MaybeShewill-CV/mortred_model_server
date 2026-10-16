# 合入

- 问题: P1-4
- 分支: cowork/r10-p1-4-tensor-contract
- worktree: /mnt/g/Codex/mortred_wt_r10d
- 问题分支提交: 73690a0c373d5bf76033066c32be99a256ed835e
- QA 结论: pass
- 重跑: 编排方在 worktree 重跑 QA 点名的 5 个二进制与 A2 的 grep，退出码与通过数与报告逐项一致；两受测文件与提交一致
- 命令: `git merge --no-ff cowork/r10-p1-4-tensor-contract -m "Merge branch cowork/r10-p1-4-tensor-contract (Round 10 P1-4)"`
- merge commit: 19c649fa1ef56fdea1b84a44a497af629bdfa66d

## 编排方重跑记录

| 命令 | 退出码 | 输出 |
| --- | --- | --- |
| `./_bin/tensor_contract_unittest` | 0 | `[  PASSED  ] 14 tests.` |
| `./_bin/http_contract_test` | 0 | `[  PASSED  ] 6 tests.` |
| `./_bin/model_output_contract_unittest` | 0 | `[  PASSED  ] 10 tests.` |
| `./_bin/object_detection_output_contract_unittest` | 0 | `[  PASSED  ] 13 tests.` |
| `./_bin/backend_unittest` | 0 | `[  PASSED  ] 3 tests.`（另有 5 例因权重缺失 SKIPPED） |
| `grep -rn 'CHECK(checked_shape_nbytes\|CHECK_EQ(dtype' src/` | 1 | 无输出 |
| `git show --name-only --format= HEAD` | 0 | 恰 `src/models/backend/tensor.h`、`test/tensor_contract_unittest.cc` |
| `git diff --stat HEAD -- <两文件>` | 0 | 空（工作区 == 提交） |

合入前 dry-run（`merge-tree --write-tree`）输出 tree `a0aac92d`，无冲突；合入后基线工作树干净。

## 合入内容

| 文件 | 变化 |
| --- | --- |
| `src/models/backend/tensor.h` | `Tensor::make` 的 `CHECK(checked_shape_nbytes(...))` 与两个 `data<T>()` 的 `CHECK_EQ(dtype, ...)` 改为 `throw std::invalid_argument(...)`（消息保留原形状文案并新增 requested dtype）；新增 `Tensor` 顶部注释说明"仅因模型层栅栏才以异常表达契约违规、栅栏外未捕获仍 terminate" |
| `test/tensor_contract_unittest.cc` | 新增 3 个用例：坏 shape 抛异常 + 同进程存活、dtype 不匹配抛异常 + 存活、`from_mat` ok-flag 惯例不变；8 处精确类型 `EXPECT_THROW(..., std::invalid_argument)` |

合计 112 insertions / 6 deletions。改后：形状/dtype 契约违规不再 abort 整个模型进程，而是经既有 `invoke_nothrow`（`base_model.h:29-39`）/ `BackendCvModel::run_batch`（`backend_cv_model.h:208-220`）栅栏映射为 `MODEL_RUN_SESSION_FAILED` → HTTP 500；`make` 的 24 个调用点与 `data<T>()` 的 23 个生产调用点签名零改动。

## 记录更正（编排方方案中的两处事实）

1. **调用点计数**：`03-design-final.md` 写 "`data<T>()` 12 个生产调用点"，QA 实测 **23 处**（`.data<T>()` 与 `.template data<T>()` 合计，`src/` 内排除 `tensor.h`，分布在设计点名的同 9 个文件）；`Tensor::make` 的 24 处与设计一致。该更正只加强"否决 log+nullptr 方案"的理由，不改变验收结论。
2. **自测命令路径**：设计里的 `cd /mnt/g/Codex/mortred_wt_r10b` 系复制笔误，本项实际在 `r10d` 执行（QA 已在报告中指出）。

补记 QA 的一条非阻塞发现：`packed_batch_nvi_unittest` 在 `build`/`build-full` 均无法链接（其 `link.txt` 只链 `libcommon.so`、缺 `libmodels.so`，而符号在后者中），因此 packed 批量栅栏只有代码走查、无动态验证；该二进制在本次操作前即不存在，与 P1-4 无关（提交不含任何 CMake 文件）。

## 清理

合入记录写完后删除该 worktree（先移除 `3rd_party`/`weights` 目录联接，再 `git worktree remove --force`，避免递归删除波及主仓库依赖树）。
