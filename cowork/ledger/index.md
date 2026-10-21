# Ledger

- 基线分支: sub-agent-co-work
- 轮次上限: 20
- 当前轮次: 15（在 round-14 状态之上叠加的修复轮）
- 当前步骤: 7（默认 check 排除 heavy-golden，名单在 test/CHECK_SSOT.md）
- 当前问题: **P1-FC-3（原 P1-9 A4）** — 默认 `check` 排除 label `heavy-golden`（`model_golden_test`、`Yolov8TrtDetection`、`yolov8_detection`）和 `valgrind`。名字、原因、替代门禁见 `test/CHECK_SSOT.md`。替代门禁是 hosted golden / GPU smoke：`MORTRED_CI_REQUIRE_WEIGHTS=1` 直接跑 `model_golden_test`，再用 `scripts/ci_assert_gtest_xml.py` 拒绝 skip。内存允许时 `MORTRED_CHECK_HEAVY=1 cmake --build <dir> --target check` 才跑 heavy-golden（`RUN_SERIAL`）。这不是整套数值全绿，也不再把未执行的全量 golden 写成默认 check 已通过。
- 阻塞: 无（全量 golden 不进默认 check；约 20GB 上的 OOM 由 `heavy-golden` 排除名单承接）
- 本机证据（2026-10-21，tests-only，无 GPU）: 默认 check 走 `ctest -LE valgrind|heavy-golden`，`ctest -L heavy-golden` 为 0，`model_golden_test` 未注册。带 workflow 库的一次 check 是 68 项里 67 通过；`supervisor_multiinstance_test` 在套件中 abort，单独重跑通过。该测试只在 workflow 可用时注册，CI 的 tests-only 不跑它。全量 golden 没有在这台机器上执行。
- 已关闭: P0-2…P1-8（第 13 轮 **11/11**）；**P1-5…P1-8**（本轮补记，merge `636a3990`）；合入后复核的 D1/D2/D3（`82e70987`）
- 记录目录: cowork/ledger/round-011/（本轮修复记录：`05`…`08`）
- 运行环境: 本机 WSL Ubuntu（/mnt/g/Codex/mortred_model_server）
- 状态: running
- HEAD: `82e70987`

> 说明（2026-10-19 补记）：`ff78c485` 之后，`fix/p1-5-9-metadata` 已在 `636a3990` 合入（P1-5…P1-8 + A1–A3 判定）；随后 cv-qa 的项目级复核判 fail，三条发现（文档仍教弱口令 / `check_consistency.py` 确定性红 / mint 脚本"热加载"文案错误）已在 `82e70987` 修复并复核：`check_consistency.py` 现在退出 0，`grep your-secret-key docs/` 为 0，`mint_api_key` 已在三份文档中出现。详见 `round-011/07-fix-p1-5-p18-closure.md` 与 `round-011/08-post-merge-review-fixes.md`。

## 轮次

| 轮次 | 复核加权分 | 未关闭 P0 | 未关闭 P1 | 未关闭 P2 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 1–9 | … | 0 | 0 | 0 | done |
| 10 | 7.4 | 0 | 2 | 8 | running |
| 11 | — | — | — | — | eval-only |
| 12 | — | 0 | 0 | 0 | done |
| 13 | — | 0 | 0 | 0 | done |
| 14 | — | 0 | 0 | 0 | done |
| 15 | — | 0 | 1 | — | running |
