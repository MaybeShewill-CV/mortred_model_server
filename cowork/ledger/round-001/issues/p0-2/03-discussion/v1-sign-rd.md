# 第3步签字

- 问题: P0-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 现有 `ProcessSupervisor::autostart_all` 已先起 `__gateway` 再起 model，但失败只打 stderr 仍进 model 循环；`start_server`/`trust_tokens` 已能把 scrape 永久失败落到 `__gateway` 的 `failed`+`error`；软化 `SupervisorApp::init` 的 scrape 早退即可让 `:8787` 存活并由 autostart 阻断；`handle_status` 可只读叠加阻断摘要；`mortredctl_doctor.sh` 的 pack TRT/occupancy GATE_FAIL 在 live 之前，叠加 status 检查不会把 P0-1 改回 WARN-only。自测入口（`scripts/run_tests.sh`、`mortredctl_doctor.sh`、现有 unittest）均可在仓库内对已有 build dir 执行。
