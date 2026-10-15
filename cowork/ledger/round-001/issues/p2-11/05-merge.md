# 合入

- 问题: P2-11
- 分支: cowork/r1-p2-11-http-rps
- worktree: G:\Codex\mortred_wt_r1_p2_11
- 问题分支提交: 44b95e6f、34bd6bab、f9273a9e
- QA 结论: pass
- 重跑: 在 WSL `/mnt/g/Codex/mortred_wt_r1_p2_11` 重跑判定脚本与三条自测，退出码都是 0。`checklist-ok rows=9 yolov8s=400.677`。`calibrate_pack.py self-test passed`。`http_infer_rps.py --self-test passed`。`coverage_http_report.py --check passed`。`git diff --stat 44b95e6f^ HEAD` 在 Windows 的同一 worktree 上退出码 0，三份脚本 1610 行新增、56 行删除。与 `05-qa.md` 的退出码一致。
- 命令: `python3 /mnt/c/Users/MaybeSheWill-CV/AppData/Local/Temp/p2-11-qa-check.py`；`git diff --stat 44b95e6f^ HEAD -- scripts/calibrate_pack.py scripts/coverage_http_report.py scripts/server/http_infer_rps.py src/models/backend/trt_session.cpp scripts/product_static_bs1.py src/server/base_server_impl.h src/control/gateway/gateway_app.cpp conf/server conf/packs`；`python3 scripts/calibrate_pack.py --self-test`；`python3 scripts/server/http_infer_rps.py --self-test`；`python3 scripts/coverage_http_report.py --check`。清单 `docs/perf/coverage_set_http_rps.md` 先提交为 `f9273a9e`。然后在基线执行 `git merge --no-ff cowork/r1-p2-11-http-rps`。
- merge commit: 9f1dc0b33b2a6f778e4c305a58e3ad831f1aff32
- worktree: 合入后仍有未跟踪的 `3rd_party`、`weights` 和四张 JPEG，没有删除。
