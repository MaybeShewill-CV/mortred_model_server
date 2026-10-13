# 合入

- 问题: P2-9
- 分支: cowork/r1-p2-9-golden-drift
- worktree: G:\Codex\mortred_wt_r1_p2_9
- 问题分支提交: 3123def874691a807e4c5db07bae07e37a4cca35
- QA 结论: pass
- 重跑: 在 `/mnt/g/Codex/mortred_wt_r1_p2_9` 重跑 `--check` 退出码 0，标准输出为 `zero drift: 49 cases and 62 golden files unchanged`。名字与键检查退出码 0。两个 json 的字节 SHA-256 与账本一致，退出码 0。相对 HEAD^ 的 diff 只有 `test/golden_baseline.json`，退出码 0。与 `05-qa.md` 的退出码一致。
- 命令: `python3 scripts/golden_drift_check.py --check`；方案第二条自测；`hashlib.sha256` 对照 `dinov2_vits14.json` 与 `metric3d_512.json`；`git diff --name-only HEAD^ HEAD`。然后在基线执行 `git merge --no-ff cowork/r1-p2-9-golden-drift`。
- merge commit: f79e11ae6ceb167095ccd73483cb654f2032e57c
