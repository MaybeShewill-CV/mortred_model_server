# 合入

- 问题: P2-21
- 分支: cowork/r2-p2-21-trt-xml
- worktree: G:\Codex\mortred_wt_r2_p2_21（已删除）
- 问题分支提交: 970b276e13fccdb78fad73865b694be9f95d999f
- QA 结论: pass
- 重跑: `cd /mnt/g/Codex/mortred_model_server && python3 scripts/ci_eval_inference_paths.py --self-test`（root 为已合入基线），退出码 0，打印 `[ok] ci_eval_inference_paths.py --self-test`；另跑 `python3 scripts/ci_eval_inference_paths.py --cpu success --gpu success` 与 `--product-trt yes`（无 `--gtest-xml`）退出码 0，stdout 均含 `TRT_PROOF=no`、不含 `TRT_PROOF=yes`
- 命令: `git merge --no-ff cowork/r2-p2-21-trt-xml`
- merge commit: c3bdf5a30b89e2d1bc11810b415aa665920990be
