# 合入

- 问题: P2-20
- 分支: cowork/r2-p2-20-trt-proof
- worktree: G:\Codex\mortred_wt_r2_p2_20（已删除）
- 问题分支提交: c15b1d723658b70079d0ab84e0d7e2e76344004c
- QA 结论: pass
- 重跑: `cd /mnt/g/Codex/mortred_model_server && python3 scripts/ci_eval_inference_paths.py --self-test`（root 为已合入基线），退出码 0，打印 `[ok] ci_eval_inference_paths.py --self-test`；另跑 `python3 scripts/ci_eval_inference_paths.py --cpu success --gpu success` 退出码 0 且 stdout 含 `TRT_PROOF=no`、不含 `TRT_PROOF=yes`
- 命令: `git merge --no-ff cowork/r2-p2-20-trt-proof`
- merge commit: bb1afafb952a14f18e0c996395cf46ae48f87215
