# 合入

- 问题: P0-1
- 分支: cowork/r1-p0-1-libface
- worktree: G:\Codex\mortred_wt_r1_p0_1
- 问题分支提交: 6faa446aea51d32e5e5a8a9989f3237e6558797c
- QA 结论: pass
- 重跑: 验收 1、2、3、6、7、4、5 的退出码都是 0，与 `05-qa.md` 一致。标准 7 内部的 `python3 scripts/product_static_bs1.py --check` 退出码是 1，stderr 只有 `mobilenetv2_cpu.toml` 与 `resnet_cpu.toml` 的 MNN 报错，没有 `libface params are not 640x480`。
- 命令: 在 WSL 执行 `bash /mnt/g/Codex/rerun_p0_1.sh`。脚本在仓库外，合入后删除。然后在基线执行 `git merge --no-ff cowork/r1-p0-1-libface`。
- merge commit: 9ffd798a30b406e9ace78a59943d7aae20cf2f48
