# 实现偏差确认

- 问题: P2-12
- 方案版本: v1
- 结论: 同意
- 理由: 接受这一处路径替换。自测仍是方案 v1 的只读断言，期望退出码为 0。变的只有 `cd` 与 `root`，由 `/mnt/g/Codex/mortred_model_server` 换成 `/mnt/g/Codex/mortred_wt_r2_p2_12`，断言正文未改，判定的仍是验收标准 1 至 4。基线两份教程仍是 `--model MOBILENETV2`、`--model SUPERPOINT`。实现写在仓库外的 worktree，提交 `31f28c9d` 相对基线只改这两份教程各两处 `--model` 实参。该 worktree 里分类命令是 `--model mobilenetv2`，特征点命令是 `--model superpoint`，下一行 toml 路径未动；`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN`、节标题里的大写词和两句 `worker_nums` 原文仍在。同一断言指到基线会读到改动前的教程。QA 在这个 worktree 上执行这段断言。方案版本保持 v1。
