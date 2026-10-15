# Ledger

- 基线分支: sub-agent-co-work
- 轮次上限: 20
- 当前轮次: 1
- 当前步骤: 7
- 当前问题: P2-11
- 阻塞: 无
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10
- 本轮起点加权分: 7.1
- 记录目录: cowork/ledger/round-001/
- 运行环境: 本机 WSL Ubuntu（/mnt/g/Codex/mortred_model_server）
- 状态: stopped
- 停止原因: P2-11 已合入 `9f1dc0b33b2a6f778e4c305a58e3ad831f1aff32`，关闭后未阻塞的开放项仍是这一条。九行官方压测的 `-c` 都是该行 `worker_nums` 的两倍。PF1、PF2、PF4、PF5、TP1 仍是未满足。
- 续跑: 官方压测并发改成等于该行已部署的 `worker_nums` 之后，才能用这张清单勾销 P2-11。worktree `G:\Codex\mortred_wt_r1_p2_11` 仍有未跟踪的 `3rd_party`、`weights` 和四张 JPEG，没有删除。

## 轮次

| 轮次 | 复核加权分 | 未关闭 P0 | 未关闭 P1 | 未关闭 P2 | 结果 |
| --- | --- | --- | --- | --- | --- |
