# 第3步签字

- 问题: D-9
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 鉴权失败后紧接着调推理的人，在 `enabled = false`、`shadow = false`、`rate_per_sec = 1`、`burst = 1` 时跑三条命令。验收 1 退出码 0，标准错误为空，标准输出依次是无 Bearer 的 `401` 与 `unauthorized` 正文，以及带 `Bearer d9-infer` 的 `503` 与 `service not ready` 正文。验收 2 退出码 0，标准错误为空，同一秒内两次合法推理都是 `503` 与同一条 `service not ready` 正文。验收 3 退出码 0，标准错误为空，`scope = "none"` 且 `rate_limit_qps = 1` 的 key 连续两次都是 `401` 与同一条 `unauthorized` 正文。三条各自一次命令的退出码和输出原文就能判定冻结句里的三种变化。
