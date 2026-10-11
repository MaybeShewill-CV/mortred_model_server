# 挂起

- 问题: D-8
- 方案版本: v1
- 分支: cowork/r12-d8-gpu-user
- 工作树: G:\Codex\mortred-cowork-r12-d8
- 问题分支提交: 4a7320637d2a692cd01b58f440f43a9854a15d69
- QA 结论: blocked
- 缺的环境: 可执行的 Docker，以及能拉取 Dockerfile 钉死的 CUDA 基镜像
- 恢复条件: 本机或 WSL 有可执行的 `docker` 后，只重跑验收 1
- 未跑的验收: 1
- 重跑: 未跑。本机 `where.exe docker` 退出码 1，WSL 里没有 `docker`
