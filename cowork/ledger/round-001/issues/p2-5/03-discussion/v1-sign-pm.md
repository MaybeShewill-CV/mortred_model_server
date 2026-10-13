# 第3步确认

- 问题: P2-5
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 4 已编号，四次都直接执行 gateway，一次就能判定。临时根里已有一份启用的 `conf/api_keys.toml`。推理 token 缺失、管理 token 缺失、二者相同且 metrics token 不同时，退出码都非 0，stderr 分别是三句拒绝全文，没有监听行，也不含当次 token 字面量；进程退出后对该端口的 `GET /healthz` 失败。key 文件不能让这三种进入监听。标准 4 的三个 token 都短于 32 字符且两两不同，同一份 key 文件仍在，进程保持运行，stderr 含 `127.0.0.1` 和当次端口的监听行，不含这三个字面量，`GET /healthz` 为 200、正文为 `ok\n`。这是现有启动拒绝的扩展：metrics token 为空，或与推理 token、管理 token 之一相同时，gateway 已经拒绝启动。空的 `conf/server/` 仍能完成这次启动判定。不使用 Docker，不改监听之后的 Bearer，健康检查仍公开。
