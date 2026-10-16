# 技术方案
- 问题: P0-3
- 版本: v1
## 目标
整报文 `receive_timeout` 可配置且默认有界（非 -1）；滴流连接在 T 内被关闭。
## 边界
- 不改 packing/Docker。
- peer_response_timeout 语义不变（单次读写）。
## 实现要点
1. `ServerRuntimeConfig::receive_timeout`（ms），默认 60000；TOML `receive_timeout`（秒，与 peer_resp_timeout 风格一致）。
2. `init_http_server` 写入 `server_params.receive_timeout`。
3. schema optional int；单测默认与解析。
## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | Round 9 锁定 |
| cv-pm | 同意 | 无 | v1 | 用户锁定 |
| cv-tech-lead | 同意 | 无 | v1 | 用户锁定 |
| cv-rd | 同意 | 无 | v1 | 用户锁定 |
