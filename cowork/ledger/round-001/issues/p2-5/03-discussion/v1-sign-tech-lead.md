# 第3步确认

- 问题: P2-5
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 锁定句的三种情况分别落在验收标准 1 至 3，标准 4 是三个非空且两两不同的 token 进入监听。四次都直接执行 `mortred-gateway.out`，临时根带同一份含一个 key 的 `conf/api_keys.toml`，子进程不继承调用方的 `MORTRED_*`。拒绝时退出码非 0，stderr 为对应的 `mortred-gateway: refusing to start: ` 全文，没有 `mortred-gateway listening on`，也不出现当次 token 的值；通过时监听日志含 `http://127.0.0.1:` 和当次端口，`GET /healthz` 为 200，正文为 `ok\n`。src-control 把启动判定放在 `trust_tokens.h`，顺序是先维持现有 metrics 拒绝，再拒绝推理 token 缺失、管理 token 缺失、二者相同。`GatewayApp::init` 在 `listen` 之前调用，已加载的 API key 不跳过。`spawn_locked` 在 `exec` 之前、在「executable not found」之前用同一判定，错误进入现有永久失败，`__gateway` 为 `failed`。`scrape_token_usable` 对空 metrics、metrics 相撞，以及空推理或空管理不与 metrics 碰撞的规则保持。supervisor 主进程对空 `MORTRED_API_TOKEN` 的拒绝保持；推理 token 缺失或二者相同只让 `__gateway` 永久失败，管理面继续监听。ops 的启动点就是这个 gateway 二进制，以及 supervisor 拉起的 `__gateway`。`security_warn.sh`、`mortredctl doctor`、`docker-compose.yml` 和 `deploy/` 不充当这次拒绝；短于 32 字符的三个不同 token 可以监听，长度仍留在 doctor 警告里。Docker 不进入验收。限流归还、停止信号、每个 worker 各自的 TensorRT execution context 和动态 batch 保持原样，P2-6、P2-7、P2-11 不进入这次改动。两条自测都在本机 WSL `/mnt/g/Codex/mortred_model_server`，期望退出码为 0。
