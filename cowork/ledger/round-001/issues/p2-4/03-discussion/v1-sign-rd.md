# 第3步确认

- 问题: P2-4
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 现有 supervisor 读到 `MORTRED_PACK` 后就会 `autostart_all()`。pack 里的 id 被标成 autostart，其余 catalog id 不拉起。`conf/packs/gpu_yolo_metric3d.toml` 已是两节 `[pack.<id>]`、`worker_nums = 1`、`gpu_mem_mib = 1800`、`gpu_mem_at_workers = 1`、`gpu_mem_source = "seed"` 的同一格式，临时 pack 换成 `yolov8s` 与 `libface` 能过 `apply_pack`。`start_concurrency = 1` 只限制启动宽度：第一个进入 `running` 后才拉起第二个，两者随后留在同一次 `GET /api/v1/status` 里。该响应已有 `gateway` 和 `servers[]` 的 `id`、`state`、`pid`、`ready`，管理面用现有 `MORTRED_API_TOKEN`。`ready` 要等 `/ready` 返回 2xx，而 `/ready` 要等初始化完成且 worker 可用。gateway 已把 `POST /v1/models/{id}/infer` 按 catalog id 精确转发到 9056 与 9051 的现有 `server_uri`。请求信封只要 `images` 长度为 1 的 base64。成功响应已是 HTTP 200、根上 `status` 为 0、`results` 长度为 1、`results[0].status` 为 0；根上 68 是 `DEADLINE_EXCEEDED_PARTIAL`，仍为 HTTP 200，判定可以排除。pack 注入 `MORTRED_WORKER_NUMS=1`，两份 server toml 也是 1，每个 worker 继续自己 `createExecutionContext()`。产品 toml 仍是 `type=tensorrt`：`yolov8s` 为 `[640, 640]`，`libface` 为 `[480, 640]`，与现有 static batch=1 engine 一致。临时 pack 不写 `gpu_memory_total_mib` 时，stamp 之和相对总显存的检查不启用；`nvidia-smi` 读到空闲显存且低于 1800 MiB 时，现有门闩拒绝第二个 spawn，脚本记成非 0 即可。三个 token 从现有环境或 `conf/local/trust.env` 注入后，gateway 按现有规则启动。SIGINT 打到 supervisor 主进程会走现有的先模型、后 gateway 收尾。`_bin` 下的 supervisor、gateway、model server 都在，两份 engine、两张 demo 图和 trust.env 也可读。`--check` 只跑判定函数，不加 valgrind 标签就会进默认 `ctest -LE valgrind`。
