# 第3步确认

- 问题: P2-7
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 整数 8 空着（7 之后是 11）。`MODEL_INPUT_SIZE_REJECTED` 放进现有状态宏后，`status_code_to_str` 会给出全文 `尺寸须为 256×256`（U+00D7，`为` 与 `256` 之间一个空格），`http_status_of` 可以把 8 映到 400。`max_batch_size=1` 走 `run()` → `prepare_inputs`；该函数可在派生类里覆盖，解码后的工作图宽或高不是 256 时直接返回 8，`run_impl` 会保留这个非 OK 状态，不会收成状态 3，也不会调用 `session().run`。两边都是 256 时仍用现有的一次预处理和一次会话。验收进程没有 `MORTRED_DECODE_AUTO`，模型里的 `image_decode_backend=auto` 会落到 CPU。现有鉴权不改：非空 `MORTRED_AUTH_TOKEN` 覆盖空的 toml token，这次 POST 带同一个 `Authorization: Bearer` 就能进模型；`/ready` 仍公开。统一响应已有 `status`、`status_str`、`results[0].status`、`results[0].data` 为 null、`partial` 为 false、`model.name` 为 `realesrgan`。自测第一条编译并跑整份 `status_code_unittest`。第二条已带 `export P2_7_PID`、非空 token 和同一个 Bearer，`status_str` 做全文相等。两条命令都可以原样执行，期望退出码为 0。
