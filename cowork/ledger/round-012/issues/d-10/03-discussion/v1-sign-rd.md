# 第3步签字

- 问题: D-10
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `GpuSampler::loop` 在调用 `parse_i32_or` 之前握有整段非空 CSV。`parse_sample` 按逗号切开并裁掉空白后，验收行 `FakeGPU, [N/A], 2048, 8192, 40, 20.5, 1000, 5000, 30, 4, 16` 与 `N/A` 都是第 2 列、共 11 列，整列等于 `[N/A]` 或 `N/A`。此处把 `available_` 设为 false、样本仍入 ring（`util` 仍是 -1）、休眠回到 `gpu_sample_interval_ms`，并用现有的 `first` 只写一行 `gpu sampler offline (utilization N/A)`。空输出、整数利用率的 online 行、以及 `gpu_name_` 都留在原路径。`GET /api/v1/gpu` 已经用紧凑 JSON 写出 `available()` 和整数 `util`。`cmd_ps` 在 `samples` 非空时打印 GPU 行；`-1` 改印 `util n/a` 且不加百分号后，非终端颜色码为空，行文就是 `  GPU         ● FakeGPU  util n/a  vram 2.0G/8.0G (25%)  temp -1°C  pwr -1W`。显存按 2048/8192 现算；JSON 温度键是 `temp`，CLI 读 `temp_c`；`power_w` 是 double，`json_i64` 得到 -1。网关 `18080` 没有 healthz 时退出码已是 1，该路径不写 stderr。临时 toml 的 60000 毫秒在配置上限内，空的 `conf/server` 能初始化，缺少 gateway 二进制会在 listen 之后打出 `autostart blocked:`。四条验收都能从这两处改动一次看见。
