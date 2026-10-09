# 第3步签字

- 问题: D-2 attentive_gan.toml 输入尺寸仍 stale
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 三份 toml 现在都是 `model_input_image_size = [240, 320]`。`ATTENTIVE_GAN_DERAIN` 仅一条，`input_tensor.shape` 为 `[1, 240, 360, 3]`，`export_attentive_gan.py` 为 `HEIGHT, WIDTH = 240, 360`。`on_init` 用 session 的 `shape[1]`、`shape[2]` 做 resize 和解码预算，`params` 只参与放大倍数，不读这个尺寸键。手改三份 toml 为 `[240, 360]`，并删掉 `contract.preprocess`、`onnx_source.notes` 与 `update_onnx_sources_p2.py` 里同一句「`[240,320]` / stale」即可。notes 里已有 ASCII `240x360` 与 `[1,240,360,3]`；preprocess 里已有乘号 `240×360`。`stale` 与 `240,320` 在该脚本中只出现这一处。自测对 notes 与 preprocess 分开判定，不把两段拼起来搜一次 `240x360`。产品、CI、bench 的 backend，以及 `server_uri`、`model_config_file_path` 与验收字符串一致，方案不改它们。不改进程，不新建 engine，不改推理像素。
