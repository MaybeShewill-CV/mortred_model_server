# P0 与 P1

- 轮次: 8
- 版本: v1
- 综合分上限: 有 P1 时不超过 8.9

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | lightglue 与 openai clip 把非空故障错成 `MODEL_EMPTY_OUTPUT`，且缺无权重契约测试；须同一条收完两家族 | 现场接 lightglue / openai clip 时输出 dtype、尺寸或 embedding 形状不符 | dtype/尺寸不符与 embedding 形状不兼容仍回 `MODEL_EMPTY_OUTPUT`；输出缺失与空文本保持 EMPTY | 非空故障返回契约失败码；无权重契约测试能定位到对应分支 | `src/models/feature_point/lightglue.inl`；`src/models/clip/openai_clip.inl` |
| P1-2 | 遗留 `conf/trt_engines.json` 中 yolov8s engine 仍指向旧文件名 | 小团队按遗留清单准备 TRT | 该文件写 `yolov8s.engine`；产品路径是 `*.engine.static_bs1.fp16` | 遗留清单不再指向旧文件名，或读取方失败并点名产品路径；产品真源不改回这份文件 | `conf/trt_engines.json`；`conf/product_static_bs1.json` |
| P1-3 | 检测或分割缺 dtype / NaN / Inf 拒绝矩阵 | 交付前确认坏张量被拒绝，机器上可能没有权重 | 缺权重时相关门会 SKIP | 无权重也能对 dtype / NaN / Inf 失败定位；不以 SKIP 冒充通过 | `test/` |
| P1-4 | 停服日志分不清 drain timeout 与成员开始销毁 | 停 catalog 时排障 | 析构在 drain 超时返回后继续销毁成员，两阶段日志不分 | 日志能区分「drain timeout」与「成员开始销毁」；仍一 catalog 一进程，外层 SIGKILL 保留 | `src/server/base_server_impl.h` |

## 附录
无

## 签字
| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| cv-pm | 同意 | v1 | `cowork/ledger/round-008/02-discussion/v1-sign-pm.md` |
| cv-tech-lead | 同意 | v1 | `cowork/ledger/round-008/02-discussion/v1-sign-tech-lead.md` |
| cv-cto | 同意 | v1 | `cowork/ledger/round-008/02-discussion/v1-sign-cto.md` |
