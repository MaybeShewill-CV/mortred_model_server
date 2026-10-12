# 第2步评审

- 轮次: 1
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 评审 | 理由 |
| --- | --- | --- |
| P0-1 | 保留 | 驻留集中的 `libface` 对一张合法 JPEG 没有成功响应，业务调用发不出去，符合 P0。三处高宽一致并且这张 JPEG 成功，一次就能判定。本机 WSL Ubuntu（`/mnt/g/Codex/mortred_model_server`）上按现有 `type=tensorrt`、static shape、batch=1 对齐即可，每个 worker 仍各自持有 TensorRT execution context。高宽以 toml、profile 还是 `docs/benchmark-h3/libface.txt` 为准，都不挡住这一次验收。 |
| P1-1 | 拆开 | 现在这一条有两次验收，任何一次没过，整条都关不掉。一次是：不创建 `/tmp/mortred_stage_timing`，gateway `/metrics` 就能读到 `gw_in`、`route`、`auth`、`fwd`、`upstream`，同一次能读到模型 `server_time_ms`。这是 P1「看清一次请求的时间构成」，用一个已经能成功推理的覆盖集 id 即可。另一次是九个 id 的 HTTP 报告，每个 id 单独勾销：`mortredctl calibrate --pack <pack> --write-pack`（worker 扫描和 JPEG 解码投票都走 `--raw`），按该 pack 的 `worker_nums` 与 `decode_auto` 部署，再跑 `scripts/server/http_infer_rps.py --raw`，并发等于已部署的 `worker_nums`，时长不少于 30 秒。报告写明硬件、日期、这两项、成功 RPS、p50、p99、错误率、429 单独计数，以及 `docs/benchmark-h3/<id>.txt` 的 Throughput 和比值，并引用该次分开的排队直方图与推理直方图。`conf/server` 保持 `worker_nums=1`。模型进程里排队与推理直方图已经分开，PF2 随报告验收，不再独立成条。尚无已记录值时 PF5 就是这份报告不存在，也不单独成条。`libface` 的成功 RPS 要等 P0-1 关闭，不能挡住其余 id。拆出的报告清单对不上 P1 的四件事，完成标准仍未满足，列为 P2。取证只在上述 WSL 里算数。性能只覆盖 static shape、batch=1；多个 worker 继续各自持有 execution context。`realesrgan` 的成功 RPS 只在请求成功且送入形状为 `1×256×256×3` 时关闭。 |
| P1-2 | 保留 | 按文档给已有分类家族加模型时，示例 id 与 `catalog()`、`--list` 不一致，要靠文档外的 id 才能对上，符合 P1。打开 `docs/how_to_add_new_server.md`、`README.zh-cn.md` 的 `--model`、两份 README 的 cpu 精选，一次就能核对 `densenet`、`mobilenetv2`、`resnet`。指南已经不要求每个模型单独做一个服务器二进制，这一句留在同一条里。 |
| P2-1 | 保留 | 覆盖集 server toml 都是 `max_batch_size=1`。一次提交 N 张合法图片后，响应条数已经按张组装，缺的是随后 `/metrics` 里 `mortred_batch_size` 的样本都是 1。一次请求加一次抓取就能判定。维持 static shape、batch=1，每个 worker 仍各自持有 execution context。调用还能完成，维持 P2。 |
| P2-2 | 保留 | 限流开关打开后，错误 token 会打满来源 IP 的配额，紧接着的合法推理收到 429。默认 `enabled = false`。打开同一开关，打满原先的 burst 再发一次合法推理，一次就能判定。维持 P2。 |
| P2-3 | 保留 | ST5 未满足：拉起模型子进程时，`fork` 之后、`exec` 之前有 `prctl`；两处 `nvidia-smi` 探测在 `fork` 之后调用 `execvp`。对照这三处，一次就能判定。这段路径没有 `setenv`，调用现在仍能发出，维持 P2。验收范围就是这三处，停在 supervisor 拉起模型和这两处探测。 |
| P2-4 | 保留 | 缺一份 `yolov8s` 与 `libface` 同时 ready、并发请求、两边都有成功响应的记录。同一份记录一次就能判定；先停掉其中一个再压另一个不能当作这次验收。`libface` 的成功响应依赖 P0-1，两条分开关闭，形状问题不挡住这份双模型记录的其余判定。维持 P2。 |
| P2-5 | 保留 | 推理 token 缺失、管理 token 缺失，或二者相同而 metrics token 不同时，gateway 仍会监听。三种启动各看退出码和非监听，是同一次验收。三个 token 都已设置且彼此不同时可以进入监听。三个 token 设成同一个非空值时，现有检查已经拒绝启动。场景里的 compose 启动不进入验收：本机没有 Docker 测试环境，镜像与默认运行用户在范围外。维持 P2。 |
| P2-6 | 保留 | 停止之后，驻留集模型进程先退出，gateway 随后退出，gateway 退出后这些模型进程不再存在。一次停止就能判定。本机验收是 WSL Ubuntu。这里若没有 systemd，OP5 仍用 supervisor 主进程收到停止信号后的进程顺序验收；该行不属于可标阻塞的性能、吞吐、engine、占用。单元缺少 `KillMode` 是这条顺序的出处，不另开一条。维持 P2。 |
| P2-7 | 保留 | 解码后正好 256×256 时，`realesrgan` 已经能成功推理，这个 id 可以单独拉起并完成一次成功推理。一张解码后宽或高不是 256 的合法 JPEG，结果是二者之一即可：成功返回增强图且送入会话的形状为 `1×256×256×3`，或返回写明尺寸须为 256×256 的输入错误。一次请求就能判定，引擎保持 static shape、batch=1。维持 P2。 |
| P2-8 | 保留 | 九个覆盖集 id 的 `product_type`、`runtime_path`、`config` 与对应产品 toml 的 `type`、`model_file_path` 是否一致，并且 `config` 指向的文件能否打开，对照交换目录一次就能判定。`YOLOV8` 已是 `runtime_path` 指向 static batch=1 engine、`onnx_path` 仍保留 ONNX；其余 id 按同一对照交付。产品 toml 保持现有 `type=tensorrt`。`LIBFACE` 的 config 路径打不开与 P0-1 的高宽不一致是两次验收，分开关闭。服务按产品 toml 加载，换机已有 `prepare --id`，维持 P2。 |
| P2-9 | 保留 | `python3 scripts/golden_drift_check.py --check` 现在退出 1，CI 的 `check_consistency.py` 会走到同一检查。退出码 0，且 `case_names`、`golden_sha256` 含 `dinov2_vits14` 与 `metric3d_512`，一次就能判定。推理调用不依赖这次检查，维持 P2。 |
| P2-10 | 保留 | `README.zh-cn.md` 第一小时命令块出现 `mortredctl next`，并且同一节能走到 `mortredctl doctor`，打开该节一次就能判定。UX2 已由英文 README 和 `docs/deployment.md` 满足，这条仍有出处、现状和一次外部验收，维持 P2。与 P1-2 对的不是同一条命令，分开关闭。 |

## 不同意的条目
- P1-1：不同意维持为一整条。不创建 `/tmp/mortred_stage_timing` 时 gateway 阶段耗时默认可读，与九个 id 各自的 calibrate、`--raw` 报告，不是同一次验收。
