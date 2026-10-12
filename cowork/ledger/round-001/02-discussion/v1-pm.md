# 第2步评审

- 轮次: 1
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 评审 | 理由 |
| --- | --- | --- |
| P0-1 | 保留 | 业务后端对已拉起的 `libface` 提交一张合法 JPEG，现在得不到成功结果。驻留集包含这个 id，调用发不出去。改完后同一张请求得到成功状态，并且 `libface.toml`、`conf/trt_profiles/set_a_libface.json` 与 `docs/benchmark-h3/libface.txt` 的高和宽相同。一次提交，再打开这三处，就能判定。这是静态高宽对齐，不是动态 shape。保持 P0。 |
| P1-1 | 拆开 | 这一段一次验收确认不了。拆成两件，不要再按 id 拆开。第一件是覆盖集九个 TensorRT id 各有一份按本次性能约束完成的 HTTP 报告：硬件、日期、pack 里的 `worker_nums` 与 `decode_auto`、`--raw` 命令、成功 RPS、p50、p99、错误率、`docs/benchmark-h3/<id>.txt` 的 Throughput 和比值，429 单独计数；同一次压测能读到模型进程上分开的排队等待和推理耗时，以及响应里的 `server_time_ms`。九份报告是一张清单。服务现在仍能调用，这不属于 P0，也不属于 P1 的换机、加家族模型、看清时间构成、单模型故障，拆出后放 P2。第二件保持 P1：不创建 `/tmp/mortred_stage_timing` 时，读取 gateway `/metrics` 能看到 `gw_in`、`route`、`auth`、`fwd`、`upstream`；gateway 阶段 p50 大于模型 `server_time_ms` p50 的一半时，报告写明时间主要花在哪一段。看清一次请求的时间构成才是 P1。 |
| P1-2 | 保留 | 按 `docs/how_to_add_new_server.md` 或中文 README 启动已有分类家族时，文档写的是 `DENSENET`、`MOBILENETV2`、`resnet50`，当前 catalog 使用的是 `densenet`、`mobilenetv2`、`resnet`。改完后打开这三处，要执行的 id 就是这三个。一次对照就能判定。这是 P1 里给已有家族加模型的文档步骤，不是把覆盖集以外的尺寸做齐。保持 P1。 |
| P2-1 | 保留 | 业务后端用覆盖集任一 id 一次提交 N 张合法图片（N>1），响应 `results` 长度应为 N，随后 `/metrics` 里 `mortred_batch_size` 的样本都是 1，且 count 等于这次张数。现在响应条数有，但这条指标没有 count=1 的样本。一次请求加一次读取就能判定。不要求动态 batch。PF3 未满足，保持 P2。 |
| P2-2 | 保留 | 打开 gateway 限流后，同一 IP 用错误 token 打满 burst，紧接着的合法推理应仍成功，401 不改变该来源的剩余配额。现在合法请求收到 429 和 `Retry-After`。按这个顺序调用一次就能判定。默认限流关闭，不是整机入口不可用，保持 P2。 |
| P2-3 | 拆开 | 改完后要确认的是三处子进程在 `fork` 与 `exec` 之间不再出现 `prctl` 和 `execvp`，并把 `nvidia-smi` 改成带绝对路径的 `execve`。业务后端发请求、看状态或日志都确认不了。ST5 未满足，整条不能删除，也不能当作范围外。源码对照和指定调用方式的句子拆出去，写入附录，不能当通过条件。留下的必须是一次操作就能判定的结果。 |
| P2-4 | 保留 | 业务后端在 `yolov8s` 与 `libface` 同时 ready 时并发请求两者，两边都要有成功响应。现在没有这份记录。一次并发调用就能判定；先停掉一个再压另一个不算。`libface` 单张失败是 P0-1，两边同时成功是另一件事，不合并。TP2 未满足，保持 P2。 |
| P2-5 | 保留 | 推理 token 缺失、管理 token 缺失，或二者相同而 metrics token 不同时，gateway 以非 0 退出、不监听，并写明拒绝原因。三枚都已设置且彼此不同时可以进入监听。这几种启动是同一件事，按清单各启动一次就能判定。验收不要求 Docker。保持 P2。 |
| P2-6 | 保留 | 对已安装的 `mortred-supervisor.service` 执行停止后，驻留集模型进程先退出，gateway 随后退出，gateway 退出后这些模型进程不再存在。一次停止就能判定。OP5 未满足，保持 P2。 |
| P2-7 | 拆开 | 业务后端用一张解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan`，通过条件写成两个互斥结果：成功返回增强图，或返回写明须为 256×256 的输入错误。一次验收无法判定哪一个算通过。拆开后队列只留一个可判定的结果，另一个写入附录。「送入会话的形状为 `1×256×256×3`」业务后端看不到，随不可留下的那句写入附录。不要求动态 shape 或动态 batch。正好 256×256 时现在可以成功，这不是这个 id 完全调不通，不升到 P0。 |
| P2-8 | 删除 | 业务后端调用这些 id 时，加载的是产品 toml，不是 `conf/onnx_sources.json`。九个产品 toml 已经是 `type=tensorrt`，把交换目录的 `product_type`、`runtime_path` 改成 engine，调用结果不变。`docs/unsupported-boundaries.md` 规定该文件是 ONNX 交换、合同和许可证的依据，engine 不能当交换物；LIBFACE 允许保持 onnx。这条没有写明哪一次交付必须突破。整条删除，写入附录。 |
| P2-9 | 保留 | 运行 `python3 scripts/golden_drift_check.py --check`，退出码为 0 即可判定，baseline 含 `dinov2_vits14` 与 `metric3d_512`。这是仓库检查，业务后端的推理响应不会变，但一次命令就能判定，满足 P2 入列，不是范围外或重复。保持 P2。 |
| P2-10 | 保留 | 按 `README.zh-cn.md`「第一小时：mortredctl」操作的人，命令块里没有 `mortredctl next`。英文 README 和 `docs/deployment.md` 已经能从 `next` 走到 `doctor`，所以这不是 P1。改完后这一节出现 `mortredctl next`，并且同一节能走到 `mortredctl doctor`。读这一节就能判定。保持 P2。 |

## 不同意的条目
- P1-1：验收标准无法判定。九份 HTTP 报告和「不创建标记文件时 gateway `/metrics` 仍能读到阶段耗时」不能一次验收。
- P2-3：验收标准无法判定。对照源码确认 `fork` 与 `exec` 之间的调用，业务后端确认不了。ST5 未满足，不能整条删除。
- P2-7：验收标准无法判定。成功返回增强图，和返回写明尺寸的输入错误，是两个互斥结果。
- P2-8：没有业务后端能观察到的调用变化，并要求把 ONNX 交换目录改成 engine 清单。`docs/unsupported-boundaries.md` 不允许把 engine 当交换物，这条没有写明哪一次交付必须突破。
