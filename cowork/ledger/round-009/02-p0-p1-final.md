# 交付队列

- 轮次: 9
- 版本: v1
- 性质: 长期根本修复（非 band-aid）；顺序锁定；无 packing/Docker
- 审计基线: `d143d57e`（Round 8 已关闭；每条"现在的行为"都在该基线上复验）
- 来源: 第二次源码级评估的剩余项。**用户指令移出**上一版 P0-1（构建/门禁断裂）与上一版 P1-4（数值门禁可跳过），见附录

## P0

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | `/ready` 把"有空闲 worker"当就绪信号，满载时返回 503，一次过载会把健康实例摘出入口 | 驻留集全部 worker 都在推理时探活（LB / k8s / 外部探针 / `mortredctl doctor`） | `base_server_impl.h:354`：`ready = _m_successfully_initialized && _m_workers.available_approx() > 0`，同一分支还 `set_ready(ready)` 把 `mortred_ready` 置 0。负载下探针持续拿 503 → 健康实例被摘除 | 压满全部 worker 期间 `GET /ready` 仍为 200；饱和只用独立 gauge 暴露（`/metrics` 可读到 > 0），`mortred_ready` 不因饱和翻转。一次外部验收：`scripts/server/http_infer_rps.py --raw`（并发 = 2×已部署 `worker_nums`）压测期间同时抓 `/ready` 与 `/metrics` | `src/server/base_server_impl.h:354`；`src/server/prometheus_metrics.h` `set_ready` |
| P0-2 | 请求体大小限制在**解析之后**才生效，且 `/jobs` 完全没有限制 | 提交超过 `request_size_limit` 的 body（含 chunked、含缺 `Content-Length`） | `serve_process` 先按 Content-Type 解析 json/raw（`base_server_impl.h:398`、`:401`），到 `:415` 才判 `declared_body_exceeds`；`Content-Length` 缺失时该判断返回 false（`request_admission.h:54-65`）；异步提交路径无任何体积检查。单请求内存放大 ≈ raw msgbuf + `decode_chunked_body` 拷贝 + rapidjson DOM + base64 输出，上限 64MB × 200 连接 | 超限 body 在读取/解析前即 413，进程 RSS 不随 body 线性增长；`/jobs` 同样 413；缺 `Content-Length` 的 chunked 也受限。一次外部验收：body=128MB（limit=64）→ 413，且期间 RSS 增量 < 16MB；对 `/jobs` 重复同一验收 | `src/server/base_server_impl.h:398,401,415`；`src/server/request_admission.h:54-65`；`src/server/async_endpoints.h` 提交路径 |
| P0-3 | 整报文无超时：200 条滴流连接即可占满入口 | 慢速客户端每 14s 发一个字节，占住连接直到 `max_connections=200` 用尽 | `init_http_server` 从 `SERVER_PARAMS_DEFAULT` 起只覆盖 `max_connections`／`peer_response_timeout`／`request_size_limit`（`abstract_server.h:116-119`）；`receive_timeout` 与 `keep_alive_timeout` 沿用库默认 **-1** 与 **60000**（`3rd_party/include/workflow/WFServer.h` 的 `SERVER_PARAMS_DEFAULT`）；`peer_response_timeout` 是每次读写超时，不覆盖整报文 | 半截请求在文档化时限内被服务端关闭，连接数回落；并发滴流连接不阻止正常请求成功。一次外部验收：开 200 条滴流连接 → 正常推理请求仍 200，滴流连接在 T 秒内被关闭（T 写进 conf，错误信息里给出该值） | `src/server/abstract_server.h:105-122`；`3rd_party/include/workflow/WFServer.h` `SERVER_PARAMS_DEFAULT`（`receive_timeout=-1`、`keep_alive_timeout=60000`） |
| P0-4 | supervisor 在日志 I/O 停顿时永久挂死：子进程不再回收、`stop` 无效、收 SIGTERM 后不退出 | 日志文件系统停摆（NFS / 满盘），或某个子进程写满 stdout 后阻塞且忽略 SIGINT | `LogBuffer` 每次 append 都 flush（`log_buffer.h:61`，另有 `:35`、`:107`）；读管道线程无超时；`handle_exit` 作为唯一 joiner 在 **reaper 线程**上 join 两个读线程（`supervisor.cpp:812,815`）；`wait_shutdown()` 的 `_cv.wait` 无超时（`supervisor.cpp:1012`）。链条：子进程 write 阻塞 → 读线程阻塞持锁 → reaper 卡死（`:827`）→ 其余子进程全部不再回收 → `stop` 在 grace 后放弃 → SIGTERM 后 daemon 不退出 | 用一个"写满 stdout 且忽略 SIGINT"的假子进程复现时：其余子进程仍被正常回收；`stop` 在 grace 后返回；supervisor 收 SIGTERM 后 30s 内退出。一次外部验收：脚本（`test/fake_model_server` 变体）+ `ps` 与退出码断言 | `src/control/log_buffer.h:35,61,107`；`src/control/supervisor.cpp:812,815,827,1006,1012` |

## P1

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | 检测参数语义与实现不一致：`clip_boxes` 完全无效、`min_box_area_px` 坐标空间随检测器变化 | 请求或配置里设 `clip_boxes=true` / `min_box_area_px` | `clip_boxes` 被解析并存储（`detection_params.h:26,100-108`）但全仓无使用者；`min_box_area_px`（`:25,32-33`）在 v5/v6/v7 作用于网络空间，而在 v8 于 letterbox unmap（`yolov8_detector.inl:85`）**之前**过滤（`:76`）；其文档写的是"unmap 之前" | 同一张图、同一 `min_box_area_px` 在覆盖集检测 id 上过滤掉的框数量一致；`clip_boxes` 在贴边目标上可观察到 true/false 差异（或从 `param_spec` 与 `docs/openapi.json` 中删除该键）。一次外部验收：同一夹具在两个检测 id 上跑并比较过滤计数 | `src/models/object_detection/detection_params.h:22-33,90-108`；`yolov8_detector.inl:76,85`；`test/object_detection_output_contract_unittest.cc:297-312` |
| P1-2 | 管理/抓取凭据泄漏到每个被监管的模型进程 | supervisor 已拉起驻留集后检查任一模型进程的环境 | `supervisor_app.cpp:970` 全局 `setenv("MORTRED_API_TOKEN", auth_token_, 1)`；`spawn_locked` 把父进程整个 `environ` 拷给非 gateway 子进程（`supervisor.cpp:376`），不清理 `MORTRED_API_TOKEN` / `MORTRED_METRICS_TOKEN` / `MORTRED_GATEWAY_AUTH_TOKEN` → `/proc/<model_pid>/environ` 可读到管理 bearer | 模型进程 environ 中这三个键为 0 个，而 gateway 仍拿到它需要的 token，管理接口仍可用。一次外部验收：`tr '\0' '\n' < /proc/<model_pid>/environ \| grep -c MORTRED_API_TOKEN` = 0 | `src/control/supervisor/supervisor_app.cpp:970`；`src/control/supervisor.cpp:376` |
| P1-3 | 停止/重启只 kill 单 pid，没有进程组，孙进程存活 | 停掉或重启驻留集里的一个 id，而该进程 fork 过后代 | 只有单 pid `::kill`：`supervisor.cpp:597,603,727,890,899,915,923`；无 `setpgid`/`setsid`/`killpg`，孙进程在父进程被 SIGKILL 后继续存活 | 停止一个 id 后该 id 的所有后代进程在 10s 内消失。一次外部验收：脚本停止该 id 后断言 `ps -ef` 中无其残留进程（对应 rubric OP5） | `src/control/supervisor.cpp:597,603,727,890,899,915,923` |
| P1-4 | 请求路径上的 `CHECK` 断言会杀掉整个模型进程 | 会话输出形状或 dtype 与契约不一致 | `tensor.h:239`（`Tensor::make`）与 `:310`、`:316`（`data<T>()`）的 `CHECK`/`CHECK_EQ` 直接 abort：该 id 的所有在途请求同时丢失，只能靠 supervisor 重启恢复 | 构造形状/dtype 不一致的会话输出用例 → HTTP 返回 5xx（`MODEL_OUTPUT_CONTRACT_FAILED`）且进程存活。一次外部验收：跑该用例，退出码 0 且进程仍在 | `src/models/backend/tensor.h:239,310,316` |
| P1-5 | 无界等待与配置矛盾：`model_run_timeout<=0` 时请求永不回复 | `conf/server/**` 里写 `model_run_timeout=0` | 启动只打印警告（`server_runtime_config.h:139-143`）；worker checkout 变为无界等待、不创建请求定时器、`_m_waiting_jobs` 不回落 → 后续请求永久 429，`stop()` 可能挂起 | 该配置被拒绝启动（或强制有界并记录生效值），并有一条配置用例。一次外部验收：用 `model_run_timeout=0` 启动 → 非 0 退出且错误信息给出可用范围 | `src/server/server_runtime_config.h:139-143`；`src/server/base_server_impl.h` checkout 与定时器分支 |
| P1-6 | `/metrics` 的 method 标签取自未鉴权的原始请求方法，标签基数无界 | 未带 token、method 用随机长字符串的请求 | `base_server_impl.h:336,381,407,416,450,480` 用原始 `request_method` 作为指标 map key，永不裁剪；401 路径在鉴权前即可写入 | 未知 method 归一到固定标签集（或直接拒绝），压测后 `/metrics` 的 series 数有上界。一次外部验收：发 1000 个不同 method → series 数不增长 | `src/server/base_server_impl.h:336,381,407,416,450,480`；`src/server/prometheus_metrics.h` 标签 map |
| P1-7 | 三个契约测试与产品行为不一致（测试未随产品改动更新） | 跑 `request_envelope_unittest` / `server_e2e_contract_test` / `gateway_e2e_test` | ① `request_envelope_unittest.cc:43,57` 期望 base64 文本，而实现已在 item 生成处解码（`parsed_request.h:126-140`，2026-09-21 `d9a49675`；测试最后修改 2026-09-09）；② `server_e2e_contract_test.cc:713,742` 仍断言 `rate_limit_qps=1` 产生 429，而模型侧限流已移除；③ `gateway_e2e_test.cc:882` 期望"0.0.0.0 + 关闭限流"能启动，产品已 fail-closed 拒绝 | 三条用例与产品契约一致（改测试或改产品，并把 `rate_limit_qps` 的语义写进 conf schema 与 `docs/openapi.json`），逐个通过。一次外部验收：分别运行这三个二进制，0 失败 | `test/request_envelope_unittest.cc:43,57`；`src/server/parsed_request.h:126-140`；`test/server_e2e_contract_test.cc:713,742`；`test/gateway_e2e_test.cc:882`；本人实跑结果 |

## P2

（无）

## 附录

| 类别 | 陈述 |
| --- | --- |
| 用户指定移出（本轮不入任何档位） | 上一版 P0-1：构建/门禁在 HEAD 上不成立（`packed_batch_nvi_unittest`、`cv_image_input_unittest` 链接失败 → `check` 中断 → 57/93 用例从未运行；3 处红）。上一版 P1-4：数值门禁可跳过（`golden_process_exit` 部分通过返回 0、缺权重一律 skip、报告不写实际执行数）。证据留在 `round-008/` 与第二次评估记录中 |
| rubric 明文范围外（需用户改 rubric 才能入列） | SAM／diffusion／CLIP／LightGlue／RTDETR／MOT／matting／`msocrnet`；Docker 镜像与默认运行用户；一个 catalog id 一个 OS 进程的结构与每进程约 370MB／61 线程基线；Windows/macOS 与其他架构；MNN 与 ORT 的吞吐；动态 shape／动态 batch；多机与 K8s |
| 范围外但客观成立（本轮不入列） | `pp_matting` alpha 用 `INTER_NEAREST` + `CV_32SC1`（与同族 modnet 的 `INTER_LINEAR` + `CV_8UC1` 不一致）→ 静默输出台阶状 alpha 边缘；CLIP `simple_tokenizer` 是贪心最长匹配而非 BPE，非 ASCII 输入直接 `TOKENIZE_UNKNOWN_TOKEN`；SAM encoder 把 BGR 送进 RGB 归一化系数 |
| 需用户决定是否扩范围 | 依赖树可复现（6 个 vendored 源中 5 个无 hash）与离线安装；产物随包分发 `LICENSE`/`NOTICE` 与被捆绑 TensorRT/cuDNN 的再分发许可、SBOM 覆盖二进制；内置 TLS（现由 `deploy/nginx` 终止）；审计日志（无格式约定） |
| 上一版候选 P2（未采用，留档） | golden 掩膜裸整数 JSON 533MB；`rate_limit_qps` 在 44 份配置中无效；死代码（`do_work` 等）；异步结果每次轮询深拷贝与无界保留；stuck worker 只有日志无指标；`model_config_file_path` 相对 CWD；无响应压缩；`test/contract_baseline/*.md` 无消费者；`/metrics` 被推理 token 挡住。原文在提交 `9977b57d` 的 `cowork/ledger/round-008/02-p0-p1-final.md`（该文件随后被第 8 轮 v2 队列覆盖） |

## 分类说明

1. **编号对照**（本版重排，避免与已删除项混淆）：

| 本版 id | 上一版 id | 问题 |
| --- | --- | --- |
| P0-1 | P0-2 | `/ready` 满载 503 |
| P0-2 | P0-3 | 体积检查在解析之后 / `/jobs` 无检查 |
| P0-3 | P0-4 | 整报文无超时（slowloris） |
| P0-4 | P0-5 | supervisor 日志/回收/关机挂死链 |
| P1-1 | P1-1 | 检测参数语义不一致 |
| P1-2 | P1-2 | 凭据被模型子进程继承 |
| P1-3 | P1-3 | 只 kill 单 pid，无进程组 |
| P1-4 | P1-5 | `tensor.h` CHECK 在请求路径 abort |
| P1-5 | P1-6 | `model_run_timeout<=0` 无界等待 |
| P1-6 | P1-7 | `/metrics` method 标签基数无界 |
| P1-7 | P1-8 | 三个契约测试与产品不一致 |
| — | P0-1 | **已按用户指令删除**（构建/门禁断裂） |
| — | P1-4 | **已按用户指令删除**（数值门禁可跳过） |

2. **P1-7 的边界**：它与被删除的上一版 P0-1 同源（同三个红用例），但描述的是"测试与产品契约不一致"这一可单独验收的缺陷，因此按要求只删 P0-1 后予以保留。若也希望一并移出，请在锁定第 2 步时说明。
3. **本轮不含第 1 步评分**：`01-modules/<id>.md`、`01-score.md`、`01-score-check.md` 仍需按 rubric 由 `cv-cto` 产出；本文件只锁定交付范围。
4. **P0-4 的最小改动面**：读线程非阻塞/有界写入 + `wait_shutdown` 加超时 + 把 `handle_exit` 的 join 从 reaper 线程移开，是三条独立可验收的改动，可拆成同一 id 下的有限清单。

## 确认

| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 待确认（供给：删除上一版 P0-1、P1-4 后汇总剩余项） | v1 | 用户指令 2026-10-16 Round 9 |
| cv-pm | 待第 2 步 | v1 | — |
| cv-tech-lead | 待第 2 步 | v1 | — |
| cv-cto | 待第 2 步（只返回列表修订） | v1 | — |
