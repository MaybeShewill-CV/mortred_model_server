# 交付队列

- 轮次: 8
- 版本: v1
- 性质: 外部只读源码审计汇总（未读 README/docs/CHANGELOG）；按 rubric 的 P0/P1 定义排序；**待第 2 步三方确认，未锁定**
- 审计基线: `10d3a764`（Round 7 已全部合入）；每条"现在的行为"都在该基线上复验过
- 供给方式: 用户以外部审计结论供给本队列（同第 7 轮先例）

## P0

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | 仓库自带门禁在 HEAD 上不通过：契约套件恒红、一致性检查恒红、TSan 标签数与构建目标数不等 | 任何人 clone 后跑门禁；CI quality-gates 与 sanitizers 作业 | ① `_bin/server_e2e_contract_test` 45 例 **2 例 FAIL**（`rate_limited_returns_429`、`healthz_exempt_from_ip_rate_limit` 仍断言 `rate_limit_qps=1` 产生 429，而 `base_server_impl.h:340-342` 已把模型侧限流交给 gateway）。② `python3 scripts/check_consistency.py` 退出 1：`openapi_doc.h` 已是 runtime loader（`openapi_doc.h:8-11`），但 `gen_openapi.py:9,37,1138` 仍把它重写成 embedding 版本，检查器拿 loader 头与 `docs/openapi.json` 比对，结构上不可能相等。③ `test/CMakeLists.txt:348-354` 给 **7** 个目标打 `sanitizer` 标签，`ci.yml:328` 只构建 **6** 个，`ci.yml:335` 的 `ctest -L sanitizer` 必然出现 Not Run | 三条命令全绿：`python3 scripts/check_consistency.py` 退出 0；`./_bin/server_e2e_contract_test` 45/45；`ctest -L sanitizer` 无 Not Run 且执行用例数等于标签数。且 `rate_limit_qps` 的语义在 `conf` schema 与 `docs/openapi.json` 中与实现一致（拒绝该键，或明确标注为 no-op）。一次外部验收：重跑这三条命令 | 实测（本机 WSL，基线 `10d3a764`）：e2e 2/45 FAIL、consistency 退出 1；`ci.yml:328,335`；`test/CMakeLists.txt:348-354`；`scripts/gen_openapi.py:9,37,1138`；`src/server/base_server_impl.h:340-342` |
| P0-2 | `/ready` 在满载时返回 503，一次过载会把健康实例摘出入口 | 驻留集全部 worker 都在推理时探活（LB / `mortredctl doctor` / 外部探针） | `/ready` = `_m_successfully_initialized && _m_workers.available_approx() > 0`（`base_server_impl.h:353-357`），并在同一分支把 `mortred_ready` 置 0。"有空闲 worker"是容量信号，不是就绪信号：负载下探针持续拿到 503，健康实例被摘除 | 占满全部 worker 后 `GET /ready` 仍为 200；饱和只用独立 gauge 暴露（`mortred_workers_busy` 或新增 `saturated`），并写明该 gauge 的用法。一次外部验收：`http_infer_rps.py --raw`（并发 = 2×已部署 `worker_nums`）压测期间 `GET /ready` = 200，`/metrics` 饱和 gauge > 0 | `src/server/base_server_impl.h:353-357`；`src/server/prometheus_metrics.h` `set_ready` |
| P0-3 | 请求体大小限制在**解析之后**才生效，`/jobs` 完全没有限制 | 客户端提交超过 `request_size_limit` 的 body（含 chunked、含无 `Content-Length`） | `serve_process` 先按 Content-Type 解析 json/raw（`base_server_impl.h:399-414`），到 `:415` 才 `declared_body_exceeds`；`Content-Length` 缺失时该检查直接返回 false（`request_admission.h:54-65`）；async submit 路径无任何检查（`async_endpoints.h` `handle_async_submit`）。单请求内存放大 ≈ raw msgbuf + `decode_chunked_body` 拷贝 + rapidjson DOM + base64 输出，上限 64MB × 200 连接 | 超限 body 在读取/解析前即 413，进程 RSS 不随 body 线性增长；`/jobs` 同样 413；无 `Content-Length` 的 chunked 也受限。一次外部验收：body=128MB（limit=64）→ 413，且测试期间 RSS 增量 < 16MB；对 `/jobs` 重复同一验收 | `src/server/base_server_impl.h:399-415`；`src/server/request_admission.h:54-65`；`src/server/async_endpoints.h` submit 路径；`src/server/http_wire.h` `decode_chunked_body` 按值返回 |
| P0-4 | 整报文无超时：200 条滴流连接即可占满入口 | 慢速客户端每 14s 发一个字节，占住连接直至 `max_connections=200` 用尽 | `init_http_server` 从 `SERVER_PARAMS_DEFAULT` 起只覆盖 `max_connections`／`peer_response_timeout`／`request_size_limit`（`abstract_server.h:116-119`），`receive_timeout` 与 `keep_alive_timeout` 沿用库默认 -1 与 60s（`3rd_party/include/workflow/WFServer.h` 的 `SERVER_PARAMS_DEFAULT`）；`peer_response_timeout` 是每次读写超时，不覆盖整报文，因此滴流连接可长期存活 | 半截请求在文档化时限内被服务端关闭，连接数回落；并发滴流连接不阻止正常请求成功。一次外部验收：开 200 条滴流连接 → 正常推理请求仍 200，滴流连接在 T 秒内被关闭（T 写进 conf 并在错误信息里给出） | `src/server/abstract_server.h:105-122`；`3rd_party/include/workflow/WFServer.h` `SERVER_PARAMS_DEFAULT`（`receive_timeout=-1`、`keep_alive_timeout=60000`） |
| P0-5 | supervisor 在日志盘停顿时永久挂死：子进程不再回收、`stop` 无效、收 SIGTERM 后不退出 | 日志文件系统停摆（NFS/满盘），或某个子进程写满 stdout 后阻塞且忽略 SIGINT | `LogBuffer` 每行 flush（`log_buffer.h:59-66`）；读管道线程无超时；`handle_exit` 作为唯一 joiner 在 **reaper 线程**上 join 读线程（`supervisor.cpp:808-824`）；`wait_shutdown()` 的 `_cv.wait` 无超时（`supervisor.cpp:1006+`）。链条：子进程 write 阻塞 → 读线程阻塞持锁 → reaper 卡死 → 其余子进程全部不再回收 → `stop_server` 5s 后放弃 → SIGTERM 后 daemon 不退出 | 用一个"写满 stdout 且忽略 SIGINT"的假子进程复现：其余子进程仍被正常回收；`stop` 在 grace 后返回；supervisor 收 SIGTERM 后 30s 内退出。一次外部验收：脚本用 `test/fake_model_server` 变体 + `ps`/退出码断言 | `src/control/log_buffer.h:59-66`；`src/control/supervisor.cpp:311,563,808-824,929,1002,1006` |
| P0-6 | 覆盖集内 `libface` 的框位置系统性错误：按 /32 padding 预处理，却按未 padding 尺寸反解 | 任意非方形图走 `libface` HTTP 推理 | 预处理把图 padding 到 32 的倍数，但反解用 `context.network_size`（已 padding）与 `_m_input_size_host`（未 padding）混用，且不减 pad 偏移。实例：1920×1080 → 640×672 时 x 方向约 5% 偏移，随长宽比放大；每个框与关键点都偏 | 非方形图上的框与上游 YuNet 参考输出 IoU ≥ 0.9；新增固定夹具（含带 pad 偏移与非方形两种）的回归用例，修复前失败、修复后通过。一次外部验收：跑该用例 | `src/models/object_detection/libface_detector.inl:94-104` vs `:116-123`、`:207-213`；覆盖集 id `libface` |

## P1

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | 检测参数语义与实现不一致：`clip_boxes` 完全无效、`min_box_area_px` 坐标空间随检测器变化 | 请求或配置里设 `clip_boxes=true` / `min_box_area_px` | `clip_boxes` 被解析存储但全仓无使用者；`min_box_area_px` 在 v5/v6/v7 作用于网络空间、在 v8 作用于源图空间，而文档写的是"letterbox unmap 之前"；对应测试名断言的行为与代码相反，仅在 r=1 夹具下通过 | 同一 `min_box_area_px` 在覆盖集检测 id 上过滤掉的框数量一致；`clip_boxes=true/false` 在贴边目标上可观察到差异；`param_spec` 与 `docs/openapi.json` 列出这两个键（或明确删除）。一次外部验收：同一图同一参数在两个检测 id 上跑，比较过滤计数 | `src/models/object_detection/detection_params.h:22-26,100-109`；`yolov8_detector.inl:75-85`；`test/object_detection_output_contract_unittest.cc:297-312` |
| P1-2 | 管理/抓取凭据泄漏到每个被监管的模型进程 | supervisor 已拉起驻留集后检查任一模型进程环境 | `supervisor_app.cpp:970` 全局 `setenv("MORTRED_API_TOKEN", ...)`；`spawn_locked` 把父进程整个 `environ` 拷给非 gateway 子进程，不清理 `MORTRED_API_TOKEN` / `MORTRED_METRICS_TOKEN` / `MORTRED_GATEWAY_AUTH_TOKEN` → `/proc/<model_pid>/environ` 可读到管理 bearer | 模型进程 environ 中这三个键为 0 个，而 gateway 仍拿到其需要的 token，管理接口仍可用。一次外部验收：`tr '\0' '\n' < /proc/<model_pid>/environ \| grep -c MORTRED_API_TOKEN` = 0 | `src/control/supervisor/supervisor_app.cpp:969-971`；`src/control/supervisor.cpp:373-409` |
| P1-3 | 停止/重启只 kill 单 pid，没有进程组，孙进程存活 | 停掉或重启驻留集里的一个 id，而该进程 fork 过后代 | 只有 `::kill(child->pid, SIGINT\|SIGKILL)`，无 `setpgid`/`setsid`/`killpg`；孙进程（辅助进程、`nvidia-smi` 子进程、模型内部子进程）在父进程被 SIGKILL 后继续存活（对应完成标准 OP5） | 停止一个 id 后该 id 的所有后代进程在 10s 内消失。一次外部验收：脚本停止该 id 后 `ps -ef` 中无其残留进程 | `src/control/supervisor.cpp` 各 `::kill` 调用点；rubric OP5 |
| P1-4 | 数值门禁可被跳过：部分通过也返回 0，缺权重一律按 skip | 权重不全或 GPU 不可用的机器上跑 golden | ctest 把 exit 77 当 skip（`test/CMakeLists.txt` `SKIP_RETURN_CODE 77`）；`model_golden_test.cc` 的 `golden_process_exit` 在"部分通过 + 部分跳过"时返回 0；报告不写实际执行的数值用例数 | 报告中写出 `executed/selected`；`0 < executed < selected` 时退出码非 0；nightly rest 腿不再用 `--allow-skips` 掩盖。一次外部验收：故意只备一半权重跑 golden，报告出现执行数与非 0 退出码 | `test/model_golden_test.cc` `golden_process_exit`；`test/CMakeLists.txt` `SKIP_RETURN_CODE` |
| P1-5 | 请求路径上的 `CHECK` 断言会杀掉整个模型进程 | 会话输出形状或 dtype 与契约不一致 | `tensor.h:239` 与 `data<T>()` 的 `CHECK`/`CHECK_EQ` 直接 abort：该 id 的所有在途请求同时丢失，只能靠 supervisor 重启恢复 | 构造形状/dtype 不一致的会话输出用例 → HTTP 返回 5xx（`MODEL_OUTPUT_CONTRACT_FAILED`）且进程存活。一次外部验收：跑该用例，退出码 0 且进程仍在 | `src/models/backend/tensor.h:239,310,316` |
| P1-6 | 无界等待与配置矛盾：`model_run_timeout<=0` 时请求永不回复 | `conf/server/**` 写成 `model_run_timeout=0` | 启动只打印警告（`server_runtime_config.h:139-143`）；worker checkout 变为无界等待、不创建请求定时器、`_m_waiting_jobs` 不回落 → 后续请求永久 429，`stop()` 可能挂起 | 该配置被拒绝启动（或强制有界并记录生效值），并有一条配置用例。一次外部验收：用 `model_run_timeout=0` 启动 → 非 0 退出且错误信息给出可用范围 | `src/server/server_runtime_config.h:139-143`；`src/server/base_server_impl.h` checkout 与定时器分支 |
| P1-7 | `/metrics` 的 method 标签取自未鉴权的原始请求方法，标签基数无界 | 未带 token、method 用随机长字符串的请求 | 401/404/405 分支用原始 `request_method` 作为指标 map key，永不裁剪；401 路径在鉴权前即可写入 | 未知 method 归一到固定标签集（或拒绝），压测后 `/metrics` 的 series 数有上界。一次外部验收：发 1000 个不同 method → series 数不增长 | `src/server/base_server_impl.h` 401/404/405 分支；`src/server/prometheus_metrics.h` 标签 map |

## P2

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P2-1 | golden 掩膜用裸整数 JSON 存储，`test/golden` 共 533MB | clone、跑 golden、审阅 diff | 8 个 ~57MB 文件形如 `{"height":5616,"width":3744,"values":[3,3,3,...]}`；PNG 可压约 500×；解析时构造 ~2100 万元素 rapidjson DOM | 掩膜改 PNG/npz 后 `test/golden` < 50MB，现有 overlap 断言（IoU ≥ 0.90）不变。一次外部验收：`du -sh test/golden` 与 golden 用例结果 | `test/golden/*.json`；`test/model_golden_registry.h` `write_mask_golden`/`load_mask_golden` |
| P2-2 | `rate_limit_qps` 在 44 份 server 配置里存在但已无效果 | 运维读 conf、按注释推断限流行为 | 键被 schema 接受、被解析、随后 `(void)` 丢弃（`base_server_impl.h:127-128,208,342`），44 份 `conf/server/**` 仍在写它 | 该键从 schema 与配置中删除，或被显式标注为 no-op 并在 `docs/openapi.json` 说明。一次外部验收：跑配置 schema 测试 + grep 结果 | `src/server/base_server_impl.h:127-128,208,340-342`；`conf/server/**/*.toml` |
| P2-3 | 死代码：生产不可达的同步路径与无人调用的 helper | 代码审阅、加模型时误判入口 | `do_work` 仅测试调用；`http_wire.h` 的 `peer_ip_of`/`is_json_content_type`/`is_raw_body_content_type`/`reply_rate_limited` 零调用点；`rate_limiter.h` 仅单测使用 | 删除或在注释里写明保留理由；`src/server` 只剩一条同步执行图。一次外部验收：grep 调用点为空 | `src/server/base_server_impl.h` `do_work`；`src/server/http_wire.h`；`src/server/rate_limiter.h` |
| P2-4 | 异步结果每次轮询深拷贝，保留 100 条无大小上限且惰性回收 | 反复 `GET /jobs/{id}/result` | 每次轮询复制完整结果（图像类为 MB 级 base64）并重新序列化；保留集无大小界；只在下次 submit 时惰性回收 | 结果读取不再整份深拷贝；保留按字节有界，空闲时也能回收。一次外部验收：轮询大结果 N 次，RSS 与保留字节数有上界 | `src/server/async_job_table.h` 取结果与保留分支 |
| P2-5 | stuck worker / checkout 超时只有日志没有指标 | 单模型故障排查 | 卡死检测只 `LOG(ERROR)`/`LOG(FATAL)`，无计数器 | `/metrics` 有 checkout 超时与卡死计数，可告警。一次外部验收：制造一次 checkout 超时，指标 +1 | `src/server/worker_pool.h` 卡死检测分支；`src/server/prometheus_metrics.h` |
| P2-6 | 模型配置路径相对进程 CWD，换目录启动即失败 | 从非 `_bin` 目录、systemd、容器不同 WORKDIR 启动 | `generic_cv_server.h:147` 直接使用 TOML 里的 `model_config_file_path`，不锚定 project root；目前只靠 supervisor `chdir` 到 bin_dir 才成立（gate 侧 `trt_spawn_gate.h:72-84` 反而做了 root 解析） | 路径按 project root 解析，任意 CWD 启动成功。一次外部验收：`cd /tmp && <abs>/mortred-model-server.out --model mobilenetv2 <abs conf>` → ready | `src/server/generic_cv_server.h:147`；`src/control/trt_spawn_gate.h:72-84` |
| P2-7 | 忽略 `Accept-Encoding`，base64 图像响应无压缩 | 客户端请求 gzip | 全仓无 gzip/deflate 处理；图像响应为 base64 JSON，无压缩 | 支持 gzip（或在文档中明确不支持并给替代路径）。一次外部验收：带 `Accept-Encoding: gzip` 的响应 `Content-Encoding` 与体积 | 全仓 grep 无命中；`src/server/response_serializers.h` |
| P2-8 | `test/contract_baseline/*.md` 无任何消费者 | 兼容性审阅 | 两个文件被当作基线，但 `test/`、`scripts/`、`.github/` 中零引用 | 接入一个可执行兼容性用例，或删除。一次外部验收：grep 引用非空或文件不存在 | `test/contract_baseline/` |
| P2-9 | `/metrics` 被推理 token 挡住，空 token 的 loopback 部署无法抓取 | 本机 loopback 部署未设 `auth_token` | 只有 `/healthz`、`/ready`、`/openapi.json` 免鉴权；空 token 时 `is_bearer_authorized` 恒假 → `/metrics` 401 | 模型进程有自己的 scrape token（或在 loopback 上允许抓取），Prometheus 可抓。一次外部验收：不带推理 token 抓 `/metrics` 成功 | `src/server/base_server_impl.h:326-345`；`src/common/auth_token.h` 空 token 分支 |

## 附录

| 类别 | 陈述 |
| --- | --- |
| rubric 明文范围外（需用户改 rubric 才能入列） | CLIP 分词器不是 BPE、SAM AMG 并行解码数据竞争、SAM encoder 输入 BGR、LDM latent scale 取反、`pp_matting` alpha dtype/插值不一致 —— 与 SAM／diffusion／CLIP／LightGlue／RTDETR／MOT 同属范围外任务；容器以 root 运行与镜像默认运行用户；一个 catalog id 一个 OS 进程的结构与每进程约 370MB／61 线程基线；Windows/macOS 与其他架构；MNN 与 ORT 的吞吐；动态 shape／动态 batch；多机与 K8s；`msocrnet` |
| 已由第 7 轮关闭，不重复入列 | `ApiKeyManager::load` 解析 `max_queue`／`worker_budget`（已复验 `api_key_manager.h:200-204`）；`max_queue` residency 与 hard `worker_budget`；真 NVML GpuProbe 与 `shared_gpu_probe` 接线；删除 FixedWindow 产品成员与陈旧注释；`verify_gpu_gate_evidence` 入 CI／release；CI version 独立 fail-closed 步 |
| 需用户决定是否扩范围 | 依赖树可复现（6 个 vendored 源中 5 个无 hash，`toml++` 用行数校验）与离线安装；产物随包分发 `LICENSE`/`NOTICE` 与被捆绑的 TensorRT／cuDNN 再分发许可、SBOM 覆盖二进制；内置 TLS（现由 `deploy/nginx` 终止）；审计日志（无任何格式约定，先定格式才能验收） |
| 一次外部验收确认不了 | 无结构化日志与 trace 贯通；`min_box_area_px`／`clip_boxes` 之外的参数语义偏差清单（需按模型逐个定夹具后才能变成有限清单）；golden 容差是否过松（需参考实现对比才能判定，非本机可验收） |

## 模块归属

| id | rubric 模块 |
| --- | --- |
| P0-1 | ci、test |
| P0-2 / P0-3 / P0-4 | src-server-apps |
| P0-5 | src-control |
| P0-6 | src-models |
| P1-1 | src-models、conf |
| P1-2 / P1-3 | src-control |
| P1-4 | test、ci |
| P1-5 / P1-7 | src-server-apps、src-models |
| P1-6 | src-server-apps、conf |
| P2-1 / P2-3 / P2-4 / P2-5 / P2-8 | test、src-server-apps |
| P2-2 / P2-6 | conf、src-server-apps |
| P2-7 | src-server-apps |
| P2-9 | src-server-apps、ops |

## 分类说明（锁定前请确认）

1. **P0-1 的档位。** rubric 的 P0 是四条服务结果（调用发不出去、失败被报成成功、一次过载或单模型故障让整机入口不可用、崩溃后不按已写明策略恢复）。"门禁恒红"不直接落在四条里，但本项的后果落在其中两条的**可检测性**上：模型侧限流已移除而契约套件仍断言 429，等于"契约和产品不一致"；套件恒红后，ST／TP 各行的证据都不再可信。若锁定队列时按 rubric 严格判档，请降为 P1／P2；可观察变化不变。
2. **P0-6 的档位。** 依据是 rubric P0 的"失败被报成成功"：结果是错的，HTTP 仍 200。若认为"结果错"不等于"失败报成功"，请降为 P1。
3. **P0-7（原审计的 TRT pinned 内存错误路径）未入 P0。** 其触发条件是 TRT 调用失败后的早退路径，不满足四条 P0 中任何一条的验收方式；已并入 P1-5 同一类"请求路径不该以 abort／未定义行为结束"的证据面，不另列 id。
4. **本队列不含第 1 步评分。** 本文件只锁定交付范围；九份模块记录（`01-modules/<id>.md`）与 `01-score.md`、`01-score-check.md` 仍需按 rubric 由 `cv-cto` 产出。外部审计的模块分（`src-models` 5.5／`src-server-apps` 6.0／`src-control` 6.0／`src-factory-common` 6.0／`test` 7.0／`ci` 6.5／`ops` 7.5／`conf` 6.5／`docs` 未评）**不是** rubric 分：两者标定不同，不得直接写入模块记录。

## 确认

| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 待确认（本队列由外部审计结论供给） | v1 | 用户指令 2026-10-16 第 8 轮 |
| cv-pm | 待第 2 步 | v1 | — |
| cv-tech-lead | 待第 2 步 | v1 | — |
| cv-cto | 待第 2 步（只返回列表修订） | v1 | — |
