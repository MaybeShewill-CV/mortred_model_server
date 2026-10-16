# 交付队列

- 轮次: 10
- 版本: v1
- 性质: 长期根本修复（非 band-aid）；顺序锁定；无 packing/Docker
- 加权分上限: P1 上限 8.9（本轮无未关闭 P0）
- 审计基线: `aad689a2`（第 9 轮已关闭 4 个 P0；本轮 P1 各项已在该基线上重新定位行号）
- 来源: 第二次源码级评估的剩余项。**用户指令移出**上一版 P0-1（构建/门禁断裂）与上一版 P1-4（数值门禁可跳过），见附录
- 顺延说明: 本队列的 4 个 P0 已于第 9 轮合入（见附录"已由第 9 轮关闭"），本轮范围因此只剩 7 个 P1

## P0

（无。原 4 个 P0 已由第 9 轮关闭，见附录。）

## P1

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | 检测参数 `clip_boxes` 是无效开关，且与实现相反：解析并存储但无人使用，而 unmap 实际上恒定裁剪到源图边界 | 配置里写 `clip_boxes = true/false` | `clip_boxes` 只出现在 `detection_params.h:26`（声明）与 `:100-108`（解析），`src/`、`test/`、`conf/` 内**零使用者**；同时 `unmap_letterbox_bbox`（`request_geometry.h:127-144`）对每个框恒定夹到 `[0, source]`（`:139-142`），所以实际行为永远是"已裁剪"，该键既不能开启也不能关闭任何东西。**更正**：上一版记的"`min_box_area_px` 坐标空间随检测器变化"经逐行复核不成立——v5/v6/v7 与 v8 都经 `yolo_decode.h` 在网络空间过滤（`yolo_decode.h:24,59,118,172`；v8 在 `yolov8_detector.inl:75-76` 过滤、`:85` 才 unmap），`test/object_detection_output_contract_unittest.cc:297-312` 的用例名与代码一致 | 该键从 `DetectionParams` 与解析分支中删除（推荐，与"无使用者的能力不入契约"一致），或在 `docs/openapi.json` 与注释里改写为真实语义；随后 `clip_boxes` 在源码与文档中为 0 处，且检测参数契约不再包含无法生效的键。一次外部验收：`grep -rn clip_boxes src/ test/ conf/ docs/openapi.json` 为空，且检测相关单测全绿 | `src/models/object_detection/detection_params.h:26,100-108`；`src/models/backend/request_geometry.h:127-144`；`src/models/object_detection/yolo_decode.h:24,59,118,172`；`yolov8_detector.inl:75-85` |
| P1-2 | 管理/抓取凭据泄漏到每个被监管的模型进程 | supervisor 已拉起驻留集后检查任一模型进程的环境 | `supervisor_app.cpp:970` 全局 `setenv("MORTRED_API_TOKEN", auth_token_, 1)`；`spawn_locked` 把父进程整个 `environ` 拷给非 gateway 子进程（`supervisor.cpp:376`），不清理 `MORTRED_API_TOKEN` / `MORTRED_METRICS_TOKEN` / `MORTRED_GATEWAY_AUTH_TOKEN` → `/proc/<model_pid>/environ` 可读到管理 bearer | 模型进程 environ 中这三个键为 0 个，而 gateway 仍拿到它需要的 token，管理接口仍可用。一次外部验收：`tr '\0' '\n' < /proc/<model_pid>/environ \| grep -c MORTRED_API_TOKEN` = 0 | `src/control/supervisor/supervisor_app.cpp:970`；`src/control/supervisor.cpp:376` |
| P1-3 | 停止/重启只 kill 单 pid，没有进程组，孙进程存活 | 停掉或重启驻留集里的一个 id，而该进程 fork 过后代 | 只有单 pid `::kill`：`supervisor.cpp:597,603,727,898,907,923,931`；无 `setpgid`/`setsid`/`killpg`，孙进程在父进程被 SIGKILL 后继续存活 | 停止一个 id 后该 id 的所有后代进程在 10s 内消失。一次外部验收：脚本停止该 id 后断言 `ps -ef` 中无其残留进程（对应 rubric OP5） | `src/control/supervisor.cpp:597,603,727,898,907,923,931` |
| P1-4 | 请求路径上的 `CHECK` 断言会杀掉整个模型进程 | 会话输出形状或 dtype 与契约不一致 | `tensor.h:239`（`Tensor::make`）与 `:310`、`:316`（`data<T>()`）的 `CHECK`/`CHECK_EQ` 直接 abort：该 id 的所有在途请求同时丢失，只能靠 supervisor 重启恢复 | 构造形状/dtype 不一致的会话输出用例 → HTTP 返回 5xx（`MODEL_OUTPUT_CONTRACT_FAILED`）且进程存活。一次外部验收：跑该用例，退出码 0 且进程仍在 | `src/models/backend/tensor.h:239,310,316` |
| P1-5 | 无界等待与配置矛盾：`model_run_timeout<=0` 时请求永不回复 | `conf/server/**` 里写 `model_run_timeout=0` | 启动只打印警告（`server_runtime_config.h:153-157`）：worker checkout 变为无界等待、不创建请求定时器、`_m_waiting_jobs` 不回落 → 后续请求永久 429，`stop()` 可能挂起。注意同一文件里第 9 轮修的 `receive_timeout<=0` 已经是"夹到默认值 + 警告"（`:134-142`），两者语义不一致 | 该配置被拒绝启动，或按 `receive_timeout` 的既有做法夹到有界默认值并记录生效值，并有一条配置用例。一次外部验收：用 `model_run_timeout=0` 启动 → 非 0 退出，或 `/metrics`/启动日志显示生效值不等于 0，且请求仍会在有界时间内得到 504 | `src/server/server_runtime_config.h:134-142,151-157`；`src/server/base_server_impl.h` checkout 与定时器分支 |
| P1-6 | `/metrics` 的 method 标签取自未鉴权的原始请求方法，标签基数无界 | 未带 token、method 用随机长字符串的请求 | `base_server_impl.h:339,387,408,426,438,464,494` 用原始 `request_method` 作为指标 map key，永不裁剪；401 路径在鉴权前即可写入 | 未知 method 归一到固定标签集（或直接拒绝），压测后 `/metrics` 的 series 数有上界。一次外部验收：发 1000 个不同 method → series 数不增长 | `src/server/base_server_impl.h:339,387,408,426,438,464,494`；`src/server/prometheus_metrics.h` 标签 map |
| P1-7 | 三个契约测试与产品行为不一致（测试未随产品改动更新） | 跑 `request_envelope_unittest` / `server_e2e_contract_test` / `gateway_e2e_test` | ① `request_envelope_unittest.cc:43,57` 期望 base64 文本，而实现已在 item 生成处解码（`parsed_request.h:126-140`，2026-09-21 `d9a49675`；测试最后修改 2026-09-09）；② `server_e2e_contract_test.cc:713,742` 仍断言 `rate_limit_qps=1` 产生 429，而模型侧限流已移除；③ `gateway_e2e_test.cc:882` 期望"0.0.0.0 + 关闭限流"能启动，产品已 fail-closed 拒绝 | 三条用例与产品契约一致（改测试或改产品，并把 `rate_limit_qps` 的语义写进 conf schema 与 `docs/openapi.json`），逐个通过。一次外部验收：分别运行这三个二进制，0 失败 | `test/request_envelope_unittest.cc:43,57`；`src/server/parsed_request.h:126-140`；`test/server_e2e_contract_test.cc:713,742`；`test/gateway_e2e_test.cc:882` |

## P2

（无）

## 附录

| 类别 | 陈述 |
| --- | --- |
| 已由第 9 轮关闭（本队列原 4 个 P0） | `/ready` 改为 init-only（`b9bb8247`，merge `b7338dd0`）；体积检查前移到解析之前并覆盖 `/jobs`（`724c7140`，merge `aab361e2`）；整报文 `receive_timeout` 默认 60s 且可配（`c2f2876d`，merge `d99de1c3`）；supervisor 日志停顿不再挂死（`65ac1a4d`，merge `f3eab77e`）。外部复核：四个合并的 diff 与验收标准逐条相符，`prometheus_metrics_test` 7/7、`request_admission_unittest` 5/5、`server_runtime_config_unittest` 5/5、`log_buffer_unittest` 1/1 通过（`log_buffer_unittest` 需先重新 configure，目标才存在）。P0-4 残留：`handle_exit` 现在 close fd 后 **detach** joiner 线程，阻塞在 `read()` 的线程能否被唤醒在 Linux 上未定义（fd 号可能被复用），因此仍会按停顿子进程泄漏线程、该 id 重启需等 `spawn_locked` 的 10s 上界；`LogBuffer` 仍是阻塞写在单锁下（只是不再每行 flush）。建议后续按"有界队列 + 丢弃计数"收口 |
| 用户指定移出（不入列） | 上一版 P0-1：构建/门禁在 HEAD 上不成立（`packed_batch_nvi_unittest`、`cv_image_input_unittest` 链接失败 → `check` 中断 → 57/93 用例从未运行；3 处红；已在干净房间与 CI tests-only 两种配置下复现）。上一版 P1-4：数值门禁可跳过（`golden_process_exit` 部分通过返回 0、缺权重一律 skip、报告不写实际执行数） |
| rubric 明文范围外（需用户改 rubric 才能入列） | SAM／diffusion／CLIP／LightGlue／RTDETR／MOT／matting／`msocrnet`；Docker 镜像与默认运行用户；一个 catalog id 一个 OS 进程的结构与每进程约 370MB／61 线程基线；Windows/macOS 与其他架构；MNN 与 ORT 的吞吐；动态 shape／动态 batch；多机与 K8s |
| 范围外但客观成立（本轮不入列） | `pp_matting` alpha 用 `INTER_NEAREST` + `CV_32SC1`（与同族 modnet 的 `INTER_LINEAR` + `CV_8UC1` 不一致）→ 静默输出台阶状 alpha 边缘；CLIP `simple_tokenizer` 是贪心最长匹配而非 BPE，非 ASCII 输入直接 `TOKENIZE_UNKNOWN_TOKEN`；SAM encoder 把 BGR 送进 RGB 归一化系数 |
| 需用户决定是否扩范围 | 依赖树可复现（6 个 vendored 源中 5 个无 hash）与离线安装；产物随包分发 `LICENSE`/`NOTICE` 与被捆绑 TensorRT/cuDNN 的再分发许可、SBOM 覆盖二进制；内置 TLS（现由 `deploy/nginx` 终止）；审计日志（无格式约定） |
| 上一版候选 P2（未采用，留档） | golden 掩膜裸整数 JSON 533MB；`rate_limit_qps` 在 44 份配置中无效；死代码（`do_work` 等）；异步结果每次轮询深拷贝与无界保留；stuck worker 只有日志无指标；`model_config_file_path` 相对 CWD；无响应压缩；`test/contract_baseline/*.md` 无消费者；`/metrics` 被推理 token 挡住。原文在提交 `9977b57d` 的 `cowork/ledger/round-008/02-p0-p1-final.md` |

## 分类说明

1. **相对 round-010 v0（提交 `47b96321`）的变化**：4 个 P0 移入附录（已由第 9 轮关闭）；P0 节改为「无」；加权分上限由"P0 上限 7.9"改为"P1 上限 8.9"；P1 各项行号按 `aad689a2` 重新定位；附录补入第 9 轮关闭记录与外部复核结论。P1 的 7 条编号与内容未变。
2. **P1-7 的边界**：它与被用户移出的"门禁断裂"同源（同三个红用例），但描述的是"测试与产品契约不一致"这一可单独验收的缺陷，因此保留。若也要一并移出，请在锁定第 2 步时说明。
3. **本轮不含第 1 步评分**：`01-modules/<id>.md`、`01-score.md`、`01-score-check.md` 仍需按 rubric 由 `cv-cto` 产出；本文件只锁定交付范围。第 9 轮的 `index.md` 起点加权分为 7.9。
4. **执行前置**：本队列尚未经第 2 步三方确认。按 skill，`02-discussion/v1-{pm,tech-lead}.md`、`v1-cto-list.md`、`v1-sign-{cto,pm,tech-lead}.md` 齐备并通过，或用户直接锁定后，才从 P1-1 进入第 3 步。
5. **第 3 步开工前的更正（v1 → v1.1）**：P1-1 原陈述含一条不成立的判断——"`min_box_area_px` 坐标空间随检测器变化"。逐行复核（`yolo_decode.h:24,59,118,172`、`yolov8_detector.inl:75-85`、`test/object_detection_output_contract_unittest.cc:297-312`）确认 v5/v6/v7/v8 都在网络空间过滤，用例名与实现一致，故该判断撤销；P1-1 收敛为"`clip_boxes` 无效且与 unmap 恒定裁剪的实际行为相反"。凡本队列条目中源自外部审计转述、未经逐行复核的判断，一律以本文件当前文本为准。
6. **执行方式（用户选定）**：本轮为**单一执行方模式**——五个角色由通用子代理按角色卡提示词执行，流程记录中逐条标注"非 Cursor 角色 agent"，编排方（本次对话主代理）负责重跑 QA 命令、`--no-ff` 合入与流程记录书写。第 9 轮的实际执行方（Cursor 会话）已于 15:27 后停止活动，本轮在其空闲窗口内进行。

## 确认

| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 待确认（供给：删除上一版 P0-1、P1-4 后汇总剩余项；4 个 P0 已由第 9 轮完成） | v1 | 用户指令 2026-10-16：先写第 9 轮 → 冲突后顺延第 10 轮 |
| cv-pm | 待第 2 步 | v1 | — |
| cv-tech-lead | 待第 2 步 | v1 | — |
| cv-cto | 待第 2 步（只返回列表修订） | v1 | — |
