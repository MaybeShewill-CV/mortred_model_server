# 模块：ci

- 路径: `.github/`
- 权重: 5
- 分数: 5.0
- 基线: sub-agent-co-work @ `337810ff`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- `.github/workflows/` 三个工作流：`ci.yml`（953 行，4 类触发：push / pull_request / schedule / workflow_dispatch，`ci.yml:4-25`）、`codeql.yml`、`release.yml`；另有 `dependabot.yml`、`actionlint.yaml`、`pull_request_template.md`。
- `ci.yml` 的 job 划分清晰且各有职责：`tests`（`:54`）、`quality-gates`（`:147`）、`sanitizers`（`:285`）、`cpu-profile`（`:368`）及其后若干。
- **OpenAPI / 契约一致性在 CI 中被三处检查**：`quality-gates` 的 "Repository consistency checks" 跑 `scripts/check_version_single_source.py`、`scripts/verify_gpu_gate_evidence.py --manifest docs/ci/gpu_gate_evidence.example.json --allow-missing-artifacts`、`scripts/check_consistency.py`、`python3 scripts/gen_openapi.py --check`（`ci.yml:264-272`）；`cpu-profile` 另跑 `python3 scripts/check_contract_sync.py --dump-bin build-cpu/bin/contract_dump`（`ci.yml:455-460`）。
- `tests` job 用 `ctest` 汇总（`ci.yml:110-114`：`CTEST_OUTPUT_ON_FAILURE=1 cmake --build build-ci --target check -j$(nproc)`），并把 ctest 日志作为 artifact 上传（`:136-140`，`name: ctest-log-<compiler>`），失败时输出完整。
- 独立的内存检查门：`valgrind` 标签单独列出并运行（`ci.yml:121-125`：先 `ctest -N -L valgrind` 计数，再 `ctest -L valgrind --output-on-failure`），标签由 `test/CMakeLists.txt:498` 施加。
- 环境诚实性有明文：`ci.yml:45-47` 写明维护者 GPU smoke 是 **fail-closed**、"not a claim that the full zoo ran"、且从不在 fork PR 上调度；`:686-687` 写明托管 runner 无 GPU、故该 job 不含 `--profile gpu`；GPU 证据由 `scripts/verify_gpu_gate_evidence.py` + manifest 单独校验。
- 一个维护者专用矩阵 job 明确标为 `continue-on-error: true` 且只允许 `workflow_dispatch` + `inputs.full-matrix == 'true'`（`ci.yml:481-482`），不会把不确定性混进主门。

## 问题

1. **`sanitizers` job 的 TSAN 门按现状不可能通过（红门由构造决定）**：`test/CMakeLists.txt:349-355` 给**7 个**目标打上 `sanitizer` 标签（`async_job_unittest`、`async_job_stress_test`、`api_key_manager_unittest`、`do_work_lifetime_unittest`、`worker_pool_unittest`、`batch_collector_unittest`、**`rate_limit_gcra_unittest`**），注释还写着"the sanitizer CI gate runs exactly these"；但 `ci.yml:328` 只显式构建**前 6 个**，随后 `ci.yml:335` 用 `ctest --test-dir build-tsan -L sanitizer` 按**标签**选择测试。实测（编排方在三个已配置目录跑 `ctest -N -L sanitizer`）：`build`、`build/full-cpu`、`build/probe-cpu` 均登记 **7** 个带标签测试，含 `rate_limit_gcra_unittest`（Test #35/#36）。而 `rate_limit_gcra_unittest` 在**基础** `TEST_LIST`（`test/CMakeLists.txt:78`）中，故该配置下 `add_test` 一定生成、标签一定生效。后果：被选中但未构建的测试没有可执行文件，该步以失败告终——**TSAN 门是红的，数据竞争实际上没有被门控**；注释与实现不符。修法二选一：该步补建第 7 个目标，或摘掉它的标签（后者会静默缩小覆盖，须同时改注释）。
2. **`tests` job 同样不可能通过**：`ci.yml:114` 构建 `check` 目标，而 `test/CMakeLists.txt:348` 对 `TEST_LIST` 每一项都 `add_dependencies(check ${src})`；`packed_batch_nvi_unittest`（`:54`）与 `cv_image_input_unittest`（`:68`）都在 `TEST_LIST` 内，且二者目前**无法链接**。**编排方在基线 `ae6c5bac` 上直接复现（非转述）**，主检出 `build/`（Release、`MORTRED_BUILD_FULL=ON`、`MORTRED_BUILD_PROFILE=gpu`）上三个目标均 `build_exit=2`：
   - `packed_batch_nvi_unittest`：`undefined reference to jinq::models::backend::gpu_jpeg::GpuDecodeSlot::GpuDecodeSlot()` 与 `~GpuDecodeSlot()`（该目标的 `link.txt` 只链 `libcommon.so`，缺 `libmodels.so`，而符号在后者）；
   - `cv_image_input_unittest`：同一组未定义引用；
   - `enlightengan_output_contract_unittest`：**编译期**失败——`src/models/enhancement/enlightengan.inl:169,170: error: ‘named_tensor_as’ is not a member of ‘jinq::models::backend’; did you mean ‘NamedTensor’?`（缺 `model_runtime.h` 的 include 链）。
   P1-4 的 QA 另在 worktree 的 `build` 与 `build-full` 两个配置下复现过前两者的链接失败，第 6 轮审计在 `build/probe-ci`（CI tests-only 口径）亦复现，故该失败**跨配置成立**。**注：此条即用户在本轮明确移出队列的"构建/门禁不可用"类问题（原 P0-1 移入附录），故本次不新开条目**，仅作为 `ci` 模块的问题记录在案，供后续轮次取用。
3. 由此产生的整体信号问题：7 个 job 中至少 2 个结构性为红，团队会习惯性忽略 CI；`quality-gates` / `cpu-profile` 的门虽然内容扎实，但在一个恒红的流水线里不再具备"门"的意义。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| UX3（第一半：CI 仍检查 OpenAPI 与仓库一致） | **已满足** | `ci.yml:272` `python3 scripts/gen_openapi.py --check`；`ci.yml:460` `python3 scripts/check_contract_sync.py --dump-bin build-cpu/bin/contract_dump`；另有 `scripts/check_consistency.py`（`:271`）与版本单一来源检查（`:267`） |
| UX3（第二半：GPU 作业被跳过时不得写成 TensorRT 证明） | **已满足** | `ci.yml:45-47` 明文"fail-closed；not a claim that the full zoo ran"，GPU smoke 从不在 fork PR 调度；`:686-687` 写明托管 runner 无 GPU、该 job 不含 `--profile gpu`；GPU 证据由 `scripts/verify_gpu_gate_evidence.py` + `docs/ci/gpu_gate_evidence.example.json`（`:268-270`）单独校验，缺失时用 `--allow-missing-artifacts` 显式放行而不是当作证据 |
| UX1 / UX2（README、`docs/deployment.md`、`mortredctl next`） | 待编排方判定（不属本模块） | 归属 `docs` / `ops` 模块，应由那两份记录判定；本记录不代替 |
| PF/TP 系列 | 不适用（本模块） | 需 GPU 压测，属性能/吞吐行，由持有压测证据的模块判定 |

## 评分依据

- 给 5.0 而不是更高的理由：CI 是"能不能放进生产机"的交付门，而这个模块**两个 job 按现状无法变绿**（`tests` 因 `check` 依赖不可链接目标、`sanitizers` 因标签选择到未构建目标），TSAN 这条数据竞争门实际上没有在门控；在一个恒红的流水线里，内容扎实的 `quality-gates`、`cpu-profile` 也失去了门禁意义。按 rubric 的 10（"一个小组可以在这一台上线…"）衡量，当前 CI 不能被当作上线凭证。
- 给 5.0 而不是更低的理由：门的内容本身高于平均——OpenAPI/契约同步在 **3 处**被检查、valgrind 作为独立标签门、ctest 日志作为 artifact 留证、GPU 证据与托管 CI 的边界写得很诚实（fail-closed + 不在 fork PR 上调度 + 不用 `--profile gpu` 冒充）、维护者矩阵 job 显式 `continue-on-error` 且限定 `workflow_dispatch`。缺陷集中在"两个目标的链接/标签登记"这类**有限、可一次修好**的工程问题上，而非流程设计。
- 修好上述第 1、2 条后，本模块有充分理由升到 7 以上；反过来说，只要它还恒红，分数不应超过 6。
