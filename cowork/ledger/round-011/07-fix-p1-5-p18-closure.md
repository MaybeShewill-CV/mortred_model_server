# 修复收口：P1-5 / P1-6 / P1-7 / P1-8（+ P1-9 的探查结论）

- 基线: `sub-agent-co-work @ 4d56fa2d`
- 分支: `fix/p1-5-9-metadata`（5 个提交）
- 合入: **`636a3990`**（`--no-ff`，16 文件 +546/−43）
- 独立测试: **cv-qa 结论 `pass`**（26 行在范围内验收全绿，缺陷：无；6 条残余风险见 §4）
- 编排方合入后复核: 带断言的 `logs/_mergeqa_p159.sh` → **`SUMMARY FAILS=0`**

## 1. 四条已修 P1

| 项 | 问题 | 修法 | 验证 |
| --- | --- | --- | --- |
| **P1-5** | `_bin`/`_lib` 默认固定在仓库根，被**所有构建目录共享**：切 profile 会静默替换二进制，且正确 profile 的构建因"目标比对象新"而**不重链** | `CMakeLists.txt` 在配置期把 profile 写进 `_bin/.mortred-build-profile`；**冲突 profile 的配置直接 FATAL_ERROR**，报出戳路径与两种补救（独立输出目录 / 删戳） | 共享 `_bin` 已 gpu 时配置 cpu → 退出 1 且日志含 `already holds a gpu-profile build`、戳不变；加独立 BIN/LIB → 退出 0、自身戳 cpu、共享戳仍 gpu |
| **P1-8** | `src/common/mortred_version.h` 是 `configure_file` 生成物却被 git 跟踪，检出即 dirty | 生成到 `${CMAKE_BINARY_DIR}/generated/common/`，该目录加进 `mortred_common_includes`；`git rm --cached`；`.gitignore` 增加该文件与 `build*/` | `git ls-files` 为空、`src/` 下无该文件、`build/generated/...` 含 `MORTRED_VERSION "1.0.0"`、`mortred-supervisor.out` 内嵌该版本、reconfigure 后工作树仍干净 |
| **P1-6** | API key 用**无盐 SHA-256**，而示例教用户哈希人类口令（`your-secret-key`）——文件泄漏即可离线爆破 | 新增 `scripts/mint_api_key.sh`（32 字节随机密钥，打印一次；`--append` 以 mode 600 写配置）与 `scripts/check_api_keys_hashing.sh`（拒绝文档示例弱口令的哈希、拒绝非 64 小写 hex）；`mortredctl doctor` 跑该守卫；示例改为"不要哈希人类口令" | 两个 `--self-test` 均 `SUMMARY FAILS=0`；含示例弱口令哈希的临时配置被判 1；`--append` 产物 mode 600、guard 接受、无明文 |
| **P1-7** | `SKIP_RETURN_CODE 77` 让"跳过"与"通过"在汇总里不可区分 | `check` 目标产出 `--output-junit`（CMake ≥3.21）；新增 `scripts/ctest_summary.py` 打印 `CTEST SUMMARY total/passed/failed/skipped`，**未给 `--allow-skips` 时退出码 3**；CI 的 tests 与 cpu-profile 步骤显式 `--allow-skips` 并计数 | `--self-test` 绿；`build.make` 含 `output-junit`；真实 5 项子集 `total=5 passed=5 failed=0 skipped=0`；合成 1 个 skip → 严格 3、加开关 0 |

## 2. P1-9 的探查结论（A1–A3）——三次判定，其中一次推翻我自己的结论

FULL `check` 在本机第一次跑通时暴露了 4 个测试问题 + 1 个链接失败。链接失败与 3 个陈旧断言已修（见 §3）。剩下三项判定如下：

**A1 —— 我上一轮说"出厂配置有 8 组重复 `[*_SERVER]`，是真实配置缺陷"，这个结论是错的，更正为"陈旧断言"。**
- `server_section` 只在**该模型进程自己的配置文件内部**解析（`model_server_main.h:63` `config[server_section]`；`model_server_main.cpp:78-92` 还要求该段 `model` 等于目录项的 `model_section`），**没有任何代码跨文件按段名查找**。
- 控制面的要求是"**每个文件内**恰有一个 `[*_SERVER]` 段"（`catalog.cpp:87`），已满足。
- 因此 yolov8n/s/l/x 共用 `[YOLOV8_DETECTION_SERVER]`、depth vitb/vitl/vits 共用 `[DEPTH_ANYTHING_ESTIMATION_SERVER]` 是拆分配置后的**正常设计**。
- 改法（不是删断言，而是换成**运行时真正的不变量**，且更强）：新增 `ServerSectionsResolveToShippedConfigs` —— 每个 factory entry 的 `server_section` 必须被至少一个 `conf/server/**` TOML 声明，**且该段必须声明 `model = <entry.model_section>`**。旧规则完全不读 conf，段名改名或型号错配都漏检；新规则两条都覆盖。`model_catalog_unittest` 10/10。

**A2 —— 有意变更（测试陈旧）。**
- `09b0ed2b`："Same-family detectors previously diverged (v8 NMS-then-unmap). **Align on unmap-then-finalize**."；`ac515586` 把同一处理推广到 v5/v6/v7/nano/centerface/libface。
- 夹具计数 `3/2` → `2/1`（0.6→1、0.95→0 不变）：NMS 现在跑在源图坐标（各向异性反解 + 裁剪之后）。
- 改法：pin 更新为 `2/1/1/0`，并**加显式单调断言**（`at_0 ≥ at_04 ≥ at_06 ≥ at_095`），让性质不依赖具体数字。`object_detection_output_contract_unittest` 13/13。

**A3 —— 有意变更（测试陈旧）。**
- `ac515586`："pp_matting INTER_LINEAR+**CV_8UC1**"；代码注释（`pp_matting.inl:97-102`）"bilinear resize then CV_8UC1 matte, **matching modnet**"。
- 改法：期望 `CV_32SC1` → `CV_8UC1` 并注明出处。`src/factory/matting_task.h` 无任何 dtype 假设，故消费者不受影响；但这是**调用方可见的类型变更**（见 §4）。`model_output_contract_unittest` 10/10。

## 3. 同一轮修掉的其余 FULL-check 阻塞（P1-9 的残骸）

- `do_work_lifetime_unittest` 在 FULL 档**链接失败**（`gpu_jpeg::render_decode_metrics` 未定义）→ 补 `EXTRA_SRCS gpu_jpeg_decoder.cpp`（与前两个 jpeg 目标同一模式）。
- `server_runtime_config_unittest` 期望 `handler_threads=50`/`max_queue_depth=0`（r13 改为 8、r14 改为 clamp 到 32）→ 按新契约更新。
- `catalog_unittest` 的 `boots_cpu_profile` 期望 cpu 档非空（`ea22867d` 明确"Drop shipped cpu product configs"）→ 改为断言"按决策应为空"，并保留"cpu 过滤逻辑由合成用例覆盖"的说明。
- `gateway_dual_upstream_isolation_test` 夹具用 `conf/servers`（复数）、且写的是无 `[*_SERVER]` 段头的裸键 → gateway `routes: 0`、所有请求 404；改为 `conf/server` + 段头后 `routes: 2`、1/1 通过。

## 4. QA 列出的残余风险（我认可，原样记录）

1. **A1 无法绑定具体启动配置文件**：新规则证明"存在某份 conf TOML 声明该 section 且 model 匹配"，不能证明产品启动时传的就是那一份；启动期 `model_server_main.cpp:78-92` 仍会兜底拒绝缺段/错配。
2. **A2 夹具偏弱**：3 个候选中 2 个中心在 10×10 网络内已越界（x=40/70），反算+截断后贴边被 NMS 合并，所以 `2/1` 更多反映"截断后合并"而非真实尺度下的 NMS 顺序。**建议补一个完全在界内的重叠框用例**（未做）。
3. **pp_matting alpha 由 CV_32SC1 改为 CV_8UC1 是调用方可见变更**（与 modnet 对齐，`ac515586` 有意为之）；依赖 32S 的外部消费方需同步。
4. **A4 未做**：全量 101 项 `check` 仍未跑到完成（用户明确延后），`model_golden_test` 明令禁止运行 → **"整套是否全绿"仍是未证明状态**，只有 A4 能给它结论。本机 20 GB 内存下 `model_golden_test` 会把机器压到 OOM（实测 18.8/20 GB、swap 7.9/8 GB），需要串行/分区/加内存。
5. **doctor 在本机整体退出 1**（缺 product TensorRT 引擎）：本次只证明两个安全段**先于**引擎门禁、且凭据段为 `[ok]`，不证明 doctor 全绿。
6. **catalog 测试新增 `conf/server` 相对 cwd 依赖**：ctest 已设 `WORKING_DIRECTORY ${PROJECT_ROOT_DIR}`，但在其他目录直接跑该二进制会误报。

## 5. 编排方合入后复核（`SUMMARY FAILS=0`）

在 `636a3990` 上重跑：4 个脚本 `bash -n` + `py_compile`；`stop_order --check` 退出 0 且首行 `KillMode=mixed 为通过`；credentials / mint / api-key guard / ctest_summary 四个 `--self-test` 均 `SUMMARY FAILS=0`；`check` 目标含 `output-junit`；profile 戳为 `gpu`；生成头 untracked 且在 build 树内；8 个测试二进制全部退出 0（`occupancy_gate`、`server_runtime_config`、`catalog`、`gateway_dual_upstream`、`do_work_lifetime`、`model_catalog`、`model_output_contract`、`object_detection_output_contract`）；`ctest -R 'stop_order_record_rules|prometheus_credentials_check'` 2/2；工作树干净。
