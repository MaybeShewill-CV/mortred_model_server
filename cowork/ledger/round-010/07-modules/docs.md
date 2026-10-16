# 模块：docs

- 路径: `docs/`、`README.md`、`README.zh-cn.md`
- 权重: 5
- 分数: 8.0
- 基线: sub-agent-co-work @ `0492a517`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- **31 份顶层文档，绝大多数是中英成对**：`about_model_configuration{,.zh-cn}.md`、`about_model_server_configuration{,.zh-cn}.md`、`api-contract{,.zh-cn}.md`、`api-keys{,.zh-cn}.md`、`async-jobs-customer-test{,.zh-cn}.md`、`benchmark-set-a{,.zh-cn}.md`、`deployment{,.zh-cn}.md`、`how_to_add_new_server{,.zh-cn}.md`、`model-developer-guide{,.zh-cn}.md`、`monitoring-guide{,.zh-cn}.md`、`onnx-interchange{,.zh-cn}.md`、`tutorials_of_model_servers{,.zh-cn}.md`、`unsupported-boundaries{,.zh-cn}.md`，另有 `openapi.json`、`contract_dump.json`、`repository-layout.md`、`ci-golden-regression.md`、`ui-redesign-gallery.md`。子目录：`benchmark-h3/`(39)、`ci/`(3)、`images/`(26)、`perf/`(4)、`ui-redesign/`(54)。
- **UX1 有机器检查且当前为绿**：`python3 scripts/check_readme_gpu_lines.py` **退出码 0**；`CMakePresets.json` 实有 presets 为 `tests-only`、`tests-only-werror`、`tests-only-tsan`、`tests-only-asan`、`full-cpu`、`full`、`full-werror`——README 的 CPU/GPU 源码构建行与其中的 `full`/`full-cpu` 一致（该检查脚本就是为此设立的）。
- **UX2 首小时路径在两种语言里都存在且命令真实**：`README.md:51`（"**First-hour path:** pick an install entry below, then run `mortredctl next`"）、`:66`（"the same `mortredctl doctor` acceptance gate"）、`:112`（`mortredctl next`）、`:118-119`（`mortredctl doctor` / `doctor --strict`）、`:129`、`:294`；`README.zh-cn.md:57`（"终点都是 `mortredctl doctor`"）、`:105-106`。两个命令都存在于 CLI 分派表（`src/apps/control/cli/cli_app.cpp:945-947` 列出 `doctor`、`next`），满足"文档中的命令在仓库里不存在 = 未满足"这一反条件。
- **UX4 的迁移指引写得很具体**：`docs/api-contract.md:77` 明写被删除字段（含 `img_data`）回答 **422**；`:114-115` 给出示例体 `"pointer": "/img_data"` 与 `"message": "field 'img_data' was removed; use images: [\"<base64>\"] (migration: img_data -> images[0])"`；`:224` 的错误码表把 66/422 与 `img_data`、未知键、坏 `params` 并列。这正是"错误必须指出该用哪个字段"的要求。
- **引擎参照与 GPU 证据有专门落点**：`docs/benchmark-h3/`（39 份，作 trtexec Throughput 参照）、`docs/ci/`（3 份，含 `gpu_gate_evidence.example.json` 的 manifest 语义）、`docs/perf/`(4)。
- **范围诚实**：`unsupported-boundaries{,.zh-cn}.md` 明确写出不支持面；`monitoring-guide.md` 与 `deploy/` 的 prometheus/grafana 资产对应。
- **本轮改动**：P1-5 改了两份 `about_model_server_configuration*`（各 1 行），删除了中文版"`-1` 代表该值无限大"这一**与实现相反的表述**——即本模块的一处实际错误在本轮被修掉。

## 问题

1. **三份文档谈同一件事（加模型）**：`model-developer-guide{,.zh-cn}.md`、`how_to_add_new_server{,.zh-cn}.md`、`tutorials_of_model_servers{,.zh-cn}.md` 覆盖面重叠，完成标准 EX1 要求它们与当前 `catalog()`、`mortred-model-server.out --list` 一致——重叠意味着**三处都要同步**，任一处漂移都会让 EX1 半真半假。
2. **一半以上的文档体量是设计与图片存档**：`docs/ui-redesign/`(54) + `docs/images/`(26) 合计 80 个文件，超过顶层操作文档数量。它们对"上线这台机器"没有直接价值，却与操作文档混在同一模块下，且显然不在任何 CI 检查范围内。
3. **`docs/contract_dump.json` 是生成物但入库**：与本轮 `src/factory-common` 记录的 `mortred_version.h` 同类问题——生成物入库会带来"工作区恒 dirty"或"忘记重新生成"的风险；它的正确性目前靠 `scripts/check_contract_sync.py` 与 `openapi_consistency_test` 兜（这两道门在 CI 里，属 `ci` 模块的已满足项）。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| UX1（中英 README 的 CPU/GPU 构建命令不得互相矛盾，且与 `CMakePresets.json` 的 `full`/`full-cpu` 一致） | **已满足** | `python3 scripts/check_readme_gpu_lines.py` 退出码 **0**；`CMakePresets.json` 实有 `full` 与 `full-cpu` presets（编排方实读 preset 列表） |
| UX2（README 或 `docs/deployment.md` 的首小时路径能从 `mortredctl next` 走到 `doctor`；文档中的命令必须在仓库里存在） | **已满足** | 两个 README 都给出该路径（`README.md:51,66,112,118-119,129`、`README.zh-cn.md:57,105-106`）；`next` 与 `doctor` 均在 `cli_app.cpp:945-947` 的分派表中，且 `scripts/mortredctl_next.sh`/`mortredctl_doctor.sh` 提供同名包装 |
| UX4（覆盖集任一 id 都要有与 `docs/openapi.json` 一致的请求/响应示例；已删除字段须返回 422 并指出该用哪个字段） | **已满足（示例一致性由 CI 门兜底）** | 迁移指引见 `docs/api-contract.md:77,114-115,224`（`/img_data` 指针 + "use images: [...]" + 错误码表 66/422）；"每个 id 的示例与 openapi 一致"由 `scripts/gen_openapi.py --check`、`scripts/check_contract_sync.py`、`test/openapi_consistency_test.cc` 三处门覆盖（`ci.yml:272`、`:460`、`:173-176`），本轮 `ci` 模块记录已判该门为已满足 |
| EX1（`model-developer-guide.md` 或 `how_to_add_new_server.md` 不得要求为每个模型单独做服务器二进制，且步骤要与 `catalog()`、`--list` 一致） | **待编排方判定** | 机制侧已在 `src-factory-common` 记录中活体验证（`--list` 44 个 id、网关零硬编码 id）；本记录未逐字通读三份"加模型"文档并比对步骤，故不代替判定。若下一轮要做，验收很轻：按文档新增一个 id 的步骤实际执行一遍，`--list` 应出现该 id |

## 评分依据

- 给 8.0 而不是更高：① "加模型"这一主题散在三份（中英各三）文档里，任何一致性要求都要三处同步，EX1 的可靠性因此打折；② `ui-redesign/`(54) 与 `images/`(26) 让模块的"文档面"里超过一半是存档而非操作说明，查找成本上升；③ `docs/contract_dump.json` 是入库生成物（同类问题本轮已作为 P2 记录在 `src/factory-common`）。
- 给 8.0 而不是更低：① **中英成对**是硬功夫，且本轮实际修掉了中文版一处与实现相反的表述（`-1` 无限大）；② 构建行有一道**专用检查脚本**且当前为绿（UX1）；③ 首小时路径在两种语言里都完整，且文档里出现的命令在仓库里真实存在（UX2）；④ 迁移指引具体到"指针 + 该用哪个字段 + 错误码"，并有 CI 门兜底其与 openapi 的一致性（UX4）；⑤ 有 `unsupported-boundaries` 与 `monitoring-guide` 这类"把边界与运维写清楚"的文档，符合 rubric 对 10 的描述里"讲清一次失败"的方向。
- 参照系：把三份"加模型"文档合并成一份并按 `catalog()` 重写步骤、把 `ui-redesign/`/`images/` 移出操作文档面之后，本模块有理由到 8.5 以上。
