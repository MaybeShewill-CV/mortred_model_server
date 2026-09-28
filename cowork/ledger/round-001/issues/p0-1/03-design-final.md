# 技术方案

- 问题: P0-1
- 版本: v2
- 对应列表: `/mnt/g/Codex/mortred_model_server/cowork/ledger/round-001/02-p0-p1-final.md`

## 边界

- 只收口 GPU/TRT **产品交付**链上的 engine 路径真源，以及 `mortredctl init` / `doctor` 默认不 fail-closed。
- 守住：一 catalog id 一进程；gateway `:8080`、supervisor `:8787`；后端仍为 MNN / ONNX Runtime / TensorRT；产品 toml 保持 `type=mnn` 或 `type=tensorrt`（已是 ONNX 的 id 与 LIBFACE 除外）；CUDA 12.x + TensorRT 10.x；engine 本机本卡 `prepare`；cpu profile 不链 TensorRT。
- 不改：进程树/HTTP 对外契约形态、训练与标注、非 Linux、P0-2、任一 P1（含往 pack 提交校准样本、主机 systemd 安装收口、文档路径统一）。

## 做法

### 推荐

**以 `conf/product_static_bs1.json` 为 GPU 产品 TRT 路径唯一真源，起服 toml / prepare 转换清单 / doctor 检查共用同一解析结果；遗留 `conf/trt_engines.json` 退出 OOB；`init`/`doctor` 默认对「未就绪」非 0 退出并点名缺失物。**

代价：改 conf 对齐与几条 ops 脚本/一致性校验；不改模型 C++ 推理路径。一次换机可沿 `init → prepare → doctor → 起 pack` 复述，且不会再因遗留清单建出错误 `.engine` 却仍显示可交付。

### 否决

1. **以遗留 `trt_engines.json` 为真源，把产品改回 `*.engine` + 旧 onnx**
   代价高，推翻已落地的 static_bs1 产品命名与 `prepare_pack` 已走的 `--write-manifest` 链；与冻结列表「改完后单一真源可被 prepare/doctor/起服一致解析」方向相反。

2. **只把 init/doctor 改成 fail-closed，不修真源分裂**
   实现便宜，但 `init` 仍提示裸跑 `convert_trt_engines.sh`（默认读遗留清单）时，fail-closed 只会拦「文件不存在」，拦不住「建错路径」；达不到「单一真源」。

## 改动面

- **真源与对齐（conf + 校验）**
  - 冻结规则：产品 catalog 的 onnx/engine 相对路径以 `conf/product_static_bs1.json` 为准。
  - 产品 model toml 的 `*.backend.model_file_path` 必须与表中 `engine` 一致（现有 `yolov8s.toml` 的 `*.engine.static_bs1.fp16` 方向保持）。
  - `conf/trt_engines.set_a.json` 必须与由产品表导出的 trtexec manifest 同路径（可保持手维，但须被 `product_static_bs1.py --check` 或等价门禁卡住漂移）。
  - `conf/onnx_sources.json` 中已进入产品 TRT 交付的条目（至少 YOLOV8）：`runtime_path` 与产品表 `engine` 对齐（消除 `fp16in` 与产品 `fp16` 分裂）；不借此改 `product_type` 约定。
  - `scripts/check_consistency.py`：产品 `type=tensorrt` engine 不得再强制登记进遗留 `trt_engines.json`；改为对照 `product_static_bs1`（及/或 set_a / 导出 manifest）。
- **OOB 入口（ops）**
  - `mortredctl_init.sh`：权重拉取失败 → 非 0，且不得打印成功口径的 `init done`；gpu profile 默认下一跳指向 `mortredctl prepare`（pack 作用域），不得再以「可选」口吻把裸 `convert_trt_engines.sh`（默认遗留清单）当主路径；gpu 下若当前 `MORTRED_PACK`（默认 demo 或文档约定的客户检测 pack，至少覆盖 `conf/packs/yolov8.toml`）缺 engine，init 结束须非 0 并点名缺哪份 engine。
  - `mortredctl_doctor.sh`：pack occupancy / pack TRT 门禁默认 fail-closed（不必 `--strict` 才失败）；失败输出须点名缺失的 engine 路径与缺失的 occupancy stamp（或明确的 calibrate 提示）；`--strict` 可保留给安全警告类，但不得再是「缺 engine/stamp 才失败」的唯一开关。
  - `mortredctl_prepare.sh` / `prepare_pack.sh`：继续只吃产品表导出的 manifest + `pack_trt` 解析的 toml 路径；保证 `prepare` 写出的 engine 路径 = doctor 检查路径 = 起服加载路径。
  - `convert_trt_engines.sh`：保留给显式 `--manifest`；文档/ OOB 文案标明默认遗留 `trt_engines.json` **不是**产品交付入口（cpu 构建不调用）。

## 明确不改

- `src/models`、`src/server`、`src/control` 进程模型与 spawn 语义（属 P0-2 的观测/拒启另案）。
- gateway/supervisor 端口与一 id 一进程。
- 产品 toml 的 `type=mnn` / `type=tensorrt` 约定（及既有 ONNX/LIBFACE 例外）。
- 往示例 pack 写入可复用的 `gpu_mem_mib` 样本（P1-2）。
- 主机 systemd/edge 安装收口（P1-3）、文档路径统一（P1-4）。
- 训练、标注、非 Linux、CPU profile 链接 TensorRT。

## 契约

- **路径契约（产品 GPU）**：同一 lowercase catalog id（例 `yolov8s`）在下列处解析出同一仓库相对 engine 路径：`product_static_bs1.json` → 产品 model toml → prepare 使用的 manifest → `pack_trt --check` / doctor。
- **HTTP / tensor**：不改 `/v1/models/{id}/infer`、任务信封与后端类型字段；仅消除「同 id 多路径」。
- **命令契约**：
  - `mortredctl init`：未就绪（权重失败，或 gpu 下当前 pack 缺 engine）→ 退出码 ≠ 0。
  - `mortredctl doctor`（无 `--strict`）：缺 pack TRT engine 或缺必需 occupancy stamp → 退出码 ≠ 0，且输出可定位到具体路径/字段。
  - `mortredctl prepare`：仍按现有 pack 作用域失败语义；成功后上述 doctor engine 门禁应对该 pack 变绿。
- **遗留清单**：`trt_engines.json` 不再作为产品 OOB 默认真源；不要求产品 static_bs1 路径出现在其中。

## 性能与资源

- 不改常驻进程数、显存策略或 engine 精度配方；仅保证本机本卡 `prepare` 落到与起服相同的文件。
- cpu profile：不触发 TRT convert、不以缺 engine 失败。
- engine 仍为机卡本地产物；不引入预置跨机 engine 分发。

## 验收标准

1. **AC1（单一路径）**：对 `yolov8s`，从 `conf/product_static_bs1.json`、`conf/model/object_detection/yolov8/yolov8s.toml`、`conf/trt_engines.set_a.json`（或 prepare 导出的同等 manifest）、`conf/onnx_sources.json` 的 YOLOV8 `runtime_path` 读出的 engine 相对路径字符串一致，且为 `*.engine.static_bs1.fp16`（不得再出现同 id 的 `fp16in` 或裸 `yolov8s.engine` 作为产品路径）。
2. **AC2（prepare 同路径）**：`mortredctl prepare --pack conf/packs/yolov8.toml --skip-ready`（或等价仅转换+检查）所针对的 engine 仓库相对路径与 AC1 相同；不得写入遗留 `yolov8s.engine`。
3. **AC3（doctor 缺 engine fail-closed）**：判定前将当前 pack 设为 `conf/packs/yolov8.toml`（例如设置 `MORTRED_PACK` 指向该文件）；在 AC1 路径上 engine 文件缺失或空文件时，无 `--strict` 执行 `mortredctl doctor`，退出码 ≠ 0，且输出出现该缺失路径（或等价「id + 路径」）。不得在未绑定该 pack、仍走默认 demo 的口径下宣称本条已通过。
4. **AC4（doctor 缺 stamp fail-closed）**：判定前将当前 pack 设为 `conf/packs/yolov8.toml`（例如设置 `MORTRED_PACK` 指向该文件）；对含 TensorRT backend 且策略要求 stamp 的该 pack，缺少 `gpu_mem_mib`（或门禁认定的 stamp）时，无 `--strict` 的 `mortredctl doctor` 退出码 ≠ 0，且输出指向 stamp/calibrate，不得仅 WARN 后继续当成功。不得在未绑定该 pack、仍走默认 demo 的口径下宣称本条已通过。
5. **AC5（init 权重失败 fail-closed）**：模拟/制造 `fetch_weights.py` 失败后跑 `mortredctl init --profile gpu`，退出码 ≠ 0，且不得出现可被理解为成功的 `init done` 收尾。
6. **AC6（init gpu 缺 engine）**：判定前将当前 pack 设为 `conf/packs/yolov8.toml`（例如设置 `MORTRED_PACK` 指向该文件）；权重可用但该 pack 缺 AC1 engine 时，无额外 `--strict` 语义下执行 `mortredctl init --profile gpu`，退出码 ≠ 0，并点名缺哪份 engine；成功文案不得声称已可交付 TRT 包。不得在未绑定该 pack、仍走默认 demo 的口径下宣称本条已通过。
7. **AC7（cpu 不链 TRT）**：`mortredctl init --profile cpu` 不以缺 TensorRT engine 失败（可因权重等其它门禁失败，但失败原因不得要求 TRT）。
8. **AC8（就绪闭环）**：在本机对本卡完成 `mortredctl prepare --pack conf/packs/yolov8.toml`（可按环境加 `--skip-ready` 若二进制缺失）且 occupancy 已满足门禁后，无 `--strict` 的 doctor 对 engine/stamp 门禁不再因二者失败；`pack_trt --check` 对该 pack 退出 0。

## 测试义务

- **QA（黑盒）**：逐条跑 AC1–AC8；至少用 `conf/packs/yolov8.toml` 覆盖客户检测 TRT 场景；记录退出码与输出中的路径/stamp 字样；确认 cpu profile 满足 AC7。
- **白盒**：核对 `product_static_bs1` / set_a / `onnx_sources` / 产品 toml 无同 id 多路径；确认 `check_consistency` 不再把产品 static_bs1 engine 绑死到遗留 `trt_engines.json`；确认 `init` 文案不再把裸 `convert_trt_engines.sh` 当默认产品入口。
- **回归边界**：不测 P0-2（token/autostart）、不测 P1 pack 样本提交、不测非 Linux。

## 自测命令

在仓库根目录执行（路径按本机调整；以下为仓库内既有入口）：

| 目的 | 命令 | 期望退出码 |
| --- | --- | --- |
| 产品表自洽 | `python3 scripts/product_static_bs1.py --check` | `0` |
| pack id 合法 | `python3 scripts/product_static_bs1.py --check-pack conf/packs/yolov8.toml` | `0` |
| 缺 engine 时 pack TRT 门禁 | （确保 AC1 路径无文件或为空后）`python3 scripts/pack_trt.py --project-root "$PWD" --pack conf/packs/yolov8.toml --check` | `≠0` |
| doctor 默认 fail-closed（缺 engine/stamp；AC3/AC4） | `MORTRED_PACK="$PWD/conf/packs/yolov8.toml" bash scripts/mortredctl_doctor.sh` | `≠0`，输出含缺失 engine 或 stamp |
| init 权重失败 | （使 `scripts/fetch_weights.py --profile gpu` 失败的前提下）`bash scripts/mortredctl_init.sh --profile gpu` | `≠0` |
| init gpu 缺 engine（AC6） | （权重可用且 AC1 engine 缺失时）`MORTRED_PACK="$PWD/conf/packs/yolov8.toml" bash scripts/mortredctl_init.sh --profile gpu` | `≠0`，输出点名缺失 engine |
| init cpu 不因 TRT | `bash scripts/mortredctl_init.sh --profile cpu` | 不以缺 `.engine*` 失败（权重齐全时可为 `0`） |
| prepare 写到真源路径 | `bash scripts/mortredctl_prepare.sh --pack conf/packs/yolov8.toml --skip-ready`（需 GPU/trtexec/onnx） | `0`；产物落在 AC1 路径 |
| prepare 后 engine 门禁绿 | `python3 scripts/pack_trt.py --project-root "$PWD" --pack conf/packs/yolov8.toml --check` | `0` |
| 一致性（含真源规则） | `python3 scripts/check_consistency.py` | `0`（改完后不得再因产品 static_bs1 路径不在遗留 json 而红） |
| pack_trt / occupancy 单测 | `python3 scripts/pack_trt.py --self-test`；`python3 scripts/pack_occupancy.py --self-test` | 均为 `0` |

## RD 不得自行决定

- 产品 GPU TRT 路径真源选哪份文件（本方案定为 `product_static_bs1.json`，不得改回以遗留 `trt_engines.json` 为 OOB 真源）。
- engine 产品命名（保持 `*.engine.static_bs1.fp16` / 表内 fp32 对应后缀；不得用 `fp16in` 或裸 `*.engine` 当产品路径）。
- `doctor`/`init` 对缺 engine/stamp（及 init 权重失败）是否默认 fail-closed（必须默认非 0；不得只靠 `--strict`）。
- 是否扩大到 P0-2、P1，或改动一 id 一进程、`:8080`/`:8787`、后端类型与产品 toml `type` 约定。
- cpu profile 是否链接或强制 TensorRT（否）。


## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v2 | `cowork/ledger/round-001/issues/p0-1/03-discussion/v2-sign-pm.md` |
| cv-rd | 同意 | 无 | v2 | `cowork/ledger/round-001/issues/p0-1/03-discussion/v2-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v2 | `cowork/ledger/round-001/issues/p0-1/03-discussion/v2-sign-tech-lead.md` |
