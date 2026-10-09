# 技术方案

- 问题: P1-6
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

只收口这一条：无自托管 GPU runner 时，`gpu-pr-gate` 的结果是 `skipped`，`inference-paths` 不得再因此退出 0。改完后，该 required 检查以退出码 2 结束，并标明这不是 TensorRT/CUDA 证明。

`gpu-pr-gate` 仍只在 `vars.MORTRED_HAS_GPU_RUNNER == 'true'` 且同源 push 或同源 PR 时排队。未设变量时 job 保持 skipped，不改到 GitHub hosted，也不空等 runner。`cpu-profile` 成功条件不变。PR/push 的 T2 仍是 NOT RUN。

同源且该变量为 true、`gpu-pr-gate` 为 `success` 时，包装检查仍退出 0。`needs` 结果同为 `skipped` 的 fork 与未设变量走同一退出码，不另开 fork 专项。

不涉及 P1-5、D-1 至 D-6、已销账项。不改产品 toml，不改 HTTP，不改 tensor。一个 catalog id 仍是一个 OS 进程。后端仍是 MNN、ONNX Runtime、TensorRT。方案不含训练、标注、非 Linux。

## 做法

### 推荐

把 `inference-paths` 里决定退出码的内联分支抽成仓库脚本 `scripts/ci_eval_inference_paths.py`。该 job 在 `if: always()` 下照常跑完，用脚本退出码作为检查结论。`cpu=success` 且 `gpu=skipped` 时退出码固定为 2，标准输出有 `NOT_A_TRT_PROOF`，并打印「不是 TensorRT 或 CUDA 证明」。`cpu=success` 且 `gpu=success` 时退出码 0，标准输出有 `TRT_PROOF=yes`。`cpu` 不是 `success` 时退出码 1。`gpu` 为 `failure` 或 `cancelled` 时退出码 1。其余 gpu 结果退出码 1。

脚本同时打出现有 T0/T1/T2 摘要：PR/push 的 T2 固定为 NOT RUN；`skipped` 时 gpu-smoke 为 SKIPPED。`deploy-tools` 对该脚本做 `py_compile` 并跑 `--self-test`。`docs/ci-golden-regression.md` 里「无 runner 时 skip 仍算成功」的句子改成与退出码 2 一致。

代价：把检查名 `inference paths` 设为 required 的仓库，在 `gpu-pr-gate` 为 skipped 时该检查失败。CPU 证据仍看 `cpu-profile` 是否成功。退出码 2 表示没有 TRT 证明，退出码 1 表示 cpu 未成功或 GPU smoke 实际失败。验收在本机跑该脚本，不推 GitHub Actions，也不需要自托管 GPU runner。

### 否决

1. skipped 仍退出 0，只在日志里加「不是 TRT 证明」。required 检查仍然为绿，冻结的现在行为还在。现有 T2 的 “NOT a full-zoo claim” 已是这类句子，没有关掉这条劣势。
2. 去掉 `gpu-pr-gate` 的变量条件，让无 runner 时排队，或改到 `ubuntu-22.04` 上跑 smoke。无 runner 会空等，hosted 机器没有这条门要用的 CUDA/TensorRT。本机无法用仓库命令验收。

## 改动面

- `ci`：`.github/workflows/ci.yml` 的 `inference-paths` 判定步改为调用 `scripts/ci_eval_inference_paths.py`，并保留 `if: always()`。`deploy-tools` 的 `py_compile` 与 self-test 挂上该脚本。
- `ops`：新增 `scripts/ci_eval_inference_paths.py`。`scripts/ci_print_tier_summary.py` 在 gpu-smoke 为 skipped 时增加一句 `This result is not a TensorRT or CUDA proof.`
- `docs`：`docs/ci-golden-regression.md` 中无 `MORTRED_HAS_GPU_RUNNER`、以及 wrapper 把 skip 当成成功的那几句，改成退出码 2 与 `NOT_A_TRT_PROOF`。

## 明确不改

`src/models`、`src/server`、`src/apps`、`src/control`、`src/factory`、`src/common`、`conf/`、`test/`。`gpu-pr-gate`、`gpu-nightly-full`、`cpu-profile`、`t2-tier-notice` 的触发条件与测试内容。`gpu-pr-gate` 的 `runs-on` 仍是 `[self-hosted, X64, gpu]`。gateway `:8080`、supervisor `:8787`、一 catalog 一进程、产品 `type`、StatusCode、engine 生成、CHANGELOG、README。

## 契约

HTTP 与 tensor 契约不变。产品 toml 的 `type` 不变。

CI 结论契约改为：

| cpu | gpu | 退出码 | 必须出现的声明 |
| --- | --- | --- | --- |
| success | skipped | 2 | `NOT_A_TRT_PROOF` |
| success | success | 0 | `TRT_PROOF=yes` |
| success | failure 或 cancelled | 1 | `gpu-pr-gate ended as` |
| 非 success | 任意 | 1 | `cpu-profile must succeed` |
| success | 上表以外的值 | 1 | 不得出现 `TRT_PROOF=yes` 或 `NOT_A_TRT_PROOF` |

退出码 2 的标准输出同时含 gpu-smoke `SKIPPED`、T2 `NOT RUN`，以及 `This result is not a TensorRT or CUDA proof.` 退出码 2 的标准输出不得含 `TRT_PROOF=yes`。退出码 0 的标准输出不得含 `NOT_A_TRT_PROOF`。该步不得设 `continue-on-error`。

## 性能与资源

判定是现有 `inference-paths`（`ubuntu-22.04`，超时 5 分钟）里的一次 Python 进程。不配置 cmake，不编译，不调用 `trtexec`，不读权重，不生成 engine。无新增常驻进程，不占显存。`gpu-pr-gate` 在变量未设时仍 skipped，不为本条占用自托管 runner。

## 验收标准

1. `python3 scripts/ci_eval_inference_paths.py --cpu success --gpu skipped` 退出码为 2。标准输出含 `NOT_A_TRT_PROOF`、gpu-smoke 的 `SKIPPED`、T2 的 `NOT RUN`、`This result is not a TensorRT or CUDA proof.`，且不含 `TRT_PROOF=yes`。
2. `python3 scripts/ci_eval_inference_paths.py --cpu success --gpu success` 退出码为 0。标准输出含 `TRT_PROOF=yes` 与 gpu-smoke 的 `RAN`，且不含 `NOT_A_TRT_PROOF`。
3. `--cpu success --gpu failure` 与 `--cpu success --gpu cancelled` 退出码都是 1，标准错误含 `gpu-pr-gate ended as`。
4. `--cpu failure --gpu skipped` 与 `--cpu failure --gpu success` 退出码都是 1，标准错误含 `cpu-profile must succeed`。
5. `--cpu success --gpu neutral` 退出码为 1。
6. `python3 scripts/ci_eval_inference_paths.py --self-test` 退出码为 0。该命令在无 GPU、无 `trtexec`、无 GitHub 的环境下完成。
7. `python3 scripts/ci_eval_inference_paths.py --check-workflow` 退出码为 0。它判定 `.github/workflows/ci.yml` 的 `inference-paths` 调用该脚本、保留 `always()`、该 job 中不存在把 `skipped` 与 `success` 一起 `exit 0` 的分支、该步没有 `continue-on-error`，且 `gpu-pr-gate` 的 `if` 仍含 `vars.MORTRED_HAS_GPU_RUNNER == 'true'`。
8. `python3 scripts/ci_print_tier_summary.py --t0 unknown --t1 ran --t2 not_run --gpu-smoke skipped` 退出码为 0，标准输出含 `This result is not a TensorRT or CUDA proof.` `--gpu-smoke ran` 时退出码为 0，标准输出不含这句。
9. 上述命令不修改 `conf/` 与 `src/`。

## 测试义务

QA 按验收标准 1–8 记录退出码与声明字符串。白盒核对决策顺序：cpu 非 success 先退出 1，然后才处理 skipped 的退出码 2；`failure`/`cancelled` 不得走退出码 2。白盒核对 `--self-test` 会调用与 `--check-workflow` 相同的 workflow 断言。确认 `inference-paths` 未设 `continue-on-error`。确认 `conf/` 与 `src/` 无 diff。不推 GitHub Actions，不要求自托管 GPU runner。不测 P1-5 的 TRT 数值门，不测 D-1 至 D-6，不重测已销账项。

## 自测命令

```bash
python3 scripts/ci_eval_inference_paths.py --cpu success --gpu skipped
```

期望退出码 2。

```bash
python3 scripts/ci_eval_inference_paths.py --cpu success --gpu success
```

期望退出码 0。

```bash
python3 scripts/ci_eval_inference_paths.py --cpu success --gpu failure
python3 scripts/ci_eval_inference_paths.py --cpu success --gpu cancelled
python3 scripts/ci_eval_inference_paths.py --cpu failure --gpu skipped
python3 scripts/ci_eval_inference_paths.py --cpu success --gpu neutral
```

以上四条期望退出码都是 1。

```bash
python3 scripts/ci_eval_inference_paths.py --self-test
python3 scripts/ci_eval_inference_paths.py --check-workflow
python3 scripts/ci_print_tier_summary.py --self-test
python3 -m py_compile scripts/ci_eval_inference_paths.py
```

以上四条期望退出码都是 0。

## RD 不得自行决定

- skipped 的退出码固定为 2。不得改回 0，不得并进退出码 1。
- 声明字符串固定为 `NOT_A_TRT_PROOF` 与 `TRT_PROOF=yes`。
- cpu 非 success 时退出码固定为 1，先于 skipped 判定。
- fork 与未设变量只要 `needs` 结果同为 skipped，都是退出码 2。不得拆成两套结论。
- 不得删除或放宽 `gpu-pr-gate` 的 `MORTRED_HAS_GPU_RUNNER == 'true'`，不得改其 `runs-on`，不得去掉 `inference-paths` 的 `always()`，不得加 `continue-on-error`。
- 不得把 PR/push 的 T2 打成 RAN。
- 不得改产品 toml、HTTP、tensor、StatusCode、后端集合、一 catalog 一进程。
- 不得把验收改成必须推送 GitHub Actions 或必须有 GPU runner。
- 不得顺手实现 P1-5 或 D-1 至 D-6。
