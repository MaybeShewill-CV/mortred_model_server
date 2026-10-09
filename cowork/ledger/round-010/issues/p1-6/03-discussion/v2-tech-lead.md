# 技术方案

- 问题: P1-6
- 版本: v2
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

只收口这一条：无自托管 GPU runner 时，`gpu-pr-gate` 的结果是 `skipped`，`inference-paths` 不得再因此退出 0。改完后，该 required 检查以退出码 2 结束，并标明这不是 TensorRT/CUDA 证明。

`gpu-pr-gate` 仍只在 `vars.MORTRED_HAS_GPU_RUNNER == 'true'` 且同源 push 或同源 PR 时排队。未设变量时 job 保持 skipped，不改到 GitHub hosted，也不空等 runner。`cpu-profile` 成功条件不变。PR/push 的 T2 仍是 NOT RUN。

同源且该变量为 true、`gpu-pr-gate` 为 `success` 时，包装检查仍退出 0。`needs` 结果同为 `skipped` 的 fork 与未设变量走同一退出码，不另开 fork 专项。

判定步有没有把 `needs.cpu-profile.result` 与 `needs.gpu-pr-gate.result` 传给脚本，对照 `.github/workflows/ci.yml` 原文。`--check-workflow` 退出码 0 不作为这条接线的通过条件。

不涉及 P1-5、D-1 至 D-6、已销账项。不改产品 toml，不改 HTTP，不改 tensor。一个 catalog id 仍是一个 OS 进程。后端仍是 MNN、ONNX Runtime、TensorRT。方案不含训练、标注、非 Linux。

## 做法

### 推荐

把 `inference-paths` 里决定退出码的内联分支抽成仓库脚本 `scripts/ci_eval_inference_paths.py`。该 job 在 `if: always()` 下照常跑完，用脚本退出码作为检查结论。判定步在 `set -euo pipefail` 之后只调用该脚本，实参用单引号写成：

```yaml
python3 scripts/ci_eval_inference_paths.py \
  --cpu '${{ needs.cpu-profile.result }}' \
  --gpu '${{ needs.gpu-pr-gate.result }}'
```

`cpu=success` 且 `gpu=skipped` 时退出码固定为 2，标准输出有 `NOT_A_TRT_PROOF`，并打印「不是 TensorRT 或 CUDA 证明」。`cpu=success` 且 `gpu=success` 时退出码 0，标准输出有 `TRT_PROOF=yes`。`cpu` 不是 `success` 时退出码 1。`gpu` 为 `failure` 或 `cancelled` 时退出码 1。其余 gpu 结果退出码 1。

脚本同时打出现有 T0/T1/T2 摘要：PR/push 的 T2 固定为 NOT RUN；`skipped` 时 gpu-smoke 为 SKIPPED。`deploy-tools` 对该脚本做 `py_compile` 并跑 `--self-test`。脚本可以保留 `--check-workflow`，并且 `--self-test` 调用与它相同的原文断言；该退出码 0 不代替验收第 7 条。

`docs/ci-golden-regression.md` 里 “The wrapper treats that skip as success only when `cpu-profile` succeeded.” 改为这一句：`When MORTRED_HAS_GPU_RUNNER is unset, the wrapper exits 2 and prints NOT_A_TRT_PROOF. This result is not a TensorRT or CUDA proof.`

代价：把检查名 `inference paths` 设为 required 的仓库，在 `gpu-pr-gate` 为 skipped 时该检查失败。CPU 证据仍看 `cpu-profile` 是否成功。退出码 2 表示没有 TRT 证明，退出码 1 表示 cpu 未成功或 GPU smoke 实际失败。验收在本机跑该脚本，并对照仓库里的 `ci.yml` 与上述文档。不推 GitHub Actions，也不需要自托管 GPU runner。

### 否决

1. skipped 仍退出 0，只在日志里加「不是 TRT 证明」。required 检查仍然为绿，冻结的现在行为还在。现有 T2 的 “NOT a full-zoo claim” 已是这类句子，没有关掉这条劣势。
2. 去掉 `gpu-pr-gate` 的变量条件，让无 runner 时排队，或改到 `ubuntu-22.04` 上跑 smoke。无 runner 会空等，hosted 机器没有这条门要用的 CUDA/TensorRT。本机无法用仓库命令验收。
3. 验收只跑 `--check-workflow` 并看退出码 0。现场对照不到 `needs` 结果有没有传入 `--cpu` 与 `--gpu`。

## 改动面

- `ci`：`.github/workflows/ci.yml` 的 `inference-paths` 判定步改为调用 `scripts/ci_eval_inference_paths.py`，把 `needs.cpu-profile.result` 传给 `--cpu`，把 `needs.gpu-pr-gate.result` 传给 `--gpu`，并保留 `if: always()`。该步不写 `continue-on-error`。`deploy-tools` 的 `py_compile` 与 self-test 挂上该脚本。
- `ops`：新增 `scripts/ci_eval_inference_paths.py`。`scripts/ci_print_tier_summary.py` 在 gpu-smoke 为 skipped 时增加一句 `This result is not a TensorRT or CUDA proof.`
- `docs`：`docs/ci-golden-regression.md` 中无 `MORTRED_HAS_GPU_RUNNER`、以及 wrapper 把 skip 当成成功的那一句，改成退出码 2、`NOT_A_TRT_PROOF`，并写明这不是 TensorRT 或 CUDA 证明。

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

`inference-paths` 判定步把 `needs.cpu-profile.result` 作为 `--cpu` 的实参，把 `needs.gpu-pr-gate.result` 作为 `--gpu` 的实参。该步退出码等于脚本退出码。

## 性能与资源

判定是现有 `inference-paths`（`ubuntu-22.04`，超时 5 分钟）里的一次 Python 进程。不配置 cmake，不编译，不调用 `trtexec`，不读权重，不生成 engine。无新增常驻进程，不占显存。`gpu-pr-gate` 在变量未设时仍 skipped，不为本条占用自托管 runner。

## 验收标准

1. `python3 scripts/ci_eval_inference_paths.py --cpu success --gpu skipped` 退出码为 2。标准输出含 `NOT_A_TRT_PROOF`、gpu-smoke 的 `SKIPPED`、T2 的 `NOT RUN`、`This result is not a TensorRT or CUDA proof.`，且不含 `TRT_PROOF=yes`。
2. `python3 scripts/ci_eval_inference_paths.py --cpu success --gpu success` 退出码为 0。标准输出含 `TRT_PROOF=yes` 与 gpu-smoke 的 `RAN`，且不含 `NOT_A_TRT_PROOF`。
3. `--cpu success --gpu failure` 与 `--cpu success --gpu cancelled` 退出码都是 1，标准错误含 `gpu-pr-gate ended as`。
4. `--cpu failure --gpu skipped` 与 `--cpu failure --gpu success` 退出码都是 1，标准错误含 `cpu-profile must succeed`。
5. `--cpu success --gpu neutral` 退出码为 1。
6. `python3 scripts/ci_eval_inference_paths.py --self-test` 退出码为 0。该命令在无 GPU、无 `trtexec`、无 GitHub 的环境下完成。
7. 对照 `.github/workflows/ci.yml` 原文判定，不得只凭 `--check-workflow` 退出码 0：`inference-paths` 的判定步调用 `scripts/ci_eval_inference_paths.py`，并把 `needs.cpu-profile.result` 与 `needs.gpu-pr-gate.result` 传给 `--cpu` 与 `--gpu`；该 job 的 `if` 含 `always()`；该步没有 `continue-on-error`；该 job 中不存在把 `skipped` 与 `success` 一起 `exit 0` 的分支；`gpu-pr-gate` 的 `if` 仍含 `vars.MORTRED_HAS_GPU_RUNNER == 'true'`。本条通过当且仅当「自测命令」第 7 条每一条检索的退出码都等于写明的期望。`--check-workflow` 退出码 0 不算本条通过。
8. `python3 scripts/ci_print_tier_summary.py --t0 unknown --t1 ran --t2 not_run --gpu-smoke skipped` 退出码为 0，标准输出含 `This result is not a TensorRT or CUDA proof.` `--gpu-smoke ran` 时退出码为 0，标准输出不含这句。
9. 上述命令不修改 `conf/` 与 `src/`。
10. `docs/ci-golden-regression.md` 不再写无 `MORTRED_HAS_GPU_RUNNER` 时 wrapper 把 skip 当成成功，改为退出码 2 与 `NOT_A_TRT_PROOF`，并写明这不是 TensorRT 或 CUDA 证明。原文不含 `The wrapper treats that skip as success`，也不含 `skip as success`；原文含 `NOT_A_TRT_PROOF`、`exits 2` 与 `This result is not a TensorRT or CUDA proof.` 本条通过当且仅当「自测命令」第 10 条每一条检索的退出码都等于写明的期望。

## 测试义务

QA 按验收标准 1–6、8、9 记录退出码与声明字符串。按第 7 条与第 10 条记录每条检索命令的退出码。白盒核对决策顺序：cpu 非 success 先退出 1，然后才处理 skipped 的退出码 2；`failure`/`cancelled` 不得走退出码 2。白盒核对 `--self-test` 会调用与 `--check-workflow` 相同的 workflow 断言，且该断言包含 `--cpu` / `--gpu` 接到上述两个 `needs` 结果；这两条命令的退出码 0 不代替第 7 条。确认 `inference-paths` 未设 `continue-on-error`。确认 `conf/` 与 `src/` 无 diff。不推 GitHub Actions，不要求自托管 GPU runner。不测 P1-5 的 TRT 数值门，不测 D-1 至 D-6，不重测已销账项。

## 自测命令

每条单独执行。管道的退出码取最后的 `rg`。检索无命中时期望退出码是 1。退出码 2 表示检索命令本身失败，对应验收条失败。

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
python3 scripts/ci_print_tier_summary.py --self-test
python3 -m py_compile scripts/ci_eval_inference_paths.py
```

以上三条期望退出码都是 0。

```bash
python3 scripts/ci_eval_inference_paths.py --check-workflow
```

期望退出码 0。这条不作为第 7 条的通过条件。

第 7 条，对照 `.github/workflows/ci.yml`。以下五条期望退出码都是 0：

```bash
awk '
  /^  inference-paths:/ { p = 1 }
  p && /^  [A-Za-z0-9-]+:/ && !/^  inference-paths:/ { exit }
  p { print }
' .github/workflows/ci.yml | rg -n -F "python3 scripts/ci_eval_inference_paths.py"
```

```bash
awk '
  /^  inference-paths:/ { p = 1 }
  p && /^  [A-Za-z0-9-]+:/ && !/^  inference-paths:/ { exit }
  p { print }
' .github/workflows/ci.yml | rg -n -F -- "--cpu '\${{ needs.cpu-profile.result }}'"
```

```bash
awk '
  /^  inference-paths:/ { p = 1 }
  p && /^  [A-Za-z0-9-]+:/ && !/^  inference-paths:/ { exit }
  p { print }
' .github/workflows/ci.yml | rg -n -F -- "--gpu '\${{ needs.gpu-pr-gate.result }}'"
```

```bash
awk '
  /^  inference-paths:/ { p = 1 }
  p && /^  [A-Za-z0-9-]+:/ && !/^  inference-paths:/ { exit }
  p { print }
' .github/workflows/ci.yml | rg -n -F -- "if: \${{ always()"
```

```bash
awk '
  /^  gpu-pr-gate:/ { p = 1 }
  p && /^  [A-Za-z0-9-]+:/ && !/^  gpu-pr-gate:/ { exit }
  p { print }
' .github/workflows/ci.yml | rg -n -F -- "if: \${{ vars.MORTRED_HAS_GPU_RUNNER == 'true'"
```

第 7 条，以下两条期望退出码都是 1：

```bash
awk '
  /^  inference-paths:/ { p = 1 }
  p && /^  [A-Za-z0-9-]+:/ && !/^  inference-paths:/ { exit }
  p { print }
' .github/workflows/ci.yml | rg -n -F "continue-on-error"
```

```bash
awk '
  /^  inference-paths:/ { p = 1 }
  p && /^  [A-Za-z0-9-]+:/ && !/^  inference-paths:/ { exit }
  p { print }
' .github/workflows/ci.yml | rg -n -U --multiline-dotall 'success[\s\S]*skipped[\s\S]*exit 0|skipped[\s\S]*success[\s\S]*exit 0'
```

第 10 条，以下三条期望退出码都是 0：

```bash
rg -n -F "NOT_A_TRT_PROOF" docs/ci-golden-regression.md
```

```bash
rg -n -F "exits 2" docs/ci-golden-regression.md
```

```bash
rg -n -F "This result is not a TensorRT or CUDA proof." docs/ci-golden-regression.md
```

第 10 条，以下两条期望退出码都是 1：

```bash
rg -n -F "The wrapper treats that skip as success" docs/ci-golden-regression.md
```

```bash
rg -n -i -F "skip as success" docs/ci-golden-regression.md
```

## RD 不得自行决定

- skipped 的退出码固定为 2。不得改回 0，不得并进退出码 1。
- 声明字符串固定为 `NOT_A_TRT_PROOF` 与 `TRT_PROOF=yes`。
- cpu 非 success 时退出码固定为 1，先于 skipped 判定。
- fork 与未设变量只要 `needs` 结果同为 skipped，都是退出码 2。不得拆成两套结论。
- 不得删除或放宽 `gpu-pr-gate` 的 `MORTRED_HAS_GPU_RUNNER == 'true'`，不得改其 `runs-on`，不得去掉 `inference-paths` 的 `always()`，不得加 `continue-on-error`。
- `--cpu` 的实参固定为 `'${{ needs.cpu-profile.result }}'`，`--gpu` 的实参固定为 `'${{ needs.gpu-pr-gate.result }}'`。不得改成别的传参方式，不得只把 `needs` 结果打进日志。
- 不得用 `--check-workflow` 退出码 0 代替对照 `ci.yml` 原文。
- 不得在 `docs/ci-golden-regression.md` 保留无 `MORTRED_HAS_GPU_RUNNER` 时 wrapper 把 skip 当成成功的句子。替换句必须同时有 `exits 2`、`NOT_A_TRT_PROOF` 与 `This result is not a TensorRT or CUDA proof.`
- 不得把 PR/push 的 T2 打成 RAN。
- 不得改产品 toml、HTTP、tensor、StatusCode、后端集合、一 catalog 一进程。
- 不得把验收改成必须推送 GitHub Actions 或必须有 GPU runner。
- 不得顺手实现 P1-5 或 D-1 至 D-6。
