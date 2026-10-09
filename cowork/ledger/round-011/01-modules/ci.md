# 模块评分

- id: ci
- 权重: 5
- 分数: 9.0
- 读过的路径: `.github/workflows/ci.yml`、`scripts/ci_eval_inference_paths.py`、`scripts/ci_print_tier_summary.py`

## 优势
- OpenAPI 与仓库的一致性仍有两道门：`quality-gates` 的 `python3 scripts/gen_openapi.py --check`（`.github/workflows/ci.yml` 第 264–272 行，检查在第 268 行），以及 `cpu-profile` 用 `contract_dump` 对上 `scripts/check_contract_sync.py`（第 451–456 行）。
- 默认 PR/push 的 CPU 门是 fail-closed：`cpu-profile` 拉 sha256 锁定权重，`MORTRED_CI_REQUIRE_WEIGHTS=1` 加上 `ci_assert_gtest_xml.py`，并跑 `verify_deployment.sh --basic`（第 364–459 行）。同文件写明该作业不证明 TensorRT 或 CUDA（第 358–362 行）。
- 单测、valgrind 数量下限、`-Werror`、增量 clang-format、仓库一致性、TSAN/ASan 仍在 `tests` / `quality-gates` / `sanitizers`（第 54–355 行）。部署脚本语法、manifest、mock `trtexec` 契约和 actionlint 在 `deploy-tools`（第 543–677 行）。
- 自托管 GPU 真跑时，`gpu-pr-gate` 对 smoke 子集 fail-closed（第 874–883 行）。未设 `MORTRED_HAS_GPU_RUNNER`、fork PR，或非 push/同源 PR 时该作业不排队（第 822–833 行）。
- 跳过不会写成 TensorRT 证明：`inference-paths` 在 `if: always()` 下把 `needs.gpu-pr-gate.result` 传给 `scripts/ci_eval_inference_paths.py`（`ci.yml` 第 1022–1036 行），该作业没有 `continue-on-error`。`gpu=skipped` 时脚本打印 `NOT_A_TRT_PROOF` 和 `This result is not a TensorRT or CUDA proof.`，退出码 2；`TRT_PROOF=yes` 只在 `gpu=success` 时出现（`scripts/ci_eval_inference_paths.py` 第 4–7、60–66 行）。PR/push 的 T2 固定为 `NOT RUN`（同文件第 48–50 行）。`t2-tier-notice` 把 `gpu-nightly-full` 的 `skipped` 收成 `skipped`，不收成 `ran`（`ci.yml` 第 998–1020 行）。摘要在 smoke 被跳过时打印同一句非证明声明，T2 被跳过时写明不是 full-zoo（`scripts/ci_print_tier_summary.py` 第 74–83 行）。

## 劣势
无

## 不计毛病
无

## 依据
- C4 两条都在：OpenAPI 检查仍在 `ci.yml` 第 268 行和第 451–456 行；GPU 作业被跳过时，`inference-paths` 走退出码 2 和 `NOT_A_TRT_PROOF`，不打印 `TRT_PROOF=yes`。`.github/` 里其余 workflow 没有 OpenAPI 检查或 TensorRT 证明字样。
- 第 10 轮原句是「`gpu-pr-gate` 被跳过时 `inference-paths` 直接 `exit 0`，required 检查仍绿」。该句已随 P1-6 关闭，现行接线不再是这个行为，不送回劣势。
- 分数按交付可用性记录：CPU、契约、部署静态门和错误日志足够小团队定位 hosted 失败；TensorRT 数值只在自托管 runner 上产生，且跳过被标成非证明。没有未关闭准入项。

## 未读
无
