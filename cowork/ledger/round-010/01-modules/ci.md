# 模块评分

- id: ci
- 权重: 5
- 分数: 8.0
- 读过的路径: `.github/workflows/ci.yml`

## 优势
- PR/push 有 CPU 侧 fail-closed 推理门：`cpu-profile` 拉 sha256 锁定权重、`MORTRED_CI_REQUIRE_WEIGHTS=1` + `ci_assert_gtest_xml.py` 拦 skip，并跑 contract sync / `verify_deployment.sh --basic`（约 L358–459）
- 默认门覆盖单测、valgrind 数量下限、Werror、增量 clang-format、仓库一致性/OpenAPI、TSAN/ASan（`tests` / `quality-gates` / `sanitizers`，约 L54–355）
- 部署工具链有静态门：脚本语法、manifest JSON/YAML、`convert_trt_engines.sh` mock trtexec 契约、actionlint（`deploy-tools`，约 L543–675）
- 维护者自托管 GPU 开启时，`gpu-pr-gate` 对 smoke 子集本身是 fail-closed（`MORTRED_CI_REQUIRE_WEIGHTS=1` + XML 断言，约 L872–881）
- `inference-paths` 把 CPU/GPU 结果收成一个可设 required check 的包装，并打印 tier 摘要（约 L1015–1051）

## 劣势
- 默认 PR 下 GPU/TRT 不是 fail-closed：`gpu-pr-gate` 仅在 `vars.MORTRED_HAS_GPU_RUNNER == 'true'` 且同源 PR/push 时排队，未设变量则 job skipped；`inference-paths` 对 `gpu-pr-gate` 为 `skipped` 直接 `exit 0`，required 检查可在无 TensorRT/CUDA 跑通时仍绿（约 L820–831、L1015–1051；注释亦写明 cpu-profile 不证明 TRT/CUDA，约 L358–363）

## 不计毛病
无

## 依据
- `.github/workflows/ci.yml` 全文（触发、jobs `tests`/`quality-gates`/`sanitizers`/`cpu-profile`/`deploy-tools`/`container-boot`/`full-build-pr`/`gpu-pr-gate`/`gpu-nightly-full`/`inference-paths`）
- 评分口径：交付范围内短板进劣势；默认 PR GPU/TRT 非 fail-closed 按提示计入劣势

## 未读
无
