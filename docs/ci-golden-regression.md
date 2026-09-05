# Inference CI: what each path is allowed to claim

Weights are **not** in git. GitHub holds code, `conf/weights_manifest.json`
(path + sha256), `conf/ci_hosted_golden.json` (which goldens which path must
run), and `test/golden/` expectations. Bytes live on Hugging Face
(`MaybeShewill-CV/mortred_model_server`) and, for CUDA/TensorRT, on the
maintainer GPU runner at `/opt/mortred-cache/weights`.

A green check must not be read as “every model still works on every
contribution path.” Use the table.

| Path | Runner | Green means | Does **not** mean |
|---|---|---|---|
| Fork PR | GitHub `ubuntu-22.04` (`cpu-profile`) | cpu profile compiled; output contracts passed; **MNN CPU goldens in `conf/ci_hosted_golden.json` hosted set** (sha256 locked, skipped=0): classification, detection, OCR, keypoints, segmentation | TensorRT, CUDA, YOLOv8 engine, full zoo, ORT-CUDA |
| Same-repo PR / push `main`, **no** `MORTRED_HAS_GPU_RUNNER` | GitHub hosted only | Same as fork PR. GPU jobs are **skipped**, not queued | TensorRT / CUDA goldens ran |
| Same-repo PR / push `main`, variable `true` | Hosted + self-hosted GPU | Fork claims **plus** 8 golden cases, skipped=0 | Nightly zoo or cross-backend allclose |
| Schedule / `workflow_dispatch` + variable `true` | GPU runner (`gpu-nightly-full`) | Smoke-8 fail-closed; remaining goldens may skip **if** listed in the skip-inventory artifact | Bit-exact MNN/ONNX/TRT |

The required GitHub check should be **`inference paths`** (job `inference-paths`),
not `gpu golden smoke` by itself. GPU jobs skip when the repository variable
`MORTRED_HAS_GPU_RUNNER` is unset (no self-hosted runner registered) and on
fork PRs. The wrapper treats that skip as success only when `cpu-profile`
succeeded. Do not set `inference paths` required while GPU jobs are still
*queued* waiting for a missing runner.

Changing `src/models/backend/trt_session.cpp` (or any TensorRT-only path) is
**not** proven on a fork PR. Maintainer PRs need `MORTRED_HAS_GPU_RUNNER=true`
so `gpu-pr-gate` runs. Hosted CPU goldens use `device=cpu` via
`force_cpu_backend` and only `mnn`/`onnx` configs; `yolov8_config.toml`
(`type=tensorrt`) stays on the GPU smoke list.

## Enable maintainer GPU jobs

GitHub does not let `GITHUB_TOKEN` list self-hosted runners, so presence is
an explicit flag:

1. Register a Linux x64 runner with labels `self-hosted`, `X64`, `gpu`.
2. Create `/opt/mortred-cache/weights` and `/opt/mortred-cache/3rd_party`.
3. Repo **Settings → Secrets and variables → Actions → Variables** →
   `MORTRED_HAS_GPU_RUNNER` = `true`.
4. Delete the variable (or set it to anything else) to skip GPU jobs again.

## Hosted fork liveness (`cpu-profile`)

Source of truth: `conf/ci_hosted_golden.json`. `scripts/check_hosted_golden.py`
(also invoked from `scripts/check_consistency.py`) rejects a set that is not
on Hugging Face, not tagged `cpu`, or that points at a TensorRT engine.

1. Cache `weights/` keyed by that JSON + `conf/weights_manifest.json`.
2. `scripts/fetch_weights.py --only …` for each hosted weight (stdlib urllib
   if `requests` / `huggingface_hub` are absent).
3. `--check` against the manifest sha256 (mismatch fails the job).
4. `MORTRED_CI_REQUIRE_WEIGHTS=1` on `backend_unittest` (`MnnSession` + config
   tests) and the hosted gtest filter from `--print-gtest-filter`.
5. `scripts/ci_assert_gtest_xml.py` rejects zero tests or any skip.

The earlier `cmake --build --target check` step still allows skip-as-pass for
the rest of the zoo (local-style). Only the explicit hosted step is
fail-closed.

Local developers without weights keep `GTEST_SKIP` (env unset).

YOLOv8 HTTP serving still uses `conf/model/object_detection/yolov8/yolov8_config.toml`
(TensorRT). Hosted detection coverage is **NanoDet MNN**, not that engine.
There is no separate YOLO CPU toml: TensorRT plus `device=cpu` is a
configuration error.

## Maintainer GPU smoke (`gpu-pr-gate`)

Runs only when **all** of these hold:

- Repository variable `MORTRED_HAS_GPU_RUNNER` is exactly `true`.
- `push` to `main`, or a `pull_request` whose **head repo is this repository**
  (not a fork).

Eight cases (`gpu_smoke.cases` in `conf/ci_hosted_golden.json`, must match
`MORTRED_GPU_SMOKE_FILTER` in `.github/workflows/ci.yml`):

| Case | Family |
|---|---|
| `yolov5_detection` | object detection |
| `yolov8_detection` | TensorRT decode + geometry |
| `yolov8_mixed_size_batch_matches_single_runs` | mixed-size batch |
| `nanodet_detection` | anchor-free decode |
| `centerface_detection` | landmarks |
| `dbnet_text_detection` | OCR boxes |
| `mobilenetv2_classification` | scores + `k_score_tol` |
| `fastsam_segmentation` | fingerprint png |

Engine refresh is **`convert_trt_engines.sh --only yolov8` only**. The other
smoke cases use MNN files, which are not in `conf/trt_engines.json`; converting
those names fails the job. Missing cache or a gtest skip fails the job
(`MORTRED_CI_REQUIRE_WEIGHTS=1` + XML audit).

Goldens still force `device=cpu` in the test harness (`force_cpu_backend`).
MNN cases therefore run MNN-CPU even on the GPU box; `type=tensorrt` (YOLOv8)
still uses CUDA. Changing that requires a golden refresh on the runner — a
follow-up, not this gate.

## Nightly (`gpu-nightly-full`)

Same `MORTRED_HAS_GPU_RUNNER=true` gate as the PR smoke (otherwise the
schedule would queue for 120 minutes). Smoke-8 is fail-closed. The rest of
the committed golden zoo runs with `--allow-skips` so an incomplete cache
prints an inventory instead of a fake all-green. The skip list is written to
`gpu-rest-skips.json` and uploaded with the nightly log artifact.
`model_lifecycle_unittest`, `backend_unittest`, and `gateway_e2e_test` still
run via ctest.

Workflow `concurrency` includes `github.event_name` so a push to `main` does
not cancel a running schedule.

## Catalog CI tiers

Every HTTP catalog id in `src/factory/*_task.h` must appear in
`catalog_tiers` inside `conf/ci_hosted_golden.json`:

| Tier | Meaning |
|---|---|
| `hosted` | Fail-closed on GitHub-hosted `cpu-profile` (fork-visible) |
| `gpu-smoke` | Fail-closed on maintainer GPU PR gate; not claimed on forks |
| `nightly` | Allowed to skip on PR; exercised on `gpu-nightly-full` when weights exist |

Adding an HTTP model without a tier fails `python3 scripts/check_consistency.py`.

## Self-hosted runner layout

| Item | Requirement |
|---|---|
| OS | Ubuntu 22.04 LTS |
| NVIDIA driver | >= 535.x |
| CUDA / TensorRT | 11.8 / 8.6.1 — match `3rd_party/` |
| Labels | `self-hosted`, `X64`, `gpu` |
| Concurrency | One runner process per machine |

```
/opt/mortred-cache/
|-- weights/      # HF blobs + this GPU/TRT pair's .engine files
`-- 3rd_party/    # scripts/install_deps.sh --all
```

Jobs symlink those into the checkout. Engines are GPU- and TRT-version
specific: after a driver/TRT bump, wipe `weights/**/*.engine` and rebuild
from ONNX before PR traffic.

Preflight before relying on fail-closed smoke:

```bash
# on the runner, repo root with weights linked
python3 scripts/fetch_weights.py --only yolov8
bash scripts/convert_trt_engines.sh --only yolov8
MORTRED_CI_REQUIRE_WEIGHTS=1 ./build-gpu/bin/model_golden_test \
  --gtest_filter="$MORTRED_GPU_SMOKE_FILTER" \
  --gtest_output=xml:/tmp/gpu-smoke.xml
python3 scripts/ci_assert_gtest_xml.py /tmp/gpu-smoke.xml
```

## Refreshing goldens

Generate on the **same** GPU runner that gates PRs:

```bash
MORTRED_UPDATE_GOLDEN=1 ./build-gpu/bin/model_golden_test
git add test/golden/
```

Update-mode `GTEST_SKIP`s after writing artifacts (not a weight skip). Commit
golden files separately from logic changes.
