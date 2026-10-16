# GPU / gate evidence (non-Docker)

Forks without a GPU self-hosted runner are not blind: maintainers publish a
**declarable evidence manifest** (`docs/ci/gpu_gate_evidence.example.json`) with:

- `product_version` aligned to the single version source (`CMakeLists.txt` `project(VERSION)` / `MORTRED_VERSION`)
- per-artifact **sha256** digests
- **reproduce** commands

## Verify without GPU artifacts

```bash
python3 scripts/check_version_single_source.py --root .
python3 scripts/verify_gpu_gate_evidence.py \
  --manifest docs/ci/gpu_gate_evidence.example.json \
  --allow-missing-artifacts
```

## Emit after a real GPU smoke

```bash
python3 scripts/emit_gpu_gate_evidence.py --version 1.0.0 \
  --artifact name=gpu-smoke,path=path/to/gpu-smoke.xml \
  --out evidence/gpu_gate_evidence.json
```

CI quality gates should run `check_version_single_source.py`. GPU jobs remain
opt-in via `MORTRED_HAS_GPU_RUNNER`; skipped GPU is never claimed as TensorRT green.
