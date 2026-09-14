# SME-10 evidence — cmake fail-closed ORT/workflow + doc --check (WSL)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-10-cmake-doc-gates` @ `92f0e79`
- **Host:** WSL (`DESKTOP-B8MJFUS`), `/mnt/g/Codex/mortred_model_server`

## A — healthy full-cpu configure

```text
cmake --preset full-cpu -B build/full-cpu-sme10
→ vendored onnxruntime headers: ORT_API_VERSION=29 (.../onnxruntime_c_api.h)
→ Configuring done / Generating done
```

## B — wrong ORT_API_VERSION

```text
sed: ORT_API_VERSION 29 → 18 in onnxruntime_c_api.h
cmake --preset full-cpu -B build/full-cpu-sme10-bad
→ CMake Error ... ORT_API_VERSION=18 (want 29 for libonnxruntime.so.1.29.0)
→ fix: ./scripts/install_deps.sh --cpu --all (or ... --onnxruntime)
→ verify: ./scripts/install_deps.sh --cpu --check
→ Configuring incomplete
# header restored from backup
```

## C — leftover ORT 1.18 soname

```text
touch 3rd_party/libs/libonnxruntime.so.1.18.0
cmake --preset full-cpu -B build/full-cpu-sme10-bad2
→ CMake Error ... leftover ONNX Runtime 1.18 ...
→ fix: ./scripts/install_deps.sh --cpu --all ...
→ verify: ./scripts/install_deps.sh --cpu --check
rm -f 3rd_party/libs/libonnxruntime.so.1.18.0
```

## Outcome

**PASS.** Configure fail-closed with profile-aware fix/verify lines; docs entry
points document `--check` before cmake.
