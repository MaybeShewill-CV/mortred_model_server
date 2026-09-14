# SME-16 evidence — convert_trt align to product TRT 10.x pin

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-16-convert-trt10-gate` @ `f8e163e`
- **Host:** Mac + WSL (`DESKTOP-B8MJFUS`)

## Scope

Stack-version convert-side patch (not rebuild-engine latch): refuse TRT major < 10;
`TRT_VERSION_MAJOR` overrides probe only (still fail-closed if < 10); drop TRT 8
`--workspace`/`--buildOnly`; no silent dry-run fallback to 10; parse real
`[TensorRT v100300]` banner.

## Commands / results (WSL)

```text
TRT_VERSION_MAJOR=8 ./scripts/convert_trt_engines.sh --only ddpm_unet_celeba-hq-128 --force --dry-run
→ [ERROR] TensorRT major 8 is not supported (... 10.x only ...)
→ non-zero

# real 3rd_party trtexec (banner TensorRT v100300)
out="$(./scripts/convert_trt_engines.sh --only ddpm_unet_celeba-hq-128 --force --dry-run)"
→ cmd: ... --skipInference --fp16 --memPoolSize=workspace:6GiB
→ no --workspace= / --buildOnly

# unparseable mock trtexec
TRTEXEC="$TMP/trtexec" ... --dry-run   # prints "nope"
→ [ERROR] cannot detect TensorRT major version ...
→ non-zero (no silent fallback)
```

## Outcome

**PASS.** CI also asserts `TRT_VERSION_MAJOR=8` fails and `v100300` banner yields TRT10 flags.
